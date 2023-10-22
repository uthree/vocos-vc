import argparse
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio

from tqdm import tqdm

from module.dataset import WaveFileDirectory
from module.spectrogram import spectrogram
from module.convertor import VoiceConvertor
from module.discriminator import Discriminator
from module.common import compute_f0

parser = argparse.ArgumentParser(description="train model")

parser.add_argument('dataset')
parser.add_argument('-p', '--model-path', default='model.pt')
parser.add_argument('-dp', '--discriminator-path', default='discriminator.pt')
parser.add_argument('-d', '--device', default='cpu')
parser.add_argument('-e', '--epoch', default=1000, type=int)
parser.add_argument('-b', '--batch-size', default=4, type=int)
parser.add_argument('-lr', '--learning-rate', default=1e-4, type=float)
parser.add_argument('-len', '--length', default=65536, type=int)
parser.add_argument('-m', '--max-data', default=-1, type=int)
parser.add_argument('-fp16', default=False, type=bool)
parser.add_argument('-gacc', '--gradient-accumulation', default=1, type=int)
parser.add_argument('--feature-matching', default=2, type=float)
parser.add_argument('--mel', default=45, type=float)
parser.add_argument('--content', default=10, type=float)
parser.add_argument('--kl', default=1, type=float)
parser.add_argument('-wpe', '--world-pitch-estimation', default=False, type=bool)
parser.add_argument('-ft', '--finetune', default=False, type=bool)

args = parser.parse_args()


def inference_mode(model):
    for param in model.parameters():
        param.requires_grad = False


def load_or_init_models(device=torch.device('cpu')):
    dis = Discriminator().to(device)
    con = VoiceConvertor().to(device)
    inference_mode(con.f0_estimator)
    if os.path.exists(args.discriminator_path):
        dis.load_state_dict(torch.load(args.discriminator_path, map_location=device))
    if os.path.exists(args.model_path):
        con.load_state_dict(torch.load(args.model_path, map_location=device))
    return con, dis


def save_models(con, dis):
    print("Saving Models...")
    torch.save(con.state_dict(), args.model_path)
    torch.save(dis.state_dict(), args.discriminator_path)
    print("complete!")


def cut_center(x):
    length = x.shape[2]
    center = length // 2
    size = length // 4
    return x[:, :, center-size:center+size]

def cut_center_wav(x):
    length = x.shape[1]
    center = length // 2
    size = length // 4
    return x[:, center-size:center+size]

device = torch.device(args.device)
con, D = load_or_init_models(device)
Ec = con.content_encoder
F0E = con.f0_estimator
AE = con.amp_estimator
G = con.generator
Es = con.speaker_encoder

if args.finetune:
    inference_mode(Ec)
    inference_mode(Es)

ds = WaveFileDirectory(
        [args.dataset],
        length=args.length,
        max_files=args.max_data
        )

dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)

scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

OptG = optim.AdamW(con.parameters(), lr=args.learning_rate, betas=(0.9, 0.99))
OptD = optim.AdamW(D.parameters(), lr=args.learning_rate, betas=(0.9, 0.99))

SchedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(OptG, 5000)
SchedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(OptD, 5000)

mel = torchaudio.transforms.MelSpectrogram(22050, n_fft=1024, n_mels=80).to(device)

step_count = 0

def log_mel(x):
    return torch.log(torch.clamp_min(mel(x), 1e-5))

for epoch in range(args.epoch):
    tqdm.write(f"Epoch #{epoch}")
    bar = tqdm(total=len(ds))
    for batch, wave in enumerate(dl):
        wave = wave.to(device) * (torch.rand(wave.shape[0], 1, device=device) * 1.5 + 0.25)
        spec = spectrogram(wave)

        N = wave.shape[0]

        # Train G.
        OptG.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.fp16):
            with torch.no_grad():
                if args.world_pitch_estimation:
                    f0 = compute_f0(wave)
                else:
                    f0 = F0E.estimate(spec)
            content = Ec(spec)
            mean, logvar = Es(spec)
            spk_src = mean + torch.exp(logvar) * torch.randn_like(logvar)
            if args.finetune:
                spk_src = spk_src * 0
            spk_tgt = spk_src.roll(1, dims=0)
            f0_tgt = f0 * (torch.rand(N, 1, 1, device=device) * 1.5 + 0.5)
            amp = AE(spec)
            wave_recon = G(content, f0, amp, spk_src)
            wave_fake = G(content, f0_tgt, amp, spk_tgt)
            loss_mel = (log_mel(wave) - log_mel(wave_recon)).abs().mean()
            loss_kl = (-1 - logvar + torch.exp(logvar) + mean ** 2).mean()
            spec_fake = spectrogram(wave_fake)
            loss_con = (content - Ec(spec_fake)).abs().mean()
            loss_feat = D.feat_loss(cut_center_wav(wave_recon), cut_center_wav(wave))

            loss_adv = 0
            logits = D.logits(cut_center_wav(wave_fake)) + D.logits(cut_center_wav(wave_recon))
            for logit in logits:
                loss_adv += (logit ** 2).mean()
            loss_g = loss_adv + loss_mel * args.mel + loss_kl * args.kl + args.content * loss_con + loss_feat * args.feature_matching

        scaler.scale(loss_g).backward()
        scaler.step(OptG)

        # Train D.
        OptD.zero_grad()
        wave_fake = wave_fake.detach()
        wave_recon = wave_recon.detach()
        with torch.cuda.amp.autocast(enabled=args.fp16):
            logits_fake = D.logits(cut_center_wav(wave_fake)) + D.logits(cut_center_wav(wave_recon))
            logits_real = D.logits(wave)
            loss_d = 0
            for logit in logits_real:
                loss_d += (logit ** 2).mean()
            for logit in logits_fake:
                loss_d += ((logit - 1) ** 2).mean()
        scaler.scale(loss_d).backward()
        scaler.step(OptD)

        scaler.update()
        SchedulerD.step()
        SchedulerG.step()

        step_count += 1
        
        tqdm.write(f"Step {step_count}, D: {loss_d.item():.4f}, Adv.: {loss_adv.item():.4f}, Mel.: {loss_mel.item():.4f}, Feat.: {loss_feat.item():.4f}, Con.: {loss_con.item():.4f}, K.L.: {loss_kl.item():.4f}")
        bar.update(N)

        if batch % 300 == 0:
            save_models(con, D)

print("Training Complete!")
save_models(con, D)

