import argparse
import sys
import json
import torchaudio
import os
import glob
import torch
import torch.nn.functional as F
from tqdm import tqdm

from module.spectrogram import spectrogram
from module.convertor import VoiceConvertor

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputs', default="./inputs/")
parser.add_argument('-o', '--outputs', default="./outputs/")
parser.add_argument('-path', '--model-path', default="model.pt")
parser.add_argument('-p', '--pitch-shift', default=0, type=float)
parser.add_argument('-int', '--intonation', default=1.0, type=float)
parser.add_argument('-t', '--target', default='NONE')
parser.add_argument('-d', '--device', default='cpu')
parser.add_argument('-g', '--gain', default=1.0, type=float)
parser.add_argument('-a', '--alpha', default=1.0, type=float)
parser.add_argument('-k', default=4, type=int)
parser.add_argument('-c', '--chunk', default=65536, type=int)
parser.add_argument('-lib', '--voice-library-path', default="NONE")
parser.add_argument('-noise', '--noise-gain', default=1.0, type=float)
parser.add_argument('-s', '--seed', default=-1, type=int)
parser.add_argument('-wpe', '--world-pitch-estimation', default=False)
parser.add_argument('-norm', '--normalize', default=False, type=bool)

args = parser.parse_args()

device = torch.device(args.device)

model = VoiceConvertor().to(device)
model.load_state_dict(torch.load(args.model_path, map_location=device))

if not os.path.exists(args.outputs):
    os.mkdir(args.outputs)

if args.target != "NONE":
    print("loading target...")
    wf, sr = torchaudio.load(args.target)
    wf = wf.to(device)
    wf = torchaudio.functional.resample(wf, sr, 22050)
    wf = wf / wf.abs().max()
    wf = wf.mean(dim=0, keepdim=True)
    spk = model.encode_spekaer(wf)
else:
    spk = torch.zeros(1, 256, 1).to(device)

if args.seed != -1:
    torch.manual_seed(args.seed)
    spk = torch.randn(1, 256, 1).to(device)

paths = glob.glob(os.path.join(args.inputs, "*"))
for i, path in enumerate(paths):
    wf, sr = torchaudio.load(path)
    wf = wf.to('cpu')
    wf = torchaudio.functional.resample(wf, sr, 22050)
    wf = wf / wf.abs().max()
    wf = wf.mean(dim=0, keepdim=True)
    total_length = wf.shape[1]
    
    wf = torch.cat([wf, torch.zeros(1, (args.chunk * 3))], dim=1)

    wf = wf.unsqueeze(1).unsqueeze(1)
    wf = F.pad(wf, (args.chunk, args.chunk, 0, 0))
    chunks = F.unfold(wf, (1, args.chunk*3), stride=args.chunk)
    chunks = chunks.transpose(1, 2).split(1, dim=1)

    result = []
    with torch.no_grad():
        print(f"converting {path}")
        for chunk in tqdm(chunks):
            chunk = chunk.squeeze(1)

            if chunk.shape[1] < args.chunk:
                chunk = torch.cat([chunk, torch.zeros(1, args.chunk - chunk.shape[1])], dim=1)
            chunk = chunk.to(device)
            
            chunk = model.convert(chunk, spk, args.pitch_shift, alpha=args.alpha)

            chunk = chunk[:, args.chunk:-args.chunk]

            result.append(chunk.to('cpu'))
        wf = torch.cat(result, dim=1)[:, :total_length]
        wf = torchaudio.functional.resample(wf, 22050, sr)
        wf = torchaudio.functional.gain(wf, args.gain)
    wf = wf.cpu().detach()
    if args.normalize:
        wf = wf / wf.abs().max()
    torchaudio.save(os.path.join("./outputs/", f"{os.path.splitext(os.path.basename(path))[0]}.wav"), src=wf, sample_rate=sr)
