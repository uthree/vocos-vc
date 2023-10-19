import torch
import torch.nn as nn
import torch.nn.functional as F
from module.common import ConvNeXt1d, AdaptiveConvNeXt1d, AdaptiveChannelNorm
from module.spectrogram import spectrogram


class F0Estimator(nn.Module):
    def __init__(self, n_fft=1024, max_freq=4096, num_layers=4):
        super().__init__()
        input_channels = n_fft // 2 + 1
        self.input_layer = nn.Conv1d(input_channels, 512, 1)
        self.mid_layers = nn.Sequential(*[ConvNeXt1d() for _ in range(num_layers)])
        self.output_layer = nn.Conv1d(512, max_freq, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.mid_layers(x)
        x = self.output_layer(x)
        return x

    def estimate(self, x):
        dtype = x.dtype
        with torch.no_grad():
            x = self.forward(x)
            f0 = torch.argmax(x, dim=1, keepdim=False).to(x.dtype).unsqueeze(1)
            return f0


class AmplitudeEstimator(nn.Module):
    def __init__(self, hop_length=256):
        super().__init__()
        self.hop_length = hop_length
    
    def forward(self, x):
        x = x.mean(dim=1, keepdim=True)
        return x
    
    def estimate(self, x):
        return self.forward(x)


class ContentEncoder(nn.Module):
    def __init__(self, n_fft=1024, num_layers=4):
        super().__init__()
        input_channels = n_fft // 2 + 1
        self.input_layer = nn.Conv1d(input_channels, 512, 1)
        self.mid_layers = nn.Sequential(*[ConvNeXt1d() for _ in range(num_layers)])

    def forward(self, x):
        x = self.input_layer(x)
        x = self.mid_layers(x)
        mu = x.mean(dim=1, keepdim=True)
        sigma = x.std(dim=1, keepdim=True) + 1e-4
        x = (x - mu) / sigma
        return x


class PhonemeQuantizer(nn.Module):
    def __init__(self, channels=512, num_phonemes=32):
        super().__init__()
        self.codebook = nn.Embedding(num_phonemes, channels)

    def forward(self, x):
        x = x.transpose(1, 2)
        N = x.shape[0]
        y = self.codebook.weight
        y = y.unsqueeze(0)
        y = y.expand(N, y.shape[1], y.shape[2]).transpose(1, 2)
        x = x / (x.std(dim=1, keepdim=True) + 1e-4)
        y = y / (y.std(dim=1, keepdim=True) + 1e-4)
        dists = torch.bmm(x, y)
        indices = torch.argmax(dists, dim=2)
        quantized = self.codebook(indices)
        loss = ((x - quantized.detach()) ** 2).mean() + ((x.detach() - quantized) ** 2).mean()
        return quantized.transpose(1, 2), loss

    def quantize(self, x, alpha=0):
        q, l = self.forward(x)
        return q * (1 - alpha) + x * alpha


class SpeakerEncoder(nn.Module):
    def __init__(self, n_fft=1024, embedding_dim=256, num_layers=4):
        super().__init__()
        input_channels = n_fft // 2 + 1
        self.input_layer = nn.Conv1d(input_channels, 512, 1)
        self.mid_layers = nn.Sequential(*[ConvNeXt1d() for _ in range(num_layers)])
        self.output_layer = nn.Conv1d(512, embedding_dim*2, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.mid_layers(x)
        x = x.mean(dim=2, keepdim=True)
        x = self.output_layer(x)
        mean, logvar = x.chunk(2, dim=1)
        return mean, logvar

    def encode(self, x):
        mean, logvar = self.forward(x)
        return mean


class F0Encoder(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.c1 = nn.Conv1d(1, 256, 1, 1, 0)
        self.c2 = nn.Conv1d(256, embedding_dim, 1, 1, 0)
        self.c1.weight.data.normal_(0, 0.3)

    def forward(self, x):
        x = self.c1(x)
        x = torch.sin(x)
        x = self.c2(x)
        return x


class Generator(nn.Module):
    def __init__(self, n_fft=1024, hop_length=256, content_dim=512, condition_dim=256, num_layers=8):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.pad = nn.ReflectionPad1d([1, 0])

        self.input_layer = nn.Conv1d(content_dim, 512, 1)
        self.f0_encoder = F0Encoder(condition_dim)
        self.amp_encoder = nn.Conv1d(1, condition_dim, 1)
        self.mid_layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.mid_layers.append(
                    AdaptiveConvNeXt1d(512, condition_emb=condition_dim))
        self.last_norm = AdaptiveChannelNorm(512, condition_dim)
        self.output_layer = nn.Conv1d(512, n_fft+2, 1)

    def forward(self, x, f0, amp, spk):
        condition = self.f0_encoder(f0) + self.amp_encoder(amp) + spk
        x = self.input_layer(x)
        condition = self.pad(condition)
        x = self.pad(x)
        for l in self.mid_layers:
            x = l(x, condition)
        x = self.last_norm(x, condition)
        x = self.output_layer(x)

        dtype = x.dtype
        x = x.to(torch.float)
        mag, phase = x.chunk(2, dim=1)
        mag = torch.clamp_max(mag, 6.0)
        mag = torch.exp(mag)
        phase = torch.cos(phase) + 1j * torch.sin(phase)
        s = mag * phase
        out = torch.istft(s, self.n_fft, hop_length=self.hop_length)
        out = out.to(dtype)
        return out

    def magnitude_and_phase(self, x, f0, amp, spk):
        pass # todo: implement it. for ONNX export


# For any-to-one fine-tuning
class ConstantSpeaker(nn.Module):
    def __init__(self, speaker_dim=256):
        super().__init__()
        self.spk = nn.Parameter(torch.randn(1, speaker_dim, 1))

    def forward(self):
        return self.spk


class VoiceConvertor(nn.Module):
    def __init__(self):
        super().__init__()
        self.f0_estimator = F0Estimator()
        self.amp_estimator = AmplitudeEstimator()
        self.generator = Generator()
        self.content_encoder = ContentEncoder()
        self.speaker_encoder = SpeakerEncoder()
        self.quantizer = PhonemeQuantizer()
    
    def encode_spekaer(self, wave):
        spec = spectrogram(wave)
        spk, _ = self.speaker_encoder(spec)
        return spk

    def convert(self, x, spk, pitch_shift=0, alpha=0.0):
        x = spectrogram(x)
        con = self.content_encoder(x)
        amp = self.amp_estimator.estimate(x)
        f0 = self.f0_estimator.estimate(x)

        # Pitch Shift and Intonation Multiply
        pitch = 12 * torch.log2(f0 / 440) - 9 # Convert f0 to pitch
        pitch += pitch_shift # Intonation Multiply
        f0 = 440 * 2 ** ((pitch + 9) / 12) # Convert pitch to f0
        f0[torch.logical_or(f0.isnan(), f0.isinf())] = 0
        
        con = self.quantizer.quantize(con, alpha)
        out_wave = self.generator.forward(con, f0, amp, spk)
        return out_wave
