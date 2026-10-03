"""Causal GRU enhancement, runtime-equivalent WOLA, and training objectives."""
import math

import torch
from torch import nn
from torch.nn import functional as F

N, H, K = 512, 160, 257


class StreamingDenoiseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_in = nn.Linear(K, 128)
        self.norm = nn.LayerNorm(128)
        self.act = nn.PReLU()
        self.gru = nn.GRU(128, 128, num_layers=2, batch_first=True)
        self.fc_out = nn.Linear(128, 2*K)
        nn.init.zeros_(self.fc_out.weight)
        with torch.no_grad():
            self.fc_out.bias[:K].fill_(math.atanh(.5))
            self.fc_out.bias[K:].zero_()

    def forward(self, x_mag, h_in):
        features = self.act(self.norm(self.fc_in(x_mag)))
        output, h_out = self.gru(features, h_in)
        mask = 2 * torch.tanh(self.fc_out(output))
        return mask[..., :K], mask[..., K:], h_out


def windows(device, dtype):
    n = torch.arange(N, device=device, dtype=dtype)
    analysis = torch.sin(math.pi*(n+.5)/N)
    phase = torch.zeros(H, device=device, dtype=dtype)
    for offset in range(0, N, H):
        length = min(H, N-offset)
        phase[:length] += analysis[offset:offset+length].square()
    return analysis, analysis / phase[torch.arange(N, device=device) % H]


def spectra(wave):
    """[B,L] -> [B,T,257], with causal left padding and three flush hops."""
    length = wave.shape[-1]
    right = (-length) % H + 3*H
    signal = F.pad(wave, (N-H, right))
    analysis, synthesis = windows(wave.device, wave.dtype)
    frames = signal.unfold(-1, N, H)
    return torch.fft.rfft(frames*analysis, n=N), synthesis, length


def reconstruct(spectrum, synthesis, length):
    spectrum = spectrum.clone()
    spectrum[..., 0] = spectrum[..., 0].real
    spectrum[..., -1] = spectrum[..., -1].real
    frames = torch.fft.irfft(spectrum, n=N) * synthesis
    total = (frames.shape[1]-1)*H + N
    signal = F.fold(frames.transpose(1, 2), output_size=(1, total),
                    kernel_size=(1, N), stride=(1, H)).squeeze(1).squeeze(1)
    return signal[:, N-H:N-H+length]


def enhance(model, wave, streaming=False):
    spec, synthesis, length = spectra(wave)
    feature = torch.log10(spec.real.square()+spec.imag.square()+1e-7)
    state = wave.new_zeros(2, wave.shape[0], 128)
    if streaming:
        reals, imags = [], []
        for frame in feature.split(1, dim=1):
            real, imag, state = model(frame, state)
            reals.append(real)
            imags.append(imag)
        real, imag = torch.cat(reals, 1), torch.cat(imags, 1)
    else:
        real, imag, state = model(feature, state)
    return reconstruct(spec*torch.complex(real, imag), synthesis, length)


def si_sdr(target, estimate):
    target = target-target.mean(dim=-1, keepdim=True)
    estimate = estimate-estimate.mean(dim=-1, keepdim=True)
    projection = (estimate*target).sum(-1, keepdim=True) / (target.square().sum(-1, keepdim=True)+1e-7) * target
    residual = estimate-projection
    return 10*torch.log10((projection.square().sum(-1)+1e-7)/(residual.square().sum(-1)+1e-7))


def magnitude_loss(target, estimate):
    losses = []
    for size in (256, 512, 1024):
        window = torch.hann_window(size, periodic=True, device=target.device, dtype=target.dtype)
        def stft(x):
            return torch.stft(x, size, hop_length=size//4, window=window,
                              center=False, return_complex=True).abs()
        reference, output = stft(target), stft(estimate)
        convergence = torch.linalg.vector_norm(reference-output, dim=(-2,-1)) / (torch.linalg.vector_norm(reference, dim=(-2,-1))+1e-7)
        log_error = (torch.log(reference+1e-7)-torch.log(output+1e-7)).abs().mean(dim=(-2,-1))
        losses.append((convergence+log_error)/2)
    return torch.stack(losses).mean(0)


def objective(target, estimate):
    magnitude = magnitude_loss(target, estimate)
    active = torch.sqrt(target.square().mean(-1)) >= 1e-5
    # Silent windows contribute magnitude only, avoiding undefined SI-SDR.
    return ((-si_sdr(target[active], estimate[active])).sum() if active.any() else magnitude.new_zeros(())) / target.shape[0] + .1*magnitude.mean()
