import torch
import torchaudio
from torch import nn


class AddNoiseTransform(nn.Module):
    def __init__(self, snr=20):
        super().__init__()
        self.snr = snr

    def __call__(self, waveform):
        # Генерируем шум той же формы что и входной сигнал
        noise = torch.randn_like(waveform)
        # Применяем AddNoise с фиксированным SNR
        noisy_waveform = torchaudio.functional.add_noise(
            waveform=waveform,
            noise=noise,
            snr=torch.tensor([self.snr])
        )
        return noisy_waveform
