"""
speech_features.py
--------------------
Estrazione delle feature MFCC tramite SpeechBrain per il progetto
SF-KWS su STM32 BlueCoin (Fase 2 della roadmap).

Parametri scelti (vedi MFCCConfig) per due motivi:
1. Confrontabilita' con Tab. 7 della Fonte 1 (Zhang et al., DS-CNN-S):
   finestra 40 ms, hop 20 ms, 40 filtri Mel, 10 coefficienti MFCC statici.
2. Compatibilita' futura con CMSIS-DSP (Fase 6): n_fft e' una potenza di 2
   (arm_rfft_fast_f32 richiede lunghezze potenza di 2).

Niente delta/delta-delta e niente context stacking (deltas=False,
context=False): il contesto temporale lo impara direttamente la DS-CNN
tramite le convoluzioni 2D, come nell'architettura originale di Hello Edge.
"""

from dataclasses import dataclass

import numpy as np
import torch
from speechbrain.lobes.features import MFCC


@dataclass
class MFCCConfig:
    sample_rate: int = 16000
    win_length_ms: float = 40.0     # 640 campioni @16kHz
    hop_length_ms: float = 20.0     # 320 campioni @16kHz
    n_fft: int = 1024               # >= 640, potenza di 2 (CMSIS-DSP friendly)
    n_mels: int = 40
    n_mfcc: int = 10
    clip_seconds: float = 1.0       # durata fissa dei clip GSCD


class SpeechBrainMFCCExtractor:
    """Wrapper attorno a speechbrain.lobes.features.MFCC.

    Uso:
        extractor = SpeechBrainMFCCExtractor()
        feats = extractor(waveform_np)   # -> np.ndarray [T, n_mfcc], float32
        extractor.feature_shape          # (T, n_mfcc), calcolata al primo uso

    Gira volutamente su CPU (device="cpu" di default): l'estrazione MFCC e'
    economica e tenerla fuori dalla GPU evita che TensorFlow (training del
    DS-CNN) e PyTorch (SpeechBrain) si contendano la stessa VRAM.
    """

    def __init__(self, config: MFCCConfig = None, device: str = "cpu"):
        self.config = config or MFCCConfig()
        self.device = torch.device(device)
        self._mfcc = MFCC(
            deltas=False,
            context=False,
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            n_mels=self.config.n_mels,
            n_mfcc=self.config.n_mfcc,
            win_length=self.config.win_length_ms,
            hop_length=self.config.hop_length_ms,
        ).to(self.device)
        self._mfcc.eval()
        self._n_samples = int(self.config.sample_rate * self.config.clip_seconds)
        self._feature_shape = None

    @property
    def feature_shape(self):
        """(T, n_mfcc) calcolata dinamicamente: evita di dover ricalcolare
        a mano il numero di frame ogni volta che si cambiano i parametri."""
        if self._feature_shape is None:
            probe = np.zeros(self._n_samples, dtype=np.float32)
            self._feature_shape = self(probe).shape
        return self._feature_shape

    def _fix_length(self, wav: np.ndarray) -> np.ndarray:
        """Forza il clip a durata fissa: pad con zeri o taglio (i file GSCD
        sono gia' <=1s, ma meglio essere robusti)."""
        if len(wav) < self._n_samples:
            wav = np.pad(wav, (0, self._n_samples - len(wav)))
        elif len(wav) > self._n_samples:
            wav = wav[: self._n_samples]
        return wav

    @torch.no_grad()
    def __call__(self, wav: np.ndarray) -> np.ndarray:
        """wav: np.ndarray 1D float32, gia' a sample_rate Hz, range [-1, 1]."""
        wav = self._fix_length(np.asarray(wav, dtype=np.float32))
        t = torch.from_numpy(wav).unsqueeze(0).to(self.device)  # [1, N]
        feats = self._mfcc(t)                                   # [1, T, n_mfcc]
        return feats.squeeze(0).cpu().numpy()


if __name__ == "__main__":
    extractor = SpeechBrainMFCCExtractor()
    print("Config:", extractor.config)
    print("Feature shape (T, n_mfcc):", extractor.feature_shape)
