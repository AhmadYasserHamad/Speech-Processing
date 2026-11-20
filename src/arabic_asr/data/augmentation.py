"""Training-time data augmentation (1C1: "normalization and augmentation").

Paper 3 used no augmentation. We add two kinds, applied on the fly to training batches only:

* Waveform augmentation for sources that ship raw audio (MSA): speed perturbation
  (0.9x / 1.1x), random gain and additive Gaussian noise at a random SNR.
* SpecAugment (Park et al., 2019) on the log-Mel features of every source, as Paper 2 did.
  It is the only option for MASC, whose audio is not released (features only). Masks are
  restricted to the real (non-padding) frames, unlike Hugging Face Whisper's built-in
  masking, which would waste most masks on the 30 s padding.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from arabic_asr.config import AugmentationConfig
from arabic_asr.data.audio import change_speed


class WaveformAugmenter:
    def __init__(self, cfg: AugmentationConfig):
        self.cfg = cfg

    def __call__(self, audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        cfg = self.cfg
        out = audio
        if cfg.speed_factors and rng.random() < cfg.speed_perturb_prob:
            out = change_speed(out, float(rng.choice(cfg.speed_factors)))
        if rng.random() < cfg.gain_prob:
            gain_db = rng.uniform(cfg.gain_db[0], cfg.gain_db[1])
            out = out * (10.0 ** (gain_db / 20.0))
        if rng.random() < cfg.noise_prob:
            snr_db = rng.uniform(cfg.noise_snr_db[0], cfg.noise_snr_db[1])
            signal_power = float(np.mean(out ** 2)) + 1e-10
            noise_power = signal_power / (10.0 ** (snr_db / 10.0))
            out = out + rng.normal(0.0, np.sqrt(noise_power), size=out.shape)
        return np.clip(out, -1.0, 1.0).astype(np.float32)


class SpecAugment:
    """Frequency and time masking on (n_mels, frames) features, inside the valid region."""

    def __init__(self, cfg: AugmentationConfig):
        self.cfg = cfg

    def __call__(self, features: np.ndarray, n_valid: int, rng: np.random.Generator,
                 fill_value: Optional[float] = None) -> np.ndarray:
        cfg = self.cfg
        if rng.random() >= cfg.specaugment_prob:
            return features
        feats = features.copy()
        n_mels = feats.shape[0]
        n_valid = int(max(1, min(n_valid, feats.shape[1])))
        if fill_value is None:
            fill_value = float(feats[:, :n_valid].mean())
        for _ in range(cfg.freq_masks):
            width = int(rng.integers(0, cfg.freq_mask_width + 1))
            if width == 0:
                continue
            start = int(rng.integers(0, max(1, n_mels - width + 1)))
            feats[start : start + width, :n_valid] = fill_value
        max_width = max(1, int(cfg.time_mask_max_ratio * n_valid))
        for _ in range(cfg.time_masks):
            width = int(rng.integers(0, max_width + 1))
            if width == 0:
                continue
            start = int(rng.integers(0, max(1, n_valid - width + 1)))
            feats[:, start : start + width] = fill_value
        return feats
