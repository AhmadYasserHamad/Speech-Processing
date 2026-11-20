"""Whisper log-Mel feature extraction (1C1: "feature extraction").

A PyTorch re-implementation of Whisper's front end (Radford et al., 2023; same algorithm as
``openai/whisper`` ``audio.log_mel_spectrogram`` and ``WhisperFeatureExtractor``):

1. pad or trim the 16 kHz waveform to 30 s (480,000 samples);
2. STFT with a 400-sample (25 ms) Hann window and 160-sample (10 ms) hop -> 3,000 frames;
3. power spectrum -> 80 (or 128 for large-v3) Slaney mel bands;
4. log10, floor at (max - 8) i.e. an 80 dB dynamic range, then scale with (x + 4) / 4.

Step 4 is the *feature normalisation* the pre-trained model expects; it must match exactly,
which ``tests/test_features.py`` checks against the Hugging Face implementation.

The module also provides lossless trimming of the zero-padding region. In Whisper features
every padded frame equals the per-utterance floor value, so we store only the real frames
plus that floor and rebuild the 3,000-frame input on the fly. For MASC this shrinks the
authors' precomputed features several-fold.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

SAMPLE_RATE = 16_000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_SECONDS = 30
N_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS  # 480,000
N_FRAMES = N_SAMPLES // HOP_LENGTH  # 3,000


def load_mel_filters(feature_extractor_name: str = "openai/whisper-small") -> np.ndarray:
    """Mel filter bank (n_freq=201, n_mels) taken from the model's own feature extractor."""
    from transformers import WhisperFeatureExtractor

    fe = WhisperFeatureExtractor.from_pretrained(feature_extractor_name)
    return np.asarray(fe.mel_filters, dtype=np.float32)


class LogMelExtractor:
    """Callable: waveform (16 kHz mono float) -> (features[n_mels, 3000], n_valid_frames)."""

    def __init__(self, mel_filters: Optional[np.ndarray] = None, feature_extractor_name: str = "openai/whisper-small"):
        if mel_filters is None:
            mel_filters = load_mel_filters(feature_extractor_name)
        self.mel_filters = torch.from_numpy(np.ascontiguousarray(np.asarray(mel_filters).T, dtype=np.float32))  # (n_mels, 201)
        self.n_mels = int(self.mel_filters.shape[0])
        self.window = torch.hann_window(N_FFT)

    @staticmethod
    def valid_frames(num_samples: int) -> int:
        return int(min(N_FRAMES, max(1, -(-num_samples // HOP_LENGTH))))

    def __call__(self, audio) -> Tuple[torch.Tensor, int]:
        waveform = torch.as_tensor(np.asarray(audio, dtype=np.float32))
        n_valid = self.valid_frames(int(waveform.shape[-1]))
        if waveform.shape[-1] >= N_SAMPLES:
            waveform = waveform[:N_SAMPLES]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, N_SAMPLES - waveform.shape[-1]))
        stft = torch.stft(waveform, N_FFT, HOP_LENGTH, window=self.window, return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2  # drop the last frame -> exactly 3,000
        mel_spec = self.mel_filters @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec, n_valid


def trim_padding(features: np.ndarray) -> Tuple[np.ndarray, float]:
    """Split Whisper features (n_mels, 3000) into the non-padding frames and the floor value.

    Trailing frames whose every bin equals the global minimum are padding (or digital
    silence, which is indistinguishable and therefore also restored exactly).
    """
    floor = float(features.min())
    is_floor_frame = np.all(features == floor, axis=0)
    not_floor = np.flatnonzero(~is_floor_frame)
    length = int(not_floor[-1]) + 1 if not_floor.size else 1
    return features[:, :length], floor


def pad_features(valid: np.ndarray, floor: float, total_frames: int = N_FRAMES) -> np.ndarray:
    """Inverse of :func:`trim_padding`: (n_mels, L) -> (n_mels, total_frames) float32."""
    n_mels, length = valid.shape
    out = np.full((n_mels, total_frames), floor, dtype=np.float32)
    out[:, : min(length, total_frames)] = valid[:, :total_frames]
    return out
