"""Audio loading and resampling (1C1: "audio loading and resampling").

Everything is converted to 16 kHz mono float32 in [-1, 1], the input Whisper expects.
Decoding uses libsndfile (WAV, FLAC, OGG, MP3); other formats (m4a, webm from a browser
microphone) fall back to librosa/audioread if installed.
"""

from __future__ import annotations

import io
import os
from fractions import Fraction
from math import gcd
from typing import Tuple, Union

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16_000


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    # soundfile returns (frames, channels)
    return audio.mean(axis=1)


def resample(audio: np.ndarray, sr_in: int, sr_out: int = TARGET_SR) -> np.ndarray:
    """Polyphase resampling with an anti-aliasing filter (e.g. 48 kHz -> 16 kHz is 1:3)."""
    if sr_in == sr_out:
        return audio.astype(np.float32, copy=False)
    g = gcd(int(sr_in), int(sr_out))
    up, down = int(sr_out) // g, int(sr_in) // g
    return resample_poly(audio, up, down).astype(np.float32)


def change_speed(audio: np.ndarray, factor: float) -> np.ndarray:
    """Speed perturbation (tempo and pitch both change, as in Kaldi/sox ``speed``).

    factor > 1 makes speech faster/shorter. Implemented by resampling with the rational
    approximation of the factor, then interpreting the result at the original rate.
    """
    if abs(factor - 1.0) < 1e-6:
        return audio
    frac = Fraction(factor).limit_denominator(100)
    return resample_poly(audio, frac.denominator, frac.numerator).astype(np.float32)


def _decode_with_fallback(source: Union[str, io.BytesIO]) -> Tuple[np.ndarray, int]:
    try:
        audio, sr = sf.read(source, dtype="float32", always_2d=False)
        return audio, int(sr)
    except Exception as sf_error:  # noqa: BLE001 - fall back to librosa for exotic formats
        try:
            import librosa  # type: ignore

            if isinstance(source, io.BytesIO):
                source.seek(0)
            audio, sr = librosa.load(source, sr=None, mono=False)
            audio = audio.T if audio.ndim == 2 else audio
            return audio.astype(np.float32), int(sr)
        except Exception as lr_error:  # noqa: BLE001
            raise RuntimeError(
                f"Could not decode audio ({sf_error}); librosa fallback also failed ({lr_error}). "
                "Convert the file to WAV/FLAC/MP3 or install librosa + ffmpeg."
            ) from lr_error


def decode_audio_bytes(data: bytes, target_sr: int = TARGET_SR) -> np.ndarray:
    audio, sr = _decode_with_fallback(io.BytesIO(data))
    return prepare_waveform(audio, sr, target_sr)


def load_audio(path: Union[str, os.PathLike], target_sr: int = TARGET_SR) -> np.ndarray:
    audio, sr = _decode_with_fallback(str(path))
    return prepare_waveform(audio, sr, target_sr)


def prepare_waveform(audio: np.ndarray, sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    """Mono, resampled, float32, clipped to [-1, 1]."""
    audio = to_mono(np.asarray(audio, dtype=np.float32))
    audio = resample(audio, sr, target_sr)
    return np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)


def float_to_int16(audio: np.ndarray) -> np.ndarray:
    return (np.clip(audio, -1.0, 1.0) * 32767.0).round().astype(np.int16)


def int16_to_float(audio: np.ndarray) -> np.ndarray:
    return audio.astype(np.float32) / 32767.0
