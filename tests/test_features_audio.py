import io

import numpy as np
import soundfile as sf
from transformers import WhisperFeatureExtractor

from arabic_asr.config import AugmentationConfig
from arabic_asr.data.audio import change_speed, decode_audio_bytes, resample
from arabic_asr.data.augmentation import SpecAugment, WaveformAugmenter
from arabic_asr.data.features import N_FRAMES, LogMelExtractor, pad_features, trim_padding


def _speechlike(seconds=4.2, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * 16000)) / 16000
    return (0.2 * np.sin(2 * np.pi * 220 * t) * (1 + np.sin(2 * np.pi * 3 * t))
            + 0.05 * rng.standard_normal(t.shape)).astype(np.float32)


def test_log_mel_matches_huggingface_whisper_extractor():
    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    audio = _speechlike()
    reference = fe(audio, sampling_rate=16000, return_tensors="np").input_features[0]
    ours, n_valid = LogMelExtractor(mel_filters=fe.mel_filters)(audio)
    assert ours.shape == (80, N_FRAMES)
    assert n_valid == int(np.ceil(len(audio) / 160))
    assert np.abs(ours.numpy() - reference).max() < 1e-3


def test_trim_and_pad_are_lossless(extractor):
    feats, n_valid = extractor(_speechlike(3.0))
    feats = feats.numpy()
    valid, floor = trim_padding(feats)
    assert valid.shape[1] <= n_valid + 2
    assert np.array_equal(pad_features(valid, floor), feats)
    # float16 storage error stays far below anything the model could notice
    restored = pad_features(valid.astype(np.float16).astype(np.float32), floor)
    assert np.abs(restored - feats).max() < 2e-3


def test_resample_and_speed_change_lengths():
    x = np.zeros(48000, dtype=np.float32)
    assert resample(x, 48000, 16000).shape[0] == 16000
    y = np.zeros(16000, dtype=np.float32)
    assert abs(change_speed(y, 1.1).shape[0] - 16000 / 1.1) <= 2
    assert abs(change_speed(y, 0.9).shape[0] - 16000 / 0.9) <= 2


def test_decode_wav_bytes_resamples_to_16k():
    buf = io.BytesIO()
    sf.write(buf, np.zeros((44100, 2), dtype=np.float32), 44100, format="WAV")
    audio = decode_audio_bytes(buf.getvalue())
    assert audio.ndim == 1 and abs(audio.shape[0] - 16000) <= 1


def test_waveform_augmenter_keeps_range_and_is_seeded():
    cfg = AugmentationConfig(speed_perturb_prob=1.0, gain_prob=1.0, noise_prob=1.0)
    aug = WaveformAugmenter(cfg)
    audio = _speechlike(1.0)
    a = aug(audio, np.random.default_rng(1))
    b = aug(audio, np.random.default_rng(1))
    assert np.array_equal(a, b)
    assert np.abs(a).max() <= 1.0


def test_specaugment_only_touches_valid_region():
    cfg = AugmentationConfig(freq_masks=2, freq_mask_width=10, time_masks=4, time_mask_max_ratio=0.1)
    feats = np.random.default_rng(0).standard_normal((80, N_FRAMES)).astype(np.float32)
    out = SpecAugment(cfg)(feats, n_valid=500, rng=np.random.default_rng(3))
    assert out.shape == feats.shape
    assert np.array_equal(out[:, 500:], feats[:, 500:])
    assert not np.array_equal(out[:, :500], feats[:, :500])
