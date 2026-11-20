"""Shared fixtures: the Whisper tokenizer/mel filters and a tiny synthetic cache.

The synthetic cache mimics both corpus types (raw-audio MSA and feature-only MASC) so the
whole data path can be tested without downloading the real datasets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from arabic_asr.config import FilterConfig
from arabic_asr.data.audio import float_to_int16
from arabic_asr.data.cache import CacheWriter, save_meta
from arabic_asr.data.features import LogMelExtractor, trim_padding
from arabic_asr.data.filtering import assign_filter_reasons
from arabic_asr.data.prepare import _derive_columns, load_tokenizer

SENTENCES = [
    "مرحبا بكم في الدرس",
    "هو يحبّها و هي تحبّه أيضا.",
    "اليوم الجو جميل جدا",
    "ذهبت إلى المدرسة صباحا",
    "♪ وأول حضن إنت حسيته ♪",
    "c'est vraiment bien يا خويا",
    "شكرا جزيلا لكم",
    "أين الكتاب الذي اشتريته؟",
]


@pytest.fixture(scope="session")
def tokenizer():
    return load_tokenizer("openai/whisper-small")


@pytest.fixture(scope="session")
def extractor():
    return LogMelExtractor(feature_extractor_name="openai/whisper-small")


def _tone(seconds: float, freq: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * 16000)) / 16000.0
    return (0.3 * np.sin(2 * np.pi * freq * t) + 0.02 * rng.standard_normal(t.shape)).astype(np.float32)


def build_tiny_cache(root, tokenizer, extractor, n_rows: int = 8):
    """Create cache/{msa,egyptian}/{train,test} with ``n_rows`` synthetic utterances each."""
    for source, kind in (("msa", "audio"), ("egyptian", "features")):
        for split in ("test", "train"):
            writer = CacheWriter(root / source / split, kind)
            for i in range(n_rows):
                text = SENTENCES[i % len(SENTENCES)]
                audio = _tone(1.0 + 0.4 * i, 200 + 30 * i, seed=i)
                extra = {"text_raw": text, "hf_file": "synthetic", "hf_row": i}
                if kind == "audio":
                    writer.add_audio(float_to_int16(audio), client_id=f"spk{i % 4}", gender="male" if i % 2 else "female",
                                     age="twenties", accent="", **extra)
                else:
                    feats, _ = extractor(audio)
                    valid, floor = trim_padding(feats.numpy())
                    writer.add_features(valid, floor, **extra)
            meta = writer.close()
            meta = _derive_columns(meta, kind, source, split, tokenizer)
            if split == "train":
                test_texts = pd.read_parquet(root / source / "test" / "meta.parquet")["text"].tolist()
                # Disable leakage check here: the synthetic train/test share sentences on purpose.
                meta = assign_filter_reasons(meta, FilterConfig(leakage_min_words=99, speaking_rate_iqr_k=0),
                                             test_texts=test_texts)
            else:
                meta["filter_reason"] = ""
            info = {"kind": kind, "n_mels": 80, "source": source, "split": split, "rows": len(meta)}
            save_meta(root / source / split, meta, info)
    return root


@pytest.fixture()
def tiny_cache(tmp_path, tokenizer, extractor):
    return build_tiny_cache(tmp_path / "cache", tokenizer, extractor)


@pytest.fixture()
def tiny_model_dir(tmp_path):
    """A randomly initialised 2-layer Whisper with the real vocabulary (no download)."""
    import torch
    from transformers import GenerationConfig, WhisperConfig, WhisperForConditionalGeneration

    config = WhisperConfig(
        vocab_size=51865, num_mel_bins=80, d_model=32, encoder_layers=1, decoder_layers=1,
        encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=64, decoder_ffn_dim=64,
        max_source_positions=1500, max_target_positions=448,
        pad_token_id=50257, bos_token_id=50257, eos_token_id=50257, decoder_start_token_id=50258,
    )
    torch.manual_seed(0)
    model = WhisperForConditionalGeneration(config)
    model.generation_config = GenerationConfig.from_pretrained("openai/whisper-small")
    path = tmp_path / "tiny-whisper"
    model.save_pretrained(path)
    return path
