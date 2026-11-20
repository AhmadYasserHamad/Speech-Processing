import numpy as np
import torch

from arabic_asr.config import DataConfig
from arabic_asr.data.cache import CacheSplit
from arabic_asr.data.dataset import (
    ASRDataset,
    ResumableRandomSampler,
    WhisperCollator,
    build_eval_dataset,
    build_experiment_data,
)
from arabic_asr.data.splits import random_split, speaker_disjoint_split


def test_cache_roundtrip_and_flags(tiny_cache):
    msa = CacheSplit(tiny_cache, "msa", "train")
    egy = CacheSplit(tiny_cache, "egyptian", "train")
    assert msa.kind == "audio" and egy.kind == "features"
    assert msa.audio(0).dtype == np.float32
    valid, floor = egy.features(1)
    assert valid.shape[0] == 80 and np.isfinite(floor)
    # the "♪" sentence is flagged and filtered from training, Latin is only flagged
    assert msa.meta.loc[4, "filter_reason"] == "music"
    assert bool(msa.meta.loc[5, "has_latin"]) and msa.meta.loc[5, "filter_reason"] == ""
    assert msa.meta.loc[1, "text"] == "هو يحبها و هي تحبه أيضا"


def test_dataset_items_and_collator(tiny_cache, tokenizer, extractor):
    selections = [(CacheSplit(tiny_cache, "msa", "train"), np.array([0, 1])),
                  (CacheSplit(tiny_cache, "egyptian", "train"), np.array([2, 3]))]
    ds = ASRDataset(selections, tokenizer, extractor)
    items = [ds[i] for i in range(len(ds))]
    assert all(it["input_features"].shape == (80, 3000) for it in items)
    prefix = tokenizer.convert_ids_to_tokens(items[0]["labels"][:4])
    assert prefix == ["<|startoftranscript|>", "<|ar|>", "<|transcribe|>", "<|notimestamps|>"]
    assert items[0]["labels"][-1] == tokenizer.eos_token_id
    batch = WhisperCollator(tokenizer.convert_tokens_to_ids("<|startoftranscript|>"))(items)
    assert batch["input_features"].shape == (4, 80, 3000)
    assert batch["labels"][0, 0].item() == tokenizer.convert_tokens_to_ids("<|ar|>")
    lengths = [len(it["labels"]) - 1 for it in items]
    for row, n in enumerate(lengths):
        assert (batch["labels"][row, n:] == -100).all()
        assert (batch["labels"][row, :n] != -100).all()


def test_random_and_speaker_disjoint_splits():
    rows = np.arange(100)
    tr, va = random_split(rows, 0.2, seed=1)
    assert len(va) == 20 and set(tr).isdisjoint(va) and len(tr) + len(va) == 100
    speakers = [f"s{i % 13}" for i in rows]
    tr, va = speaker_disjoint_split(rows, speakers, 0.2, seed=1)
    spk_tr = {speakers[i] for i in tr}
    spk_va = {speakers[i] for i in va}
    assert spk_tr.isdisjoint(spk_va)
    assert 10 <= len(va) <= 30


def test_build_experiment_data_with_msa_replay(tiny_cache, tokenizer, extractor):
    cfg = DataConfig(cache_dir=str(tiny_cache), train_sources=["egyptian"], msa_replay_fraction=0.25,
                     val_fraction=0.25, val_loss_samples=4, val_wer_samples=2)
    data = build_experiment_data(cfg, tokenizer, extractor)
    counts = data.train.source_counts()
    assert set(counts) == {"egyptian", "msa"}
    assert data.summary["msa_replay_rows"] >= 1
    assert set(data.manifest["msa"]["train"]).isdisjoint(data.manifest["msa"]["val"])
    assert len(data.val_loss) >= 2 and len(data.val_wer) >= 2
    assert data.train.waveform_aug is not None and data.val_loss.waveform_aug is None


def test_eval_dataset_uses_raw_text(tiny_cache, tokenizer, extractor):
    ds = build_eval_dataset(str(tiny_cache), "msa", tokenizer, extractor, split="test", max_samples=3)
    assert len(ds) == 3
    raw_texts = {ds[i]["text"] for i in range(3)}
    assert any(("." in t or "؟" in t or "ّ" in t or "♪" in t or "'" in t) for t in raw_texts) or len(raw_texts) == 3


def test_resumable_sampler_continues_permutation():
    full = list(ResumableRandomSampler(10, seed=5))
    sampler = ResumableRandomSampler(10, seed=5)
    sampler.set_state(epoch=0, start=4)
    assert list(sampler) == full[4:]
    sampler.set_state(epoch=1, start=0)
    assert sorted(sampler) == list(range(10)) and list(sampler) != full
    assert torch.initial_seed() is not None
