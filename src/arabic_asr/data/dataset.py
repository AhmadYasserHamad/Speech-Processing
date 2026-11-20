"""PyTorch datasets, collator and a resumable sampler.

``ASRDataset`` mixes any number of cache splits. For each utterance it returns Whisper-ready
log-Mel features (computed on the fly from audio, or rebuilt from stored features), the
token ids of the normalised transcript, and some metadata.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from arabic_asr.config import AugmentationConfig, DataConfig
from arabic_asr.data.augmentation import SpecAugment, WaveformAugmenter
from arabic_asr.data.cache import CacheSplit
from arabic_asr.data.features import LogMelExtractor, pad_features
from arabic_asr.data.splits import train_val_rows
from arabic_asr.utils import get_logger

Selection = Tuple[CacheSplit, np.ndarray]


class ASRDataset(Dataset):
    def __init__(
        self,
        selections: Sequence[Selection],
        tokenizer,
        extractor: LogMelExtractor,
        text_level: str = "light",
        augment: Optional[AugmentationConfig] = None,
    ):
        self.parts: List[CacheSplit] = [split for split, _ in selections]
        for split in self.parts:
            if split.kind == "features" and split.n_mels != extractor.n_mels:
                raise ValueError(
                    f"{split.source}/{split.split} ships {split.n_mels}-bin features but the model expects "
                    f"{extractor.n_mels} mel bins (e.g. whisper-large-v3). It can only be evaluated on raw-audio sources."
                )
        self.part_ids = np.concatenate(
            [np.full(len(rows), i, dtype=np.int32) for i, (_, rows) in enumerate(selections)]
        ) if selections else np.zeros(0, dtype=np.int32)
        self.rows = np.concatenate([np.asarray(rows, dtype=np.int64) for _, rows in selections]) \
            if selections else np.zeros(0, dtype=np.int64)
        self.tokenizer = tokenizer
        self.extractor = extractor
        self.text_level = text_level
        enabled = augment is not None and augment.enabled
        self.waveform_aug = WaveformAugmenter(augment) if enabled else None
        self.spec_aug = SpecAugment(augment) if enabled else None
        self._rng: Optional[np.random.Generator] = None

    def __len__(self) -> int:
        return int(self.rows.shape[0])

    def _random(self) -> np.random.Generator:
        # One generator per worker process, derived from torch's per-worker seed.
        if self._rng is None:
            self._rng = np.random.default_rng(torch.initial_seed() % (2 ** 32))
        return self._rng

    def source_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for pid in self.part_ids:
            name = self.parts[pid].source
            counts[name] = counts.get(name, 0) + 1
        return counts

    def item_meta(self, index: int) -> Dict:
        part = self.parts[self.part_ids[index]]
        row = int(self.rows[index])
        return part.meta.iloc[row].to_dict()

    def __getitem__(self, index: int) -> Dict:
        part = self.parts[self.part_ids[index]]
        row = int(self.rows[index])
        if part.kind == "audio":
            audio = part.audio(row)
            if self.waveform_aug is not None:
                audio = self.waveform_aug(audio, self._random())
            feats_t, n_valid = self.extractor(audio)
            feats = feats_t.numpy()
        else:
            valid, floor = part.features(row)
            n_valid = int(valid.shape[1])
            feats = pad_features(valid, floor)
        if self.spec_aug is not None:
            feats = self.spec_aug(feats, n_valid, self._random())
        text = part.texts(self.text_level)[row]
        labels = self.tokenizer(text).input_ids  # <|sot|><|ar|><|transcribe|><|notimestamps|> ... <|eot|>
        return {
            "input_features": torch.from_numpy(np.ascontiguousarray(feats, dtype=np.float32)),
            "labels": labels,
            "id": part.meta["id"].iat[row] if "id" in part.meta else f"{part.source}-{row}",
            "source": part.source,
            "text": text,
            "n_valid": n_valid,
            "index": index,
        }


@dataclass
class WhisperCollator:
    """Stacks features and pads labels with -100 (ignored by the loss).

    The decoder-start token (<|startoftranscript|>) is removed from the labels because the
    model prepends it itself when building the shifted decoder input (same as Paper 3's
    collator and the Hugging Face recipe).
    """

    decoder_start_token_id: int

    def __call__(self, items: List[Dict]) -> Dict:
        feats = torch.stack([it["input_features"] for it in items])
        max_len = max(len(it["labels"]) for it in items)
        labels = torch.full((len(items), max_len), -100, dtype=torch.long)
        for i, it in enumerate(items):
            labels[i, : len(it["labels"])] = torch.tensor(it["labels"], dtype=torch.long)
        if bool((labels[:, 0] == self.decoder_start_token_id).all()):
            labels = labels[:, 1:]
        return {
            "input_features": feats,
            "labels": labels,
            "ids": [it["id"] for it in items],
            "sources": [it["source"] for it in items],
            "texts": [it["text"] for it in items],
            "n_valid": torch.tensor([it["n_valid"] for it in items]),
            "indices": [it["index"] for it in items],
        }


class ResumableRandomSampler(Sampler):
    """Random permutation per epoch (seed + epoch) that can start mid-epoch after a resume."""

    def __init__(self, n: int, seed: int):
        self.n, self.seed = n, seed
        self.epoch, self.start = 0, 0

    def set_state(self, epoch: int, start: int) -> None:
        self.epoch, self.start = int(epoch), int(start)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        perm = torch.randperm(self.n, generator=generator).tolist()
        return iter(perm[self.start :])

    def __len__(self) -> int:
        return max(0, self.n - self.start)


def _stratified(selections: Sequence[Selection], total: int, seed: int) -> List[Selection]:
    """Take about ``total`` rows spread evenly over the selections (seeded, sorted)."""
    if not selections:
        return []
    quota = max(1, math.ceil(total / len(selections)))
    rng = np.random.default_rng(seed)
    out = []
    for split, rows in selections:
        rows = np.asarray(rows)
        take = min(len(rows), quota)
        out.append((split, np.sort(rng.choice(rows, size=take, replace=False)) if take else rows[:0]))
    return out


@dataclass
class ExperimentData:
    train: ASRDataset
    val_loss: ASRDataset
    val_wer: ASRDataset
    summary: Dict
    manifest: Dict


def build_experiment_data(cfg: DataConfig, tokenizer, extractor: LogMelExtractor, train_seed: int = 42) -> ExperimentData:
    """Assemble train/validation datasets for one experiment from the prepared cache."""
    logger = get_logger()
    train_sel: List[Selection] = []
    val_sel: List[Selection] = []
    summary: Dict = {"sources": {}}
    manifest: Dict = {}

    def add_source(source: str, select_train=None) -> None:
        """``select_train`` optionally maps the source's training rows to a subset."""
        split = CacheSplit(cfg.cache_dir, source, "train")
        tr, va = train_val_rows(split, cfg)
        if select_train is not None:
            tr = select_train(tr)
        if cfg.max_train_samples_per_source:
            tr = tr[: cfg.max_train_samples_per_source]
        train_sel.append((split, tr))
        val_sel.append((split, va))
        durations = split.meta["duration"].to_numpy()
        summary["sources"][source] = {
            "train_rows": int(len(tr)),
            "train_hours": round(float(durations[tr].sum()) / 3600, 3),
            "val_rows": int(len(va)),
            "val_hours": round(float(durations[va].sum()) / 3600, 3),
        }
        ids = split.meta["id"].to_numpy()
        manifest[source] = {"train": ids[tr].tolist(), "val": ids[va].tolist()}

    for source in cfg.train_sources:
        add_source(source)

    if cfg.msa_replay_fraction > 0 and "msa" not in cfg.train_sources:
        frac = float(cfg.msa_replay_fraction)
        n_main = sum(len(rows) for _, rows in train_sel)
        n_replay = int(round(frac / (1.0 - frac) * n_main))

        def pick(rows: np.ndarray) -> np.ndarray:
            rng = np.random.default_rng(cfg.split_seed + 1)
            take = min(n_replay, len(rows))
            return np.sort(rng.choice(rows, size=take, replace=False))

        add_source("msa", pick)
        summary["msa_replay_rows"] = int(len(train_sel[-1][1]))

    val_loss_sel = _stratified(val_sel, cfg.val_loss_samples, cfg.split_seed + 2)
    val_wer_sel = _stratified(val_sel, cfg.val_wer_samples, cfg.split_seed + 3)

    train_ds = ASRDataset(train_sel, tokenizer, extractor, cfg.text_level, augment=cfg.augmentation)
    val_loss_ds = ASRDataset(val_loss_sel, tokenizer, extractor, cfg.text_level, augment=None)
    val_wer_ds = ASRDataset(val_wer_sel, tokenizer, extractor, cfg.text_level, augment=None)
    summary.update(
        train_rows=len(train_ds),
        train_hours=round(sum(v["train_hours"] for v in summary["sources"].values()), 3),
        val_loss_rows=len(val_loss_ds),
        val_wer_rows=len(val_wer_ds),
    )
    logger.info("Training mix: %s", train_ds.source_counts())
    logger.info("Validation (loss) rows: %d | validation (WER) rows: %d", len(val_loss_ds), len(val_wer_ds))
    return ExperimentData(train_ds, val_loss_ds, val_wer_ds, summary, manifest)


def build_eval_dataset(cache_dir: str, source: str, tokenizer, extractor: LogMelExtractor, split: str = "test",
                       max_samples: Optional[int] = None, seed: int = 0) -> ASRDataset:
    """A test (or train) split of one source, optionally a seeded random subset."""
    cache_split = CacheSplit(cache_dir, source, split)
    rows = np.arange(len(cache_split))
    if max_samples is not None and max_samples < len(rows):
        rows = np.sort(np.random.default_rng(seed).choice(rows, size=max_samples, replace=False))
    return ASRDataset([(cache_split, rows)], tokenizer, extractor, text_level="raw", augment=None)
