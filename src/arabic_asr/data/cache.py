"""Compact on-disk cache of prepared data.

Layout per corpus and split::

    {cache_dir}/{source}/{split}/data.bin       # one flat binary file
    {cache_dir}/{source}/{split}/meta.parquet   # one row per utterance (text, offsets, metadata)
    {cache_dir}/{source}/{split}/info.json      # kind, dtype, sizes, filter statistics

``kind == "audio"``: ``data.bin`` holds 16 kHz int16 PCM; ``offset``/``length`` are in samples.
``kind == "features"``: ``data.bin`` holds float16 log-Mel frames (frames x n_mels) with the
padding removed (see ``features.trim_padding``); ``offset``/``length`` are in frames.

Reading goes through ``np.memmap``, so random access is fast and needs almost no RAM, and
the dataset works with multi-process data loading.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from arabic_asr.data.audio import int16_to_float
from arabic_asr.data.text_normalization import normalize_text
from arabic_asr.utils import read_json, write_json

AUDIO_DTYPE = np.int16
FEATURE_DTYPE = np.float16


def split_dir(cache_dir: os.PathLike, source: str, split: str) -> Path:
    return Path(cache_dir) / source / split


def is_prepared(cache_dir: os.PathLike, source: str, split: str) -> bool:
    return (split_dir(cache_dir, source, split) / "info.json").exists()


class CacheWriter:
    """Appends utterances to ``data.bin`` and collects their metadata rows."""

    def __init__(self, out_dir: os.PathLike, kind: str, n_mels: int = 80):
        if kind not in {"audio", "features"}:
            raise ValueError(kind)
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.kind = kind
        self.n_mels = n_mels
        self.dtype = AUDIO_DTYPE if kind == "audio" else FEATURE_DTYPE
        self._partial = self.dir / "data.bin.partial"
        self._fh = open(self._partial, "wb")
        self._offset = 0  # in samples (audio) or frames (features)
        self.rows: List[Dict] = []

    # -- transactional helpers so a failed shard download can be retried cleanly --
    def mark(self) -> Tuple[int, int]:
        return self._offset, len(self.rows)

    def rollback(self, mark: Tuple[int, int]) -> None:
        offset, n_rows = mark
        unit = np.dtype(self.dtype).itemsize * (1 if self.kind == "audio" else self.n_mels)
        self._fh.flush()
        self._fh.truncate(offset * unit)
        self._fh.seek(offset * unit)
        self._offset = offset
        del self.rows[n_rows:]

    def add_audio(self, audio_int16: np.ndarray, **meta) -> None:
        data = np.ascontiguousarray(audio_int16, dtype=AUDIO_DTYPE)
        self._fh.write(data.tobytes())
        self.rows.append({"offset": self._offset, "length": int(data.shape[0]), **meta})
        self._offset += int(data.shape[0])

    def add_features(self, valid: np.ndarray, floor: float, **meta) -> None:
        if valid.shape[0] != self.n_mels:
            raise ValueError(f"Expected {self.n_mels} mel bins, got {valid.shape[0]}")
        frames = np.ascontiguousarray(valid.T, dtype=FEATURE_DTYPE)  # (L, n_mels)
        self._fh.write(frames.tobytes())
        self.rows.append({"offset": self._offset, "length": int(frames.shape[0]), "floor": float(floor), **meta})
        self._offset += int(frames.shape[0])

    def close(self) -> pd.DataFrame:
        self._fh.close()
        os.replace(self._partial, self.dir / "data.bin")
        return pd.DataFrame(self.rows)

    def abort(self) -> None:
        self._fh.close()
        if self._partial.exists():
            self._partial.unlink()


def save_meta(out_dir: os.PathLike, meta: pd.DataFrame, info: Dict) -> None:
    out_dir = Path(out_dir)
    meta.to_parquet(out_dir / "meta.parquet", index=False)
    write_json(info, out_dir / "info.json")  # written last: marks the split as complete


class CacheSplit:
    """Read-only view of one prepared split."""

    def __init__(self, cache_dir: os.PathLike, source: str, split: str):
        self.source = source
        self.split = split
        self.dir = split_dir(cache_dir, source, split)
        info_path = self.dir / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(
                f"No prepared data at {self.dir}. Run: python scripts/prepare_data.py "
                f"--config configs/data.yaml --sources {source}"
            )
        self.info = read_json(info_path)
        self.kind = self.info["kind"]
        self.n_mels = int(self.info.get("n_mels", 80))
        self.meta = pd.read_parquet(self.dir / "meta.parquet")
        self._offsets = self.meta["offset"].to_numpy(dtype=np.int64)
        self._lengths = self.meta["length"].to_numpy(dtype=np.int64)
        self._floors = self.meta["floor"].to_numpy(dtype=np.float32) if "floor" in self.meta else None
        self._data: Optional[np.memmap] = None
        self._text_cache: Dict[str, List[str]] = {}

    def __len__(self) -> int:
        return len(self.meta)

    # memmaps must not be pickled into DataLoader workers; reopen lazily instead.
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_data"] = None
        return state

    @property
    def data(self) -> np.memmap:
        if self._data is None:
            path = self.dir / "data.bin"
            if self.kind == "audio":
                self._data = np.memmap(path, dtype=AUDIO_DTYPE, mode="r")
            else:
                raw = np.memmap(path, dtype=FEATURE_DTYPE, mode="r")
                self._data = raw.reshape(-1, self.n_mels)
        return self._data

    def audio(self, row: int) -> np.ndarray:
        """16 kHz float32 waveform of row ``row`` (audio kind only)."""
        if self.kind != "audio":
            raise TypeError(f"{self.source}/{self.split} stores features, not audio")
        start = self._offsets[row]
        return int16_to_float(np.asarray(self.data[start : start + self._lengths[row]]))

    def features(self, row: int) -> Tuple[np.ndarray, float]:
        """(valid frames as float32 (n_mels, L), padding floor) for features kind."""
        if self.kind != "features":
            raise TypeError(f"{self.source}/{self.split} stores audio, not features")
        start = self._offsets[row]
        frames = np.asarray(self.data[start : start + self._lengths[row]], dtype=np.float32)
        return frames.T, float(self._floors[row])

    def texts(self, level: str = "light") -> List[str]:
        if level not in self._text_cache:
            if level == "light" and "text" in self.meta:
                self._text_cache[level] = self.meta["text"].fillna("").tolist()
            else:
                self._text_cache[level] = [normalize_text(t or "", level) for t in self.meta["text_raw"].tolist()]
        return self._text_cache[level]

    def kept_rows(self) -> np.ndarray:
        """Row positions that passed the training filters."""
        if "filter_reason" not in self.meta:
            return np.arange(len(self.meta))
        return np.flatnonzero(self.meta["filter_reason"].fillna("").to_numpy() == "")
