"""Build the local cache from the authors' Hugging Face datasets (1C1 preprocessing pipeline).

For each corpus and split:

1. read the parquet shards (only the needed columns for MSA, via HTTP range requests);
2. MSA: decode the 16 kHz WAV bytes -> int16 PCM (features are recomputed on the fly later,
   so augmentation can act on the waveform). MASC: take the authors' log-Mel features,
   strip their padding losslessly, store as float16; decode the token ids back to text;
3. normalise the text (``light`` level = training target) and compute duration, token
   count and flags (Latin script, music, diacritics);
4. training split only: assign filter reasons (duration, token limit, music, speaking-rate
   outliers, text overlap with the test split).

The test split is prepared first so the leakage check can use its texts.
"""

from __future__ import annotations

import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from arabic_asr.config import PrepareConfig, SourceSpec
from arabic_asr.data.audio import decode_audio_bytes, float_to_int16
from arabic_asr.data.cache import CacheWriter, is_prepared, save_meta, split_dir
from arabic_asr.data.features import trim_padding
from arabic_asr.data.filtering import assign_filter_reasons, filter_summary
from arabic_asr.data.text_normalization import has_diacritics, has_latin, has_music, normalize_text
from arabic_asr.utils import get_logger, read_json, write_json

AUDIO_COLUMNS = ["audio", "sentence", "client_id", "gender", "age", "accent"]
FEATURE_COLUMNS = ["input_features", "labels"]
FEATURE_FRAME_RATE = 100.0  # Whisper: 10 ms hop


EXPECTED_PREFIX = ["<|startoftranscript|>", "<|ar|>", "<|transcribe|>", "<|notimestamps|>"]


def load_tokenizer(name: str, language: str = "arabic", task: str = "transcribe"):
    """Whisper tokenizer whose encodings start with <|sot|><|ar|><|transcribe|><|notimestamps|>.

    ``WhisperTokenizerFast.from_pretrained(..., language=...)`` stores the language but (in
    several transformers 4.x releases) does not rebuild the post-processor, so the language
    and task tokens are silently missing. ``set_prefix_tokens`` forces the rebuild; the
    assertion guards against regressions.
    """
    from transformers import WhisperTokenizerFast

    tokenizer = WhisperTokenizerFast.from_pretrained(name, language=language, task=task)
    tokenizer.set_prefix_tokens(language=language, task=task, predict_timestamps=False)
    prefix = tokenizer.convert_ids_to_tokens(tokenizer("ا").input_ids[:4])
    if language == "arabic" and task == "transcribe" and prefix != EXPECTED_PREFIX:
        raise RuntimeError(f"Unexpected Whisper prefix tokens {prefix}; expected {EXPECTED_PREFIX}")
    return tokenizer


def list_parquet_files(repo: str) -> List[str]:
    from huggingface_hub import list_repo_files

    files = [f for f in list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet")]
    if not files:
        raise FileNotFoundError(f"No parquet files found in dataset repo {repo}")
    return sorted(files)


def _features_to_numpy(column: pa.Array) -> np.ndarray:
    """list<list<float>> column of shape (rows, n_mels, frames) -> float32 ndarray, fast."""
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    n_rows = len(column)
    inner = column.flatten()
    n_mels = set(column.value_lengths().to_pylist())
    n_frames = set(inner.value_lengths().to_pylist())
    if len(n_mels) != 1 or len(n_frames) != 1:
        return np.stack([np.asarray(x, dtype=np.float32) for x in column.to_pylist()])
    flat = inner.flatten().to_numpy(zero_copy_only=False)
    return flat.reshape(n_rows, n_mels.pop(), n_frames.pop()).astype(np.float32, copy=False)


class _ShardReader:
    """Opens a parquet shard either by streaming (column subset) or by full download."""

    def __init__(self, repo: str, filename: str, download_dir: Path, stream: bool):
        self.repo, self.filename, self.stream = repo, filename, stream
        self.download_dir = download_dir
        self._local: Optional[Path] = None
        self._fh = None

    def __enter__(self) -> pq.ParquetFile:
        if self.stream:
            from huggingface_hub import HfFileSystem

            self._fh = HfFileSystem().open(f"datasets/{self.repo}/{self.filename}", "rb")
            return pq.ParquetFile(self._fh)
        from huggingface_hub import hf_hub_download

        self.download_dir.mkdir(parents=True, exist_ok=True)
        self._local = Path(
            hf_hub_download(self.repo, self.filename, repo_type="dataset", local_dir=str(self.download_dir))
        )
        return pq.ParquetFile(self._local)

    def __exit__(self, *exc) -> None:
        if self._fh is not None:
            self._fh.close()
        if self._local is not None and self._local.exists():
            self._local.unlink()  # free disk as we go


def _iter_shard_rows(pf: pq.ParquetFile, columns: Sequence[str], batch_rows: int, limit: Optional[int]):
    available = set(pf.schema_arrow.names)
    cols = [c for c in columns if c in available]
    produced = 0
    for batch in pf.iter_batches(batch_size=batch_rows, columns=cols):
        if limit is not None and produced >= limit:
            return
        if limit is not None and produced + batch.num_rows > limit:
            batch = batch.slice(0, limit - produced)
        produced += batch.num_rows
        yield batch


def _process_audio_batch(batch: pa.RecordBatch, writer: CacheWriter, filename: str, row0: int) -> None:
    data = batch.to_pydict()
    for i in range(batch.num_rows):
        wav = decode_audio_bytes(data["audio"][i]["bytes"])
        writer.add_audio(
            float_to_int16(wav),
            text_raw=(data["sentence"][i] or "").strip(),
            client_id=(data.get("client_id", [""] * batch.num_rows)[i] or ""),
            gender=(data.get("gender", [""] * batch.num_rows)[i] or ""),
            age=(data.get("age", [""] * batch.num_rows)[i] or ""),
            accent=(data.get("accent", [""] * batch.num_rows)[i] or ""),
            hf_file=filename,
            hf_row=row0 + i,
        )


def _process_feature_batch(batch: pa.RecordBatch, writer: CacheWriter, tokenizer, filename: str, row0: int) -> None:
    feats = _features_to_numpy(batch.column("input_features"))
    labels = batch.column("labels").to_pylist()
    texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
    for i in range(batch.num_rows):
        valid, floor = trim_padding(feats[i])
        writer.add_features(valid, floor, text_raw=texts[i].strip(), hf_file=filename, hf_row=row0 + i)


def _derive_columns(meta: pd.DataFrame, kind: str, source: str, split: str, tokenizer) -> pd.DataFrame:
    meta = meta.copy()
    meta.insert(0, "id", [f"{source}-{split}-{i:06d}" for i in range(len(meta))])
    meta["source"] = source
    meta["split"] = split
    meta["text"] = [normalize_text(t, "light") for t in meta["text_raw"]]
    if kind == "audio":
        meta["duration"] = meta["length"] / 16000.0
    else:
        meta["duration"] = meta["length"] / FEATURE_FRAME_RATE
    token_ids = tokenizer(meta["text"].tolist()).input_ids if len(meta) else []
    meta["n_tokens"] = [len(ids) for ids in token_ids]
    meta["has_latin"] = meta["text_raw"].map(has_latin)
    meta["has_music"] = meta["text_raw"].map(has_music)
    meta["has_diacritics"] = meta["text_raw"].map(has_diacritics)
    for col in ("client_id", "gender", "age", "accent"):
        if col not in meta:
            meta[col] = ""
    return meta


def _apply_filters(cfg: PrepareConfig, source: str, meta: pd.DataFrame) -> pd.DataFrame:
    test_dir = split_dir(cfg.cache_dir, source, "test")
    test_texts = None
    if (test_dir / "meta.parquet").exists():
        test_texts = pd.read_parquet(test_dir / "meta.parquet", columns=["text"])["text"].tolist()
    else:
        get_logger().warning("[%s] test split not prepared; skipping the train/test leakage check", source)
    return assign_filter_reasons(meta, cfg.filters, test_texts=test_texts)


def _info(spec: SourceSpec, source: str, split: str, meta: pd.DataFrame, files: List[str], partial: bool,
          cfg: PrepareConfig, seconds: float) -> dict:
    kept = meta["filter_reason"] == "" if "filter_reason" in meta else pd.Series(True, index=meta.index)
    return {
        "source": source,
        "split": split,
        "variety": spec.variety,
        "kind": spec.kind,
        "repo": spec.test_repo if split == "test" else spec.train_repo,
        "dtype": "int16" if spec.kind == "audio" else "float16",
        "sample_rate": 16000,
        "frame_rate": FEATURE_FRAME_RATE,
        "n_mels": 80,
        "rows": int(len(meta)),
        "hours": round(float(meta["duration"].sum()) / 3600.0, 3),
        "rows_kept": int(kept.sum()),
        "hours_kept": round(float(meta.loc[kept, "duration"].sum()) / 3600.0, 3),
        "filters": filter_summary(meta) if split == "train" else None,
        "flags": {
            "has_latin": int(meta["has_latin"].sum()),
            "has_music": int(meta["has_music"].sum()),
            "has_diacritics": int(meta["has_diacritics"].sum()),
        },
        "tokenizer": cfg.tokenizer_name,
        "parquet_files": files,
        "partial": partial,
        "prepare_seconds": round(seconds, 1),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def prepare_split(cfg: PrepareConfig, source: str, split: str, tokenizer, max_rows: Optional[int] = None,
                  overwrite: bool = False) -> Optional[dict]:
    logger = get_logger()
    spec = cfg.sources[source]
    out_dir = split_dir(cfg.cache_dir, source, split)
    if is_prepared(cfg.cache_dir, source, split) and not overwrite:
        logger.info("[%s/%s] already prepared at %s (use --overwrite to rebuild)", source, split, out_dir)
        return None
    if out_dir.exists():
        shutil.rmtree(out_dir)
    repo = spec.test_repo if split == "test" else spec.train_repo
    files = list_parquet_files(repo)
    logger.info("[%s/%s] %s: %d parquet shard(s)", source, split, repo, len(files))

    start = time.time()
    writer = CacheWriter(out_dir, spec.kind)
    stream = spec.kind == "audio"  # MSA: only fetch the audio/text columns, not the 45 GB of features
    columns = AUDIO_COLUMNS if spec.kind == "audio" else FEATURE_COLUMNS
    download_dir = Path(cfg.cache_dir) / "_downloads"
    remaining = max_rows
    used_files: List[str] = []
    try:
        for filename in files:
            if remaining is not None and remaining <= 0:
                break
            mark = writer.mark()
            for attempt in range(1, cfg.download_retries + 1):
                try:
                    with _ShardReader(repo, filename, download_dir, stream) as pf:
                        row0 = 0
                        for batch in _iter_shard_rows(pf, columns, cfg.read_batch_rows, remaining):
                            if spec.kind == "audio":
                                _process_audio_batch(batch, writer, filename, row0)
                            else:
                                _process_feature_batch(batch, writer, tokenizer, filename, row0)
                            row0 += batch.num_rows
                    break
                except Exception as error:  # noqa: BLE001 - network hiccups: retry this shard
                    writer.rollback(mark)
                    if attempt == cfg.download_retries:
                        raise
                    wait = 5 * attempt
                    logger.warning("[%s/%s] %s failed (%s); retry %d in %ds", source, split, filename, error, attempt, wait)
                    time.sleep(wait)
            used_files.append(filename)
            if remaining is not None:
                remaining = max_rows - len(writer.rows)
            logger.info("[%s/%s] %s done (%d rows so far)", source, split, filename, len(writer.rows))
    except BaseException:
        writer.abort()
        raise
    meta = writer.close()
    meta = _derive_columns(meta, spec.kind, source, split, tokenizer)
    if split == "train":
        meta = _apply_filters(cfg, source, meta)
    else:
        meta["filter_reason"] = ""  # test sets are never filtered
    partial = max_rows is not None
    info = _info(spec, source, split, meta, used_files, partial, cfg, time.time() - start)
    save_meta(out_dir, meta, info)
    shutil.rmtree(download_dir, ignore_errors=True)
    logger.info(
        "[%s/%s] wrote %d rows (%.2f h, kept %d rows / %.2f h) in %.0fs",
        source, split, info["rows"], info["hours"], info["rows_kept"], info["hours_kept"], info["prepare_seconds"],
    )
    return info


def refilter_split(cfg: PrepareConfig, source: str) -> dict:
    """Re-run the training filters with the current config without downloading again."""
    out_dir = split_dir(cfg.cache_dir, source, "train")
    meta = pd.read_parquet(out_dir / "meta.parquet")
    meta = _apply_filters(cfg, source, meta.drop(columns=["filter_reason"], errors="ignore"))
    info = read_json(out_dir / "info.json")
    kept = meta["filter_reason"] == ""
    info.update(
        rows_kept=int(kept.sum()),
        hours_kept=round(float(meta.loc[kept, "duration"].sum()) / 3600.0, 3),
        filters=filter_summary(meta),
    )
    meta.to_parquet(out_dir / "meta.parquet", index=False)
    write_json(info, out_dir / "info.json")
    return info


def prepare_sources(cfg: PrepareConfig, sources: Iterable[str], splits: Sequence[str] = ("test", "train"),
                    max_rows: Optional[int] = None, overwrite: bool = False) -> dict:
    tokenizer = load_tokenizer(cfg.tokenizer_name)
    ordered = [s for s in ("test", "train") if s in splits]  # test first: leakage check needs it
    report = {}
    for source in sources:
        if source not in cfg.sources:
            raise KeyError(f"Unknown source {source!r}; known: {sorted(cfg.sources)}")
        for split in ordered:
            report[f"{source}/{split}"] = prepare_split(cfg, source, split, tokenizer, max_rows, overwrite)
    return report


def cache_overview(cache_dir: os.PathLike) -> pd.DataFrame:
    """One row per prepared split: rows, hours, kept rows/hours (for the report)."""
    rows = []
    for info_path in sorted(Path(cache_dir).glob("*/*/info.json")):
        info = read_json(info_path)
        rows.append({k: info.get(k) for k in ("source", "split", "variety", "kind", "rows", "hours",
                                               "rows_kept", "hours_kept", "partial")})
    return pd.DataFrame(rows)
