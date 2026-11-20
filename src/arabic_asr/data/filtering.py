"""Training-data filters (Paper 2's "Basic" and "Data filtration" steps, adapted).

Every row gets a ``filter_reason``: the first rule it fails, or ``""`` if it is kept.
Rows are never deleted from the cache, so the effect of each rule can be reported and
the filters can be re-run with other thresholds (``prepare_data.py --refilter-only``).

Filters are applied to the *training* split only. Test sets stay exactly as the authors
released them, which keeps our results comparable with the paper.

Paper 2's model-based filter (drop samples with WER > 60% or CER > 30% under a bootstrap
model) is deliberately not used: it assumes a model at ~10% WER, while dialect WER in
Paper 3 is 72-86%, so it would discard most dialect data and bias the set towards speech
that is already easy (see docs/PREPROCESSING.md).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from arabic_asr.config import FilterConfig

REASONS = (
    "empty_text",
    "music",
    "too_short",
    "too_long",
    "too_many_tokens",
    "speaking_rate_outlier",
    "test_overlap",
)


def speaking_rate(texts: pd.Series, durations: pd.Series) -> pd.Series:
    """Characters (excluding spaces) per second of audio."""
    chars = texts.fillna("").str.replace(" ", "", regex=False).str.len().astype(float)
    return chars / durations.clip(lower=1e-3)


def tukey_fences(values: np.ndarray, k: float):
    """Return (low, high) Tukey fences Q1 - k*IQR, Q3 + k*IQR."""
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def assign_filter_reasons(
    meta: pd.DataFrame,
    cfg: FilterConfig,
    test_texts: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Add a ``filter_reason`` column to a training-split metadata table.

    Expected columns: ``text`` (light-normalised), ``has_music``, ``duration``, ``n_tokens``.
    """
    meta = meta.copy()
    reason = pd.Series("", index=meta.index, dtype=object)

    def mark(mask: pd.Series, name: str) -> None:
        nonlocal reason
        reason = reason.mask((reason == "") & mask.fillna(False), name)

    mark(meta["text"].fillna("").str.len() == 0, "empty_text")
    if cfg.drop_music:
        mark(meta["has_music"].astype(bool), "music")
    mark(meta["duration"] < cfg.min_duration_s, "too_short")
    mark(meta["duration"] > cfg.max_duration_s, "too_long")
    mark(meta["n_tokens"] > cfg.max_label_tokens, "too_many_tokens")

    # Speaking-rate outliers, computed only over rows that survived the rules above.
    alive = reason == ""
    if cfg.speaking_rate_iqr_k and alive.sum() >= 20:
        rate = speaking_rate(meta.loc[alive, "text"], meta.loc[alive, "duration"])
        log_rate = np.log(rate.clip(lower=1e-3).to_numpy())
        low, high = tukey_fences(log_rate, cfg.speaking_rate_iqr_k)
        outlier = pd.Series(False, index=meta.index)
        outlier.loc[alive] = (log_rate < low) | (log_rate > high)
        mark(outlier, "speaking_rate_outlier")

    # Text leakage: the same sentence in train and test (Common Voice re-uses prompts).
    if test_texts is not None:
        test_set = {t for t in test_texts if t}
        min_words = max(1, int(cfg.leakage_min_words))
        overlap = meta["text"].fillna("").map(
            lambda t: bool(t) and len(t.split()) >= min_words and t in test_set
        )
        mark(overlap, "test_overlap")

    meta["filter_reason"] = reason.astype(str)
    return meta


def filter_summary(meta: pd.DataFrame, duration_col: str = "duration") -> Dict[str, Dict[str, float]]:
    """Counts and hours removed by each rule (and kept)."""
    summary: "OrderedDict[str, Dict[str, float]]" = OrderedDict()
    reasons = meta["filter_reason"].replace("", "kept")
    for name in ("kept",) + REASONS:
        mask = reasons == name
        if mask.any() or name == "kept":
            summary[name] = {
                "rows": int(mask.sum()),
                "hours": round(float(meta.loc[mask, duration_col].sum()) / 3600.0, 3),
            }
    return dict(summary)
