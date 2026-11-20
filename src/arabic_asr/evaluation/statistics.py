"""Uncertainty and significance for corpus-level error rates.

* ``bootstrap_ci``: 95% confidence interval of a corpus WER by resampling utterances.
* ``paired_bootstrap``: difference between two systems on the *same* utterances, with a CI
  and a two-sided p-value (Koehn, 2004; Bisani and Ney, 2004).
* ``wilcoxon_paired``: Wilcoxon signed-rank test on per-utterance error counts, the test
  family Paper 3 used, plus Holm-Bonferroni correction for several test sets.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def _resample_ratios(errors: np.ndarray, totals: np.ndarray, n_boot: int, rng: np.random.Generator,
                     chunk_elems: int = 4_000_000) -> np.ndarray:
    n = len(errors)
    per_chunk = max(1, chunk_elems // max(1, n))
    out = []
    for start in range(0, n_boot, per_chunk):
        k = min(per_chunk, n_boot - start)
        idx = rng.integers(0, n, size=(k, n))
        out.append(errors[idx].sum(axis=1) / np.maximum(1, totals[idx].sum(axis=1)))
    return 100.0 * np.concatenate(out)


def bootstrap_ci(errors: Sequence[float], totals: Sequence[float], n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 0) -> Dict[str, float]:
    errors = np.asarray(errors, dtype=np.float64)
    totals = np.asarray(totals, dtype=np.float64)
    if len(errors) == 0 or totals.sum() == 0:
        return {"value": float("nan"), "low": float("nan"), "high": float("nan")}
    ratios = _resample_ratios(errors, totals, n_boot, np.random.default_rng(seed))
    return {
        "value": float(100.0 * errors.sum() / totals.sum()),
        "low": float(np.percentile(ratios, 100 * alpha / 2)),
        "high": float(np.percentile(ratios, 100 * (1 - alpha / 2))),
    }


def paired_bootstrap(errors_a: Sequence[float], errors_b: Sequence[float], totals: Sequence[float],
                     n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> Dict[str, float]:
    """WER(b) - WER(a) on identical utterances. Negative delta means b is better."""
    a = np.asarray(errors_a, dtype=np.float64)
    b = np.asarray(errors_b, dtype=np.float64)
    t = np.asarray(totals, dtype=np.float64)
    if not (len(a) == len(b) == len(t)) or len(a) == 0:
        raise ValueError("paired_bootstrap needs equally long, non-empty arrays")
    rng = np.random.default_rng(seed)
    n = len(a)
    deltas = []
    per_chunk = max(1, 4_000_000 // n)
    for start in range(0, n_boot, per_chunk):
        k = min(per_chunk, n_boot - start)
        idx = rng.integers(0, n, size=(k, n))
        tot = np.maximum(1, t[idx].sum(axis=1))
        deltas.append(100.0 * (b[idx].sum(axis=1) - a[idx].sum(axis=1)) / tot)
    deltas = np.concatenate(deltas)
    delta = 100.0 * (b.sum() - a.sum()) / max(1.0, t.sum())
    p_one = float(np.mean(deltas >= 0)) if delta < 0 else float(np.mean(deltas <= 0))
    return {
        "delta": float(delta),
        "low": float(np.percentile(deltas, 100 * alpha / 2)),
        "high": float(np.percentile(deltas, 100 * (1 - alpha / 2))),
        "p_value": float(min(1.0, 2 * p_one)),
    }


def wilcoxon_paired(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    from scipy.stats import wilcoxon

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if np.allclose(x, y):
        return {"statistic": 0.0, "p_value": 1.0}
    result = wilcoxon(x, y, zero_method="wilcox")
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}


def holm_bonferroni(p_values: Sequence[float]) -> List[float]:
    """Holm's step-down adjusted p-values (same order as the input)."""
    p = np.asarray(p_values, dtype=np.float64)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted.tolist()
