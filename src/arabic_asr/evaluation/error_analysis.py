"""Detailed error analysis on a per-utterance predictions table (1C5).

All functions take the ``predictions`` DataFrame written by ``evaluate.py`` (one row per
test utterance, raw ``ref``/``hyp`` plus metadata) and work at a chosen normalisation
level. :func:`add_level_columns` adds the normalised texts and per-utterance edit counts.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from arabic_asr.data.text_normalization import LEVELS, normalize_text
from arabic_asr.evaluation.metrics import char_edit_counts, word_alignment, word_edit_counts
from arabic_asr.evaluation.statistics import bootstrap_ci


def add_level_columns(df: pd.DataFrame, levels: Iterable[str] = LEVELS) -> pd.DataFrame:
    """Add ref_<level>, hyp_<level> and word/char edit counts for each level."""
    df = df.copy()
    for level in levels:
        refs = [normalize_text(r or "", level) for r in df["ref"].fillna("")]
        hyps = [normalize_text(h or "", level) for h in df["hyp"].fillna("")]
        words = [word_edit_counts(r, h) for r, h in zip(refs, hyps)]
        chars = [char_edit_counts(r, h) for r, h in zip(refs, hyps)]
        df[f"ref_{level}"] = refs
        df[f"hyp_{level}"] = hyps
        df[f"w_sub_{level}"] = [w.substitutions for w in words]
        df[f"w_del_{level}"] = [w.deletions for w in words]
        df[f"w_ins_{level}"] = [w.insertions for w in words]
        df[f"w_err_{level}"] = [w.errors for w in words]
        df[f"w_ref_{level}"] = [w.ref_len for w in words]
        df[f"c_err_{level}"] = [c.errors for c in chars]
        df[f"c_ref_{level}"] = [c.ref_len for c in chars]
        with np.errstate(divide="ignore", invalid="ignore"):
            df[f"wer_{level}"] = np.where(df[f"w_ref_{level}"] > 0,
                                          100.0 * df[f"w_err_{level}"] / df[f"w_ref_{level}"], np.nan)
    return df


def _valid(df: pd.DataFrame, level: str) -> pd.DataFrame:
    return df[df[f"w_ref_{level}"] > 0]


def corpus_wer(df: pd.DataFrame, level: str) -> float:
    d = _valid(df, level)
    return float(100.0 * d[f"w_err_{level}"].sum() / max(1, d[f"w_ref_{level}"].sum()))


def summary_by_test_set(df: pd.DataFrame, levels: Iterable[str] = LEVELS, n_boot: int = 1000) -> pd.DataFrame:
    rows = []
    for test_set, group in df.groupby("test_set", sort=False):
        for level in levels:
            d = _valid(group, level)
            n_ref = d[f"w_ref_{level}"].sum()
            ci = bootstrap_ci(d[f"w_err_{level}"], d[f"w_ref_{level}"], n_boot=n_boot)
            rows.append({
                "test_set": test_set,
                "level": level,
                "wer": ci["value"],
                "wer_ci_low": ci["low"],
                "wer_ci_high": ci["high"],
                "cer": float(100.0 * d[f"c_err_{level}"].sum() / max(1, d[f"c_ref_{level}"].sum())),
                "sub_pct": float(100.0 * d[f"w_sub_{level}"].sum() / max(1, n_ref)),
                "del_pct": float(100.0 * d[f"w_del_{level}"].sum() / max(1, n_ref)),
                "ins_pct": float(100.0 * d[f"w_ins_{level}"].sum() / max(1, n_ref)),
                "ref_words": int(n_ref),
                "n_utterances": int(len(d)),
                "n_empty_refs": int(len(group) - len(d)),
            })
    return pd.DataFrame(rows)


def orthographic_share(df: pd.DataFrame) -> pd.DataFrame:
    """How much of the raw error disappears with light and with strict normalisation.

    ``light_drop`` = errors removed by stripping diacritics/punctuation/formatting;
    ``strict_extra_drop`` = additional errors removed by alef/ta-marbuta/alef-maqsura folding,
    i.e. spelling differences that are not recognition errors.
    """
    rows = []
    for test_set, g in df.groupby("test_set", sort=False):
        raw, light, strict = (corpus_wer(g, lvl) for lvl in LEVELS)
        rows.append({
            "test_set": test_set,
            "wer_raw": raw,
            "wer_light": light,
            "wer_strict": strict,
            "light_drop_pts": raw - light,
            "strict_extra_drop_pts": light - strict,
            "orthographic_share_of_light_errors_pct": 100.0 * (light - strict) / light if light else float("nan"),
        })
    return pd.DataFrame(rows)


def alignment_counters(df: pd.DataFrame, level: str = "light"):
    subs, dels, ins = Counter(), Counter(), Counter()
    for ref, hyp in zip(df[f"ref_{level}"], df[f"hyp_{level}"]):
        for op, r, h in word_alignment(ref, hyp):
            if op == "substitute":
                subs[(r, h)] += 1
            elif op == "delete":
                dels[r] += 1
            elif op == "insert":
                ins[h] += 1
    return subs, dels, ins


def top_errors(df: pd.DataFrame, level: str = "light", top: int = 25) -> dict:
    """Most frequent substitution pairs, deleted words and inserted words."""
    subs, dels, ins = alignment_counters(df, level)
    return {
        "substitutions": pd.DataFrame([{"ref": r, "hyp": h, "count": c} for (r, h), c in subs.most_common(top)]),
        "deletions": pd.DataFrame([{"word": w, "count": c} for w, c in dels.most_common(top)]),
        "insertions": pd.DataFrame([{"word": w, "count": c} for w, c in ins.most_common(top)]),
    }


def wer_by_bucket(df: pd.DataFrame, level: str, column: str, bins: Sequence[float],
                  labels: Sequence[str]) -> pd.DataFrame:
    d = _valid(df, level).copy()
    d["bucket"] = pd.cut(d[column], bins=bins, labels=labels, right=True, include_lowest=True)
    rows = []
    for (test_set, bucket), g in d.groupby(["test_set", "bucket"], observed=True, sort=False):
        rows.append({"test_set": test_set, "bucket": str(bucket), "n_utterances": len(g),
                     "wer": float(100.0 * g[f"w_err_{level}"].sum() / max(1, g[f"w_ref_{level}"].sum()))})
    out = pd.DataFrame(rows)
    if len(out):
        out["bucket"] = pd.Categorical(out["bucket"], categories=list(labels), ordered=True)
        out = out.sort_values(["test_set", "bucket"])
    return out


def _has_repetition(text: str, n: int = 3, times: int = 3) -> bool:
    words = text.split()
    if len(words) < n * times:
        return False
    grams = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    return max(grams.values()) >= times


def hallucination_flags(df: pd.DataFrame, level: str = "light") -> pd.DataFrame:
    """Per test set: share of utterances with WER > 100%, runaway output or looping n-grams."""
    d = df.copy()
    ref_len = d[f"ref_{level}"].str.split().str.len().fillna(0)
    hyp_len = d[f"hyp_{level}"].str.split().str.len().fillna(0)
    d["flag_wer_over_100"] = d[f"wer_{level}"] > 100
    d["flag_runaway"] = hyp_len > 2 * ref_len + 5
    d["flag_repetition"] = d[f"hyp_{level}"].map(_has_repetition)
    if "finished" in d:
        d["flag_truncated"] = ~d["finished"].astype(bool)
    flags = [c for c in d.columns if c.startswith("flag_")]
    d["flag_any"] = d[flags].any(axis=1)
    return d.groupby("test_set", sort=False)[flags + ["flag_any"]].mean().mul(100).reset_index()


def slice_table(df: pd.DataFrame, level: str, by: str, min_count: int = 20, n_boot: int = 500) -> pd.DataFrame:
    """WER per group (e.g. gender, age, has_latin) with bootstrap CIs; small groups dropped."""
    rows = []
    d = _valid(df, level)
    for (test_set, group), g in d.groupby(["test_set", by], sort=False, dropna=False):
        if len(g) < min_count:
            continue
        ci = bootstrap_ci(g[f"w_err_{level}"], g[f"w_ref_{level}"], n_boot=n_boot)
        rows.append({"test_set": test_set, by: group if group != "" else "(unlabelled)", "n_utterances": len(g),
                     "wer": ci["value"], "wer_ci_low": ci["low"], "wer_ci_high": ci["high"]})
    return pd.DataFrame(rows)


def calibration_table(df: pd.DataFrame, level: str = "light", n_bins: int = 10) -> pd.DataFrame:
    """Mean model confidence vs. observed word accuracy (1 - WER, floored at 0) per bin."""
    d = _valid(df, level).dropna(subset=["confidence"]).copy()
    if d.empty:
        return pd.DataFrame()
    d["accuracy"] = (1.0 - d[f"w_err_{level}"] / d[f"w_ref_{level}"]).clip(lower=0.0)
    d["bin"] = pd.cut(d["confidence"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    out = d.groupby("bin", observed=True).agg(mean_confidence=("confidence", "mean"),
                                              mean_accuracy=("accuracy", "mean"),
                                              n_utterances=("accuracy", "size")).reset_index()
    out["bin"] = out["bin"].astype(str)
    return out
