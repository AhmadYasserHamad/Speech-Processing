"""WER / CER at three normalisation levels (1C5).

WER = (S + D + I) / N over words, CER the same over characters (spaces included, as in
jiwer's default and Paper 3's ``evaluate`` metric). Scores are *corpus-level*: errors and
reference lengths are summed over all utterances before dividing, which is what jiwer
and Paper 3 report. Utterances whose reference is empty at a given level (e.g. a subtitle
that only contained "♪") are excluded at that level and counted separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import jiwer

from arabic_asr.data.text_normalization import LEVELS, normalize_text


@dataclass
class EditCounts:
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    hits: int = 0

    @property
    def ref_len(self) -> int:
        return self.substitutions + self.deletions + self.hits

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    def __add__(self, other: "EditCounts") -> "EditCounts":
        return EditCounts(
            self.substitutions + other.substitutions,
            self.deletions + other.deletions,
            self.insertions + other.insertions,
            self.hits + other.hits,
        )


def word_edit_counts(ref: str, hyp: str) -> EditCounts:
    ref_words, hyp_words = ref.split(), hyp.split()
    if not ref_words:
        return EditCounts(insertions=len(hyp_words))
    if not hyp_words:
        return EditCounts(deletions=len(ref_words))
    out = jiwer.process_words(ref, hyp)
    return EditCounts(out.substitutions, out.deletions, out.insertions, out.hits)


def char_edit_counts(ref: str, hyp: str) -> EditCounts:
    if not ref:
        return EditCounts(insertions=len(hyp))
    if not hyp:
        return EditCounts(deletions=len(ref))
    out = jiwer.process_characters(ref, hyp)
    return EditCounts(out.substitutions, out.deletions, out.insertions, out.hits)


def word_alignment(ref: str, hyp: str) -> List[Tuple[str, str, str]]:
    """Word-level alignment as (operation, ref_word, hyp_word) triples.

    operation is one of ``equal``, ``substitute``, ``delete`` (hyp_word == "") or
    ``insert`` (ref_word == "").
    """
    ref_words, hyp_words = ref.split(), hyp.split()
    if not ref_words:
        return [("insert", "", w) for w in hyp_words]
    if not hyp_words:
        return [("delete", w, "") for w in ref_words]
    out = jiwer.process_words(ref, hyp)
    pairs: List[Tuple[str, str, str]] = []
    for chunk in out.alignments[0]:
        r = ref_words[chunk.ref_start_idx : chunk.ref_end_idx]
        h = hyp_words[chunk.hyp_start_idx : chunk.hyp_end_idx]
        if chunk.type in ("equal", "substitute"):
            pairs.extend((chunk.type, a, b) for a, b in zip(r, h))
        elif chunk.type == "delete":
            pairs.extend(("delete", a, "") for a in r)
        elif chunk.type == "insert":
            pairs.extend(("insert", "", b) for b in h)
    return pairs


def corpus_scores(refs: Sequence[str], hyps: Sequence[str]) -> Dict[str, float]:
    if len(refs) != len(hyps):
        raise ValueError("refs and hyps must have the same length")
    words, chars = EditCounts(), EditCounts()
    n_used = n_empty = 0
    for ref, hyp in zip(refs, hyps):
        if not ref.strip():
            n_empty += 1
            continue
        n_used += 1
        words = words + word_edit_counts(ref, hyp)
        chars = chars + char_edit_counts(ref, hyp)
    return {
        "wer": 100.0 * words.errors / words.ref_len if words.ref_len else float("nan"),
        "cer": 100.0 * chars.errors / chars.ref_len if chars.ref_len else float("nan"),
        "substitutions": words.substitutions,
        "deletions": words.deletions,
        "insertions": words.insertions,
        "ref_words": words.ref_len,
        "ref_chars": chars.ref_len,
        "n_utterances": n_used,
        "n_empty_refs": n_empty,
    }


def score_all_levels(refs: Sequence[str], hyps: Sequence[str], levels: Iterable[str] = LEVELS) -> Dict[str, Dict]:
    """Normalise both sides at each level, then compute corpus WER/CER."""
    return {
        level: corpus_scores([normalize_text(r, level) for r in refs], [normalize_text(h, level) for h in hyps])
        for level in levels
    }
