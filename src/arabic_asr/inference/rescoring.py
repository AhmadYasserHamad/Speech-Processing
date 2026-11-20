"""N-best rescoring with an external language model (shallow-fusion-style log-linear mix).

    score(hyp) = log P_whisper(hyp | audio) + lm_weight * log P_LM(hyp) + word_bonus * |hyp|

The word bonus compensates for both models preferring short hypotheses. ``lm_weight`` and
``word_bonus`` are tuned on the *validation* set with :func:`grid_search`, never on test.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from arabic_asr.data.text_normalization import normalize_text
from arabic_asr.evaluation.metrics import corpus_scores


def rescore(hyps: List, lm, lm_weight: float, word_bonus: float, level: str = "light") -> List:
    """Fill ``lm_logprob``/``score`` on each hypothesis and return them best-first."""
    for h in hyps:
        words = normalize_text(h.text, level)
        h.lm_logprob = lm.score(words) if lm is not None else 0.0
        h.score = h.am_logprob + lm_weight * h.lm_logprob + word_bonus * len(words.split())
    return sorted(hyps, key=lambda h: h.score, reverse=True)


def grid_search(nbest: Sequence[List], refs: Sequence[str], lm, lm_weights: Sequence[float],
                word_bonuses: Sequence[float], level: str = "light") -> Dict:
    """Pick (lm_weight, word_bonus) minimising corpus WER over pre-computed n-best lists."""
    # LM scores do not depend on the weights: compute them once.
    texts = [[normalize_text(h.text, level) for h in hyps] for hyps in nbest]
    am = [np.array([h.am_logprob for h in hyps]) for hyps in nbest]
    lm_scores = [np.array([lm.score(t) for t in ts]) for ts in texts]
    lengths = [np.array([len(t.split()) for t in ts]) for ts in texts]
    refs_norm = [normalize_text(r, level) for r in refs]
    baseline = corpus_scores(refs_norm, [ts[0] if ts else "" for ts in texts])["wer"]
    table = []
    best = {"lm_weight": 0.0, "word_bonus": 0.0, "wer": baseline}
    for lw in lm_weights:
        for wb in word_bonuses:
            picks = [ts[int(np.argmax(a + lw * l + wb * n))] if ts else ""
                     for ts, a, l, n in zip(texts, am, lm_scores, lengths)]
            wer = corpus_scores(refs_norm, picks)["wer"]
            table.append({"lm_weight": lw, "word_bonus": wb, "wer": wer})
            if wer < best["wer"] - 1e-9:
                best = {"lm_weight": lw, "word_bonus": wb, "wer": wer}
    return {"baseline_wer": baseline, "best": best, "grid": table}
