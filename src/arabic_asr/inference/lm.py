"""Word n-gram language model with interpolated Kneser-Ney smoothing (1C4: LM integration).

Whisper has no external LM; its decoder is an implicit one trained on web transcripts. We
add a small word-level n-gram LM trained only on our *training* transcripts and use it to
rescore the beam-search n-best list (see ``rescoring.py``). It is pure Python, so it needs
no compiled dependency; a KenLM ARPA/binary model can be used instead if ``kenlm`` is
installed.

Interpolated Kneser-Ney (Chen and Goodman, 1999), for the highest order n::

    P(w | h) = max(c(h, w) - D, 0) / c(h)  +  D * N1+(h .) / c(h) * P_lower(w | h')

Lower orders use continuation counts N1+(. h' w) instead of raw counts, and the unigram
level is interpolated with a uniform distribution over the vocabulary (+ </s> and <unk>).
Discounts are estimated per order as D = n1 / (n1 + 2 n2).
"""

from __future__ import annotations

import gzip
import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

BOS, EOS = "<s>", "</s>"


class KneserNeyLM:
    def __init__(self, order: int = 3):
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        self.counts: List[Dict[Tuple[str, ...], int]] = [dict() for _ in range(order + 1)]
        self.totals: List[Dict[Tuple[str, ...], int]] = [dict() for _ in range(order + 1)]
        self.types: List[Dict[Tuple[str, ...], int]] = [dict() for _ in range(order + 1)]
        self.discounts: List[float] = [0.75] * (order + 1)
        self.vocab_size = 2
        self.n_sentences = 0
        self.n_tokens = 0

    # ------------------------------------------------------------------ training
    def fit(self, sentences: Iterable[str]) -> "KneserNeyLM":
        n = self.order
        highest: Counter = Counter()
        vocab = set()
        for sentence in sentences:
            words = sentence.split()
            if not words:
                continue
            self.n_sentences += 1
            self.n_tokens += len(words) + 1
            vocab.update(words)
            seq = [BOS] * (n - 1) + words + [EOS]
            for i in range(n - 1, len(seq)):
                highest[tuple(seq[i - n + 1 : i + 1])] += 1
        self.counts[n] = dict(highest)
        # Continuation counts: each distinct (k+1)-gram contributes one left context to its suffix.
        for k in range(n - 1, 0, -1):
            cont: Counter = Counter()
            for gram in self.counts[k + 1]:
                cont[gram[1:]] += 1
            self.counts[k] = dict(cont)
        for k in range(1, n + 1):
            values = self.counts[k].values()
            n1 = sum(1 for v in values if v == 1)
            n2 = sum(1 for v in values if v == 2)
            d = n1 / (n1 + 2 * n2) if n1 and n2 else 0.75
            self.discounts[k] = min(0.95, max(0.05, d))
            totals: Counter = Counter()
            types: Counter = Counter()
            for gram, c in self.counts[k].items():
                totals[gram[:-1]] += c
                types[gram[:-1]] += 1
            self.totals[k], self.types[k] = dict(totals), dict(types)
        self.vocab_size = len(vocab) + 2  # + </s> + <unk>
        return self

    # ------------------------------------------------------------------ probabilities
    def _prob(self, k: int, word: str, context: Tuple[str, ...]) -> float:
        if k == 1:
            total = self.totals[1].get((), 0)
            uniform = 1.0 / self.vocab_size
            if total == 0:
                return uniform
            d = self.discounts[1]
            c = self.counts[1].get((word,), 0)
            return max(c - d, 0.0) / total + d * self.types[1][()] / total * uniform
        history = context[len(context) - (k - 1):] if k > 1 else ()
        lower = self._prob(k - 1, word, context)
        total = self.totals[k].get(history, 0)
        if total == 0:
            return lower
        d = self.discounts[k]
        c = self.counts[k].get(history + (word,), 0)
        return max(c - d, 0.0) / total + d * self.types[k][history] / total * lower

    def prob(self, word: str, context: Tuple[str, ...] = ()) -> float:
        context = tuple(context)[-(self.order - 1):] if self.order > 1 else ()
        context = (BOS,) * (self.order - 1 - len(context)) + context
        return self._prob(self.order, word, context)

    def score(self, sentence: str) -> float:
        """Natural-log probability of a whole sentence, including the end-of-sentence token."""
        words = sentence.split()
        seq = [BOS] * (self.order - 1) + words + [EOS]
        total = 0.0
        for i in range(self.order - 1, len(seq)):
            context = tuple(seq[i - self.order + 1 : i])
            total += math.log(max(self._prob(self.order, seq[i], context), 1e-30))
        return total

    def perplexity(self, sentences: Iterable[str]) -> float:
        log_sum, n_tokens = 0.0, 0
        for sentence in sentences:
            if not sentence.split():
                continue
            log_sum += self.score(sentence)
            n_tokens += len(sentence.split()) + 1
        return math.exp(-log_sum / max(1, n_tokens))

    # ------------------------------------------------------------------ persistence
    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wb") as handle:
            pickle.dump(self.__dict__, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path) -> "KneserNeyLM":
        # Only load LM files you created yourself: pickle can execute code.
        with gzip.open(path, "rb") as handle:
            state = pickle.load(handle)
        lm = cls(state["order"])
        lm.__dict__.update(state)
        return lm


class KenLMWrapper:
    """Same ``score`` interface for an ARPA/binary KenLM model (optional dependency)."""

    def __init__(self, path):
        import kenlm  # type: ignore

        self.model = kenlm.Model(str(path))
        self.order = self.model.order

    def score(self, sentence: str) -> float:
        return self.model.score(sentence, bos=True, eos=True) * math.log(10.0)


def load_lm(path):
    path = str(path)
    if path.endswith((".arpa", ".bin", ".klm")):
        return KenLMWrapper(path)
    return KneserNeyLM.load(path)
