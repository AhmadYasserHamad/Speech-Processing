"""Arabic text normalisation at three levels.

Why three levels (details in docs/PREPROCESSING.md):

* ``raw``    - only whitespace is tidied. This is what Ozyilmaz et al. (Paper 3) scored, so
               it is the level to use when comparing with their numbers.
* ``light``  - the normalisation steps of Grigoryan et al. (Paper 2, Table II) that fix
               *encoding* and *formatting* differences without changing spelling: NFKC,
               Eastern Arabic digits -> Western digits, tatweel removal, diacritic removal,
               punctuation/symbol removal. We train on this level and report it as the
               main metric.
* ``strict`` - ``light`` plus letter folding (alef/hamza variants -> bare alef, ta marbuta
               -> ha, alef maqsura -> ya), the convention of MASC's official test references.
               This is the dialect-aware normalisation Paper 3 names as its missing step
               (Sec. 4.4). Evaluation only: folding would make model output less readable.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

LEVELS = ("raw", "light", "strict")

# Combining marks used in Arabic script: harakat, tanween, shadda, sukun, superscript alef,
# Quranic annotation signs and extended-Arabic combining marks.
_DIACRITICS_RE = re.compile(
    "[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۤۧۨ-ۭ࣓-ࣣ࣡-ࣿ]"
)
_TATWEEL = "ـ"
# Sound-event annotations in subtitles, e.g. "[موسيقى]" or "[Music]".
_BRACKETED_RE = re.compile(r"\[[^\]]*\]")
_WHITESPACE_RE = re.compile(r"\s+")
_LATIN_RE = re.compile(r"[A-Za-z]")
_MUSIC_CHARS = ("♪", "♫", "♬", "♭", "♮", "♯", "\U0001F3B5", "\U0001F3B6")

# Arabic-Indic (U+0660..) and Extended/Persian (U+06F0..) digits -> ASCII digits.
_DIGITS = {ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")}
_DIGITS.update({ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")})

# Encoding variants that look identical but have different code points (Persian keyboards).
_ENCODING_VARIANTS = {
    ord("ک"): "ك",  # Persian keheh -> Arabic kaf
    ord("ی"): "ي",  # Persian yeh -> Arabic yeh
    ord("ۀ"): "ة",  # heh with yeh above -> ta marbuta
}

# Letter folding for the strict level.
_FOLDING = {
    ord("أ"): "ا",  # alef with hamza above -> alef
    ord("إ"): "ا",  # alef with hamza below -> alef
    ord("آ"): "ا",  # alef with madda -> alef
    ord("ٱ"): "ا",  # alef wasla -> alef
    ord("ٲ"): "ا",  # alef with wavy hamza above -> alef
    ord("ٳ"): "ا",  # alef with wavy hamza below -> alef
    ord("ة"): "ه",  # ta marbuta -> ha
    ord("ى"): "ي",  # alef maqsura -> ya
}

_TRANSLATION_LIGHT = {**_DIGITS, **_ENCODING_VARIANTS}


def _replace_punctuation_and_symbols(text: str) -> str:
    """Punctuation (P*), symbols (S*) and control characters become spaces; invisible
    format characters (ZWNJ, RLM, ...) are deleted."""
    out = []
    for ch in text:
        category = unicodedata.category(ch)
        if category[0] in ("P", "S") or category == "Cc":
            out.append(" ")
        elif category == "Cf":
            continue
        else:
            out.append(ch)
    return "".join(out)


def collapse_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalize_light(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)  # presentation forms / ligatures -> base letters
    text = _BRACKETED_RE.sub(" ", text)
    text = text.translate(_TRANSLATION_LIGHT)
    text = text.replace(_TATWEEL, "")
    text = _DIACRITICS_RE.sub("", text)
    text = _replace_punctuation_and_symbols(text)
    text = text.lower()  # only affects Latin script (code-switched words)
    return collapse_whitespace(text)


def normalize_strict(text: str) -> str:
    return normalize_light(text).translate(_FOLDING)


@lru_cache(maxsize=200_000)
def normalize_text(text: str, level: str = "light") -> str:
    if text is None:
        return ""
    if level == "raw":
        return collapse_whitespace(text)
    if level == "light":
        return normalize_light(text)
    if level == "strict":
        return normalize_strict(text)
    raise ValueError(f"Unknown normalisation level {level!r}; expected one of {LEVELS}")


class ArabicTextNormalizer:
    """Callable wrapper so the level can be configured once and passed around."""

    def __init__(self, level: str = "light") -> None:
        if level not in LEVELS:
            raise ValueError(f"Unknown normalisation level {level!r}; expected one of {LEVELS}")
        self.level = level

    def __call__(self, text: str) -> str:
        return normalize_text(text, self.level)


def has_latin(text: str) -> bool:
    """True if the transcript contains Latin letters (a proxy for code-switching)."""
    return bool(_LATIN_RE.search(text or ""))


def has_music(text: str) -> bool:
    """True if a subtitle line marks singing/music (MASC uses the ♪ symbol)."""
    return any(ch in (text or "") for ch in _MUSIC_CHARS)


def has_diacritics(text: str) -> bool:
    return bool(_DIACRITICS_RE.search(unicodedata.normalize("NFKC", text or "")))
