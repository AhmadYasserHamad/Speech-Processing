"""Multi-dialect Arabic ASR: a reimplementation of Ozyilmaz, Coler and Valdenegro-Toro,
"Overcoming Data Scarcity in Multi-Dialectal Arabic ASR via Whisper Fine-Tuning"
(Interspeech 2025), with Arabic text normalisation adapted from Grigoryan et al.
(ICASSP 2025)."""

__version__ = "0.1.0"

DIALECTS = ("egyptian", "gulf", "iraqi", "levantine", "maghrebi")
ALL_SOURCES = ("msa",) + DIALECTS
