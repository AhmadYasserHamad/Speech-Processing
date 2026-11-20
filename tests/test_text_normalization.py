from arabic_asr.data.text_normalization import (
    ArabicTextNormalizer,
    has_diacritics,
    has_latin,
    has_music,
    normalize_text,
)


def test_raw_only_collapses_whitespace():
    assert normalize_text("  مرحبا،   بكم  ", "raw") == "مرحبا، بكم"


def test_light_strips_diacritics_and_punctuation():
    # "He loves her, and she loves him too." with shadda and a full stop.
    assert normalize_text("هو يحبّها و هي تحبّه أيضا.", "light") == "هو يحبها و هي تحبه أيضا"
    # Tanween on "غداً" is removed, the hamza seat (أ) is kept at the light level.
    assert normalize_text("من الممكن أنها لن تأتي غداً.", "light") == "من الممكن أنها لن تأتي غدا"


def test_light_handles_tatweel_digits_and_symbols():
    assert normalize_text("جمـــيل", "light") == "جميل"
    assert normalize_text("عام ٢٠٢٥ و ۱۲", "light") == "عام 2025 و 12"
    assert normalize_text("♪ وأول حضن إنت حسيته ♪", "light") == "وأول حضن إنت حسيته"
    assert normalize_text("قال: «نعم»؟!", "light") == "قال نعم"


def test_light_nfkc_presentation_forms_and_ligatures():
    # U+FEFB is the lam-alef ligature; NFKC expands it to two letters.
    assert normalize_text("ﻻ", "light") == "لا"
    # Presentation forms of "سلام" -> base letters.
    assert normalize_text("ﺳﻠﺎﻡ", "light") == "سلام"


def test_light_removes_invisible_format_characters_and_brackets():
    assert normalize_text("كتاب‏‌", "light") == "كتاب"
    assert normalize_text("[موسيقى] أهلا", "light") == "أهلا"


def test_light_keeps_and_lowercases_latin_code_switching():
    assert normalize_text("c'est vraiment BIEN يا خويا", "light") == "c est vraiment bien يا خويا"


def test_strict_folds_letters():
    assert normalize_text("إن شاء الله", "strict") == "ان شاء الله"
    assert normalize_text("مدرسة على آخر", "strict") == "مدرسه علي اخر"
    # Light keeps the letters distinct (Paper 2's readability choice).
    assert normalize_text("مدرسة على آخر", "light") == "مدرسة على آخر"


def test_normalisation_is_idempotent():
    samples = ["هو يحبّها و هي تحبّه أيضا.", "♪ وأول حضن ♪", "عام ٢٠٢٥", "مدرسة على آخر"]
    for level in ("raw", "light", "strict"):
        for s in samples:
            once = normalize_text(s, level)
            assert normalize_text(once, level) == once


def test_empty_and_symbol_only_inputs():
    assert normalize_text("", "light") == ""
    assert normalize_text("♪ ♪", "light") == ""
    assert normalize_text("...", "strict") == ""


def test_flags():
    assert has_latin("salut يا")
    assert not has_latin("مرحبا")
    assert has_music("♪ غناء ♪")
    assert has_diacritics("تحبّه")
    assert not has_diacritics("تحبه")


def test_callable_wrapper():
    assert ArabicTextNormalizer("strict")("أحمد") == "احمد"
