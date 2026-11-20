# Preprocessing pipeline (Phase 2, part 1C1)

This document explains what we do to the audio and text before training and scoring, and
why. Two papers matter here:

* **Paper 3** (Ozyilmaz et al., Interspeech 2025) is the paper we reimplement. It applied
  no Arabic text normalisation and lists that as its main technical limitation (Sec. 4.4).
* **Paper 2** (Grigoryan et al., ICASSP 2025, Table II) has a careful Arabic text pipeline.
  We adopt most of it, but not all; this file says which steps and why.

Code: `src/arabic_asr/data/text_normalization.py`, `filtering.py`, `prepare.py`,
`features.py`, `augmentation.py`. Configuration: `configs/data.yaml`.

---

## 1. The problem we are fixing

Paper 3 computed WER on the raw text: Whisper's output against the transcript exactly as
released. Arabic transcripts vary in ways that have nothing to do with what was said:

* **Diacritics** (short-vowel marks) appear in some transcripts and not others.
* **Punctuation** is inconsistent, and subtitles add symbols such as `♪`.
* **Letter variants**: the same word is written with or without the hamza seat (أ / إ / ا),
  with ta marbuta or ha (ة / ه), or with alef maqsura or ya (ى / ي).

All of these count as full word errors under raw scoring. A sample of the released test
sets shows how common they are:

| Test set | References with diacritics | with punctuation | with Latin script | Other |
|---|---|---|---|---|
| MSA (Common Voice) | 18% | 53% | 0% | - |
| Egyptian (MASC) | 8% | 21% | 2% | song lines marked `♪` |
| Maghrebi (MASC) | 10% | 25% | 7% | - |

Example (unit-tested in `tests/test_evaluation.py`):

| | Text | WER vs. hypothesis `ذهبت الى المدرسه` |
|---|---|---|
| Reference, raw | `ذهبتُ إلى المدرسةِ.` | **100%** (all three words "wrong") |
| Reference, light | `ذهبت إلى المدرسة` | **66.7%** (`إلى`/`الى`, `المدرسة`/`المدرسه` remain) |
| Reference, strict | `ذهبت الى المدرسه` | **0%** (only spelling differed) |

## 2. Three normalisation levels

| Level | What it does | Used for |
|---|---|---|
| `raw` | Whitespace only | Comparing with Paper 3's numbers (same scoring as the paper) |
| `light` | Paper 2's cleanup: Unicode NFKC, Eastern Arabic digits to Western, tatweel removed, diacritics removed, punctuation and symbols removed, `[sound tags]` removed, Persian kaf/yeh code points mapped to Arabic, Latin lower-cased | **Training targets** and **main reported metric** |
| `strict` | `light` plus letter folding: أ إ آ ٱ → ا, ة → ه, ى → ي | Evaluation only: an orthography-insensitive WER |

Normalisation is always applied to **both** the reference and the hypothesis.

Why `light` for training and not `strict`? Folding letters makes the model's output harder
to read (Paper 2's argument), and it is irreversible. With `light` targets the model still
writes أ, ة and ى, and the `strict` score shows how much of the remaining error is spelling.

## 3. Paper 2's steps, one by one

Paper 2, Table II, has three groups of steps: basic, data normalisation and data filtration.

| Paper 2 step | Adopted? | What we do and why |
|---|---|---|
| Drop samples with out-of-alphabet symbols | **Changed** | Dropping samples with Latin letters would delete code-switched speech (7% of Maghrebi test references), which is exactly the speech our Phase 1 report criticised both papers for ignoring. We keep those samples, remove only non-letter symbols by normalisation, and flag `has_latin` so the evaluation can report code-switching separately. |
| Drop clips < 0.1 s or > 20 s | **Adopted, adjusted** | 0.1 s lower bound kept. Upper bound 30 s, Whisper's fixed input window (Paper 2's 20 s suited its own model). Training data only. |
| Eastern Arabic numerals → Western | **Adopted** | `light` level. |
| NeMo text normalisation | **Not adopted** | Requires the NeMo toolkit; numbers are rare here (2-3% of dialect references). Digits are kept as digits on both sides. |
| Decompose ligatures, NFKC | **Adopted** | NFKC maps presentation forms and ligatures (e.g. U+FEFB `ﻻ`) to base letters. |
| Remove rare punctuation / spaces before punctuation / map punctuation to Arabic marks | **Replaced** | We remove all punctuation from targets. Paper 2's best word-accuracy model was its "no punctuation, no diacritics" variant (MASC 8.50% WER, Table V); punctuated output did not change word accuracy. Removing punctuation avoids learning the inconsistent punctuation of subtitles. |
| Remove kashida (tatweel) | **Adopted** | `light` level. |
| Remove diacritics (their MSA model) | **Adopted** | References are inconsistently diacritised (see table above). |
| Keep hamza forms, ta marbuta, alef maqsura distinct | **Adopted for training** | Readable output. Folding is applied only in `strict` evaluation. |
| Drop words-per-second / chars-per-second outliers | **Adopted** | Paper 2 gives no thresholds. We drop training rows outside Tukey fences (Q1 - 3 IQR, Q3 + 3 IQR) of log characters per second, per corpus. This catches misaligned subtitle segments. |
| Model-based filter: drop samples with WER > 60% or CER > 30% under a bootstrap model | **Not adopted** | Those thresholds assume a model near 10% WER. Paper 3's dialect models sit at 72-86% WER, so this filter would delete most dialect data. Worse, it keeps only speech the model already handles, which our Phase 1 report identified as a bias amplifier. |
| Remove train/test sentence overlap (found in Common Voice) | **Adopted** | Training rows whose light-normalised text also appears in that corpus's test set are dropped (`test_overlap`). Common Voice re-uses prompts across speakers, so without this the model can memorise test sentences. |

## 4. Additional steps of our own

* **Music lines**: MASC subtitles of songs (marked `♪`) are dropped from training; singing
  is not the target task, and lyrics subtitles are loose. Test sets keep them.
* **Speaker-disjoint validation split for MSA** on Common Voice `client_id`. Paper 3 split
  utterances at random, so validation speakers were also training speakers. MASC has no
  speaker IDs, so its split stays random (seeded), as in the paper.
* **Test sets are never filtered.** Every rule above applies to training rows only, so our
  test numbers stay directly comparable with the paper's.
* Rows are never deleted: each training row gets a `filter_reason`, so we can report how
  many rows and hours each rule removed (`cache/<source>/train/info.json`). Thresholds can
  be changed and re-applied with `prepare_data.py --refilter-only`.

## 5. Audio and features

| | MSA (Common Voice 16.1) | Dialects (MASC) |
|---|---|---|
| What the authors released | 16 kHz WAV (resampled by them from 48 kHz MP3) plus text and speaker metadata | Only log-Mel features (80 x 3000) and token IDs |
| Loading | WAV decoded with libsndfile, converted to mono, resampled to 16 kHz if needed, stored as int16 | Features stored as float16 with the padding trimmed losslessly. Text is recovered by decoding the token IDs |
| Feature extraction | Our PyTorch log-Mel (25 ms window, 10 ms hop, 80 Slaney mel bands, log10, 80 dB floor, (x+4)/4 scaling). It matches Hugging Face's `WhisperFeatureExtractor` to within 1e-3 (`tests/test_features_audio.py`) | Authors' features, from the same extractor |
| Augmentation (training only) | Speed perturbation 0.9x/1.1x (p = 0.5), random gain ±6 dB (p = 0.3), Gaussian noise at 15-40 dB SNR (p = 0.3), SpecAugment | SpecAugment only (no audio is released) |

SpecAugment: two frequency masks (up to 15 bins) and four time masks (each up to 5% of the
utterance), applied only to real frames, not to the 30 s padding. Paper 3 used no
augmentation; Paper 2 used SpecAugment.

Checks on the real released data (first rows of each test set):

* MASC: rebuilding the 3,000-frame input from our trimmed float16 cache reproduces the
  authors' features to within 0.0005 (float16 rounding). Each utterance takes about 62 KB
  instead of 960 KB.
* MSA: our features, computed from the released WAV, are identical to Hugging Face's
  extractor on the same audio. They differ from the authors' precomputed features by
  0.0001-0.0002 on average, with under 0.4% of bins off by more than 0.01. The authors
  computed their features before the audio was stored as 16-bit WAV, so the small residual
  is quantisation noise in quiet frames.

## 6. Known remaining limitations

* Dialect spelling variants beyond letter folding (e.g. `عشان` / `علشان`) still count as
  errors. Fixing that needs a dialect lexicon or multiple references per utterance; Paper 1
  used multi-reference WER for MGB-3 and MGB-5.
* Hamza on waw/ya (ؤ / ئ) is not folded, and digits vs. spelled-out numbers are not
  reconciled.
* A code-switched word that the model writes in Arabic script instead of Latin counts as an
  error. The evaluation reports this group separately (`has_latin`).
