import math

import numpy as np
import pytest

from arabic_asr.inference.lm import KneserNeyLM
from arabic_asr.inference.rescoring import grid_search, rescore
from arabic_asr.inference.transcriber import Hypothesis, Transcriber, split_long_audio

CORPUS = [
    "ذهبت الى المدرسة صباحا",
    "ذهبت الى السوق مساء",
    "المدرسة كبيرة جدا",
    "الجو جميل اليوم",
    "ذهبت الى البيت",
]


def test_kneser_ney_distribution_sums_to_one():
    lm = KneserNeyLM(order=3).fit(CORPUS)
    vocab = sorted({w for s in CORPUS for w in s.split()}) + ["</s>", "<unk-word>"]
    for context in [(), ("ذهبت",), ("ذهبت", "الى"), ("كلمة", "مجهولة")]:
        total = sum(lm.prob(w, context) for w in vocab)
        assert total == pytest.approx(1.0, abs=1e-9)


def test_kneser_ney_prefers_seen_word_order_and_roundtrips(tmp_path):
    lm = KneserNeyLM(order=3).fit(CORPUS)
    assert lm.score("ذهبت الى المدرسة") > lm.score("المدرسة الى ذهبت")
    assert math.isfinite(lm.perplexity(["ذهبت الى الجامعة"]))
    path = tmp_path / "lm.pkl.gz"
    lm.save(path)
    assert KneserNeyLM.load(path).score("ذهبت الى المدرسة") == pytest.approx(lm.score("ذهبت الى المدرسة"))


def test_rescoring_and_grid_search():
    lm = KneserNeyLM(order=2).fit(CORPUS)
    hyps = [Hypothesis("المدرسة الى ذهبت", am_logprob=-2.0, n_tokens=4),
            Hypothesis("ذهبت الى المدرسة", am_logprob=-2.5, n_tokens=4)]
    assert rescore(list(hyps), lm, lm_weight=0.0, word_bonus=0.0)[0].text == "المدرسة الى ذهبت"
    assert rescore(list(hyps), lm, lm_weight=1.0, word_bonus=0.0)[0].text == "ذهبت الى المدرسة"
    result = grid_search([hyps], ["ذهبت الى المدرسة"], lm, [0.0, 1.0], [0.0])
    assert result["best"]["lm_weight"] == 1.0 and result["best"]["wer"] == 0.0


def test_split_long_audio_respects_limit():
    audio = np.random.default_rng(0).standard_normal(16000 * 75).astype(np.float32) * 0.1
    audio[16000 * 27 : 16000 * 27 + 1600] = 0.0  # a pause just before the 30 s limit
    bounds = split_long_audio(audio)
    assert bounds[0][0] == 0 and bounds[-1][1] == len(audio)
    assert all(0 < e - s <= 30 * 16000 for s, e in bounds)
    assert 27 * 16000 <= bounds[0][1] <= 27 * 16000 + 1600
    assert split_long_audio(audio[: 16000 * 10]) == [(0, 16000 * 10)]


def test_transcriber_end_to_end_with_tiny_model(tiny_model_dir):
    tr = Transcriber(str(tiny_model_dir), device="cpu", num_beams=2, n_best=2, max_new_tokens=5,
                     tokenizer_name="openai/whisper-small")
    t = np.arange(16000 * 2) / 16000
    result = tr.transcribe((0.1 * np.sin(2 * np.pi * 300 * t)).astype(np.float32), sample_rate=16000)
    assert len(result.chunks) == 1 and len(result.chunks[0].hypotheses) == 2
    assert 1 <= len(result.alternatives) <= 2  # identical beam texts are merged
    hyp = result.chunks[0].hypotheses[0]
    assert hyp.n_tokens >= 1 and 0.0 < hyp.confidence <= 1.0
    assert result.to_dict()["real_time_factor"] > 0
