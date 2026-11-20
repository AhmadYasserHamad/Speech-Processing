import numpy as np
import pandas as pd
import pytest

from arabic_asr.evaluation import error_analysis as ea
from arabic_asr.evaluation.metrics import corpus_scores, score_all_levels, word_alignment, word_edit_counts
from arabic_asr.evaluation.report import build_report
from arabic_asr.evaluation.statistics import bootstrap_ci, holm_bonferroni, paired_bootstrap
from arabic_asr.utils import write_json


def test_word_edit_counts_and_alignment():
    counts = word_edit_counts("a b c d", "a x c d e")
    assert (counts.substitutions, counts.deletions, counts.insertions, counts.hits) == (1, 0, 1, 3)
    ops = [op for op, _, _ in word_alignment("a b c", "a c")]
    assert ops == ["equal", "delete", "equal"]


def test_corpus_scores_skip_empty_references():
    scores = corpus_scores(["a b", "", "c d e f"], ["a b", "x", "c d"])
    assert scores["n_utterances"] == 2 and scores["n_empty_refs"] == 1
    assert scores["wer"] == pytest.approx(100 * 2 / 6)
    assert scores["deletions"] == 2


def test_levels_remove_orthographic_errors():
    refs = ["ذهبتُ إلى المدرسةِ."]
    hyps = ["ذهبت الى المدرسه"]
    levels = score_all_levels(refs, hyps)
    assert levels["raw"]["wer"] == pytest.approx(100.0)
    assert levels["light"]["wer"] == pytest.approx(100 * 2 / 3)  # إلى/الى and ة/ه remain
    assert levels["strict"]["wer"] == pytest.approx(0.0)


def test_bootstrap_and_paired_tests():
    rng = np.random.default_rng(0)
    totals = rng.integers(5, 15, size=300)
    errors_a = rng.binomial(totals, 0.3)
    errors_b = rng.binomial(totals, 0.2)
    ci = bootstrap_ci(errors_a, totals, n_boot=500)
    assert ci["low"] < ci["value"] < ci["high"]
    comp = paired_bootstrap(errors_a, errors_b, totals, n_boot=500)
    assert comp["delta"] < 0 and comp["high"] < 0 and comp["p_value"] < 0.05
    assert holm_bonferroni([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def _fake_run(path, name, hyp_noise):
    refs = ["ذهبت إلى المدرسة صباحا", "الجو جميل اليوم.", "شكرا جزيلا", "c'est bien يا خويا"] * 30
    hyps = [r if i % hyp_noise else r.replace("إلى", "الى") + " زائد" for i, r in enumerate(refs)]
    df = pd.DataFrame({
        "id": [f"u{i}" for i in range(len(refs))],
        "test_set": ["msa"] * 60 + ["egyptian"] * 60,
        "ref": refs, "hyp": hyps,
        "confidence": np.linspace(0.2, 0.95, len(refs)),
        "finished": True, "duration": 3.0,
        "gender": (["male", "female"] * 60), "age": "twenties",
        "has_latin": [("c'est" in r) for r in refs], "has_music": False,
    })
    path.mkdir(parents=True)
    df.to_csv(path / "predictions.csv.gz", index=False)
    write_json({"model": f"local/{name}", "name": name, "timing": {"msa": {"seconds": 1.0}}}, path / "summary.json")
    return path


def test_report_builds_tables_and_figures(tmp_path):
    a = _fake_run(tmp_path / "a", "baseline", 2)
    b = _fake_run(tmp_path / "b", "ours", 5)
    report = build_report([a, b], tmp_path / "report", primary="ours", compare=["baseline", "ours"])
    text = report.read_text(encoding="utf-8")
    assert "Paired comparison" in text and "orthography" in text
    assert (tmp_path / "report" / "figures" / "wer_by_model_light.png").exists()
    assert (tmp_path / "report" / "tables" / "slice_gender.csv").exists()
    ortho = pd.read_csv(tmp_path / "report" / "tables" / "orthographic_share.csv")
    assert (ortho["wer_raw"] >= ortho["wer_light"]).all() and (ortho["wer_light"] >= ortho["wer_strict"]).all()


def test_hallucination_flags():
    df = ea.add_level_columns(pd.DataFrame({"test_set": ["x", "x"], "ref": ["a b", "a b"],
                                            "hyp": ["a b", "c d c d c d c d c d c d"]}))
    flags = ea.hallucination_flags(df)
    assert flags.loc[0, "flag_wer_over_100"] == pytest.approx(50.0)
    assert flags.loc[0, "flag_repetition"] == pytest.approx(50.0)
