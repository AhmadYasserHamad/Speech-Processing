import numpy as np
import pandas as pd

from arabic_asr.config import FilterConfig
from arabic_asr.data.filtering import assign_filter_reasons, filter_summary


def _meta(n=60, seed=0):
    rng = np.random.default_rng(seed)
    durations = rng.uniform(2.0, 8.0, size=n)
    texts = ["كلمة " * int(d * 2) for d in durations]  # roughly constant speaking rate
    return pd.DataFrame(
        {
            "text": [t.strip() for t in texts],
            "has_music": [False] * n,
            "duration": durations,
            "n_tokens": [20] * n,
        }
    )


def test_each_rule_is_assigned_first_match():
    meta = _meta()
    meta.loc[0, "text"] = ""
    meta.loc[1, "has_music"] = True
    meta.loc[2, "duration"] = 0.05
    meta.loc[3, "duration"] = 31.0
    meta.loc[4, "n_tokens"] = 500
    meta.loc[5, "text"] = "كلمة " * 400  # absurd speaking rate
    meta.loc[6, "text"] = "جملة الاختبار المشتركة"  # 20 letters
    meta.loc[6, "duration"] = 2.5  # ~8 letters/s, same rate as the synthetic rows
    out = assign_filter_reasons(meta, FilterConfig(), test_texts=["جملة الاختبار المشتركة"])
    assert list(out.loc[:6, "filter_reason"]) == [
        "empty_text",
        "music",
        "too_short",
        "too_long",
        "too_many_tokens",
        "speaking_rate_outlier",
        "test_overlap",
    ]
    assert (out.loc[7:, "filter_reason"] == "").all()


def test_music_rule_can_be_disabled_and_summary_counts():
    meta = _meta()
    meta.loc[1, "has_music"] = True
    out = assign_filter_reasons(meta, FilterConfig(drop_music=False))
    assert (out["filter_reason"] == "").all()
    summary = filter_summary(out)
    assert summary["kept"]["rows"] == len(meta)


def test_leakage_min_words():
    meta = _meta()
    meta.loc[0, "text"] = "نعم"
    meta.loc[0, "duration"] = 0.4  # keep the speaking rate normal so only leakage is tested
    out =assign_filter_reasons(meta, FilterConfig(leakage_min_words=2), test_texts=["نعم"])
    assert out.loc[0, "filter_reason"] == ""
