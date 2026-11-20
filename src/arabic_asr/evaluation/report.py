"""Turn one or more evaluation runs into tables, figures and a Markdown report.

Figures use a validated colour-vision-deficiency-safe categorical palette (fixed slot order,
never cycled), a single-hue blue ramp for ordered/sequential encodings, thin marks and
recessive axes. Every figure has a matching CSV table, so no number is conveyed by colour
alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from arabic_asr.data.text_normalization import LEVELS
from arabic_asr.evaluation import error_analysis as ea
from arabic_asr.evaluation.paper_reference import CHECKPOINT_REFERENCE
from arabic_asr.evaluation.statistics import holm_bonferroni, paired_bootstrap, wilcoxon_paired
from arabic_asr.utils import read_json, write_json

# Categorical slots (light mode), in fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
             "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
LEVEL_COLORS = {"raw": "#86b6ef", "light": "#2a78d6", "strict": "#104281"}  # ordinal: light -> dark
SURFACE, INK, INK_2, RULE = "#fcfcfb", "#0b0b0b", "#52514e", "#dddcd7"
TEST_SET_ORDER = ["msa", "egyptian", "gulf", "iraqi", "levantine", "maghrebi"]


# --------------------------------------------------------------------------- loading
def load_run(run_dir: Path) -> Dict:
    run_dir = Path(run_dir)
    summary = read_json(run_dir / "summary.json")
    predictions = pd.read_csv(run_dir / "predictions.csv.gz", keep_default_na=False,
                              na_values={"confidence": ["", "nan"]})
    for col in ("has_latin", "has_music", "finished"):
        if col in predictions:
            predictions[col] = predictions[col].astype(str).str.lower().isin(["true", "1"])
    return {"dir": run_dir, "summary": summary, "name": summary.get("name") or run_dir.name,
            "predictions": ea.add_level_columns(predictions)}


def _ordered(test_sets: Sequence[str]) -> List[str]:
    known = [t for t in TEST_SET_ORDER if t in test_sets]
    return known + [t for t in test_sets if t not in known]


# --------------------------------------------------------------------------- markdown
def markdown_table(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "-" if np.isnan(v) else floatfmt.format(v)
        return str(v)

    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep] + body)


# --------------------------------------------------------------------------- figures
def _axes(figsize=(7.5, 3.6)):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.yaxis.grid(True, color=RULE, linewidth=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def _finish(fig, ax, title: str, ylabel: str, path: Path, legend: bool = True) -> None:
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=10)
    ax.set_ylabel(ylabel, fontsize=8, color=INK_2)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, ncols=min(4, len(ax.get_legend_handles_labels()[0])),
                  loc="upper left", bbox_to_anchor=(0, -0.12))
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def _grouped_bars(ax, categories: List[str], series: Dict[str, List[float]], colors: List[str]) -> None:
    n = len(series)
    width = 0.8 / max(1, n)
    x = np.arange(len(categories))
    for i, (label, values) in enumerate(series.items()):
        ax.bar(x - 0.4 + width * (i + 0.5), values, width=width, color=colors[i], label=label,
               edgecolor=SURFACE, linewidth=1.5)
    ax.set_xticks(x, categories)


def plot_wer_by_model(table: pd.DataFrame, level: str, path: Path) -> None:
    """Grouped bars: test sets on x, one bar per model (categorical slots in run order)."""
    data = table[table["level"] == level]
    models = list(dict.fromkeys(data["model"]))
    test_sets = _ordered(list(dict.fromkeys(data["test_set"])))
    series = {m: [float(data[(data.model == m) & (data.test_set == t)]["wer"].mean()) for t in test_sets]
              for m in models}
    fig, ax = _axes()
    _grouped_bars(ax, test_sets, series, SERIES)
    _finish(fig, ax, f"WER by test set ({level} normalisation)", "WER (%)", path)


def plot_normalisation_effect(table: pd.DataFrame, model: str, path: Path) -> None:
    data = table[table["model"] == model]
    test_sets = _ordered(list(dict.fromkeys(data["test_set"])))
    series = {lvl: [float(data[(data.level == lvl) & (data.test_set == t)]["wer"].mean()) for t in test_sets]
              for lvl in LEVELS}
    fig, ax = _axes()
    _grouped_bars(ax, test_sets, series, [LEVEL_COLORS[lvl] for lvl in LEVELS])
    _finish(fig, ax, f"{model}: WER under raw, light and strict normalisation", "WER (%)", path)


def plot_heatmap(matrix: pd.DataFrame, path: Path, title: str) -> None:
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    fig, ax = _axes(figsize=(1.1 * matrix.shape[1] + 2.5, 0.55 * matrix.shape[0] + 1.6))
    ax.grid(False)
    values = matrix.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    vmin, vmax = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=0)
    ax.set_yticks(range(matrix.shape[0]), matrix.index)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if np.isfinite(v):
                dark = (v - vmin) / max(1e-9, vmax - vmin) > 0.55
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8, color="#ffffff" if dark else INK)
    cbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=7, colors=INK_2, length=0)
    _finish(fig, ax, title, "", path, legend=False)


def plot_buckets(buckets: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = _axes()
    test_sets = _ordered(list(dict.fromkeys(buckets["test_set"])))
    labels = list(buckets["bucket"].cat.categories) if hasattr(buckets["bucket"], "cat") else \
        list(dict.fromkeys(buckets["bucket"]))
    x = np.arange(len(labels))
    for i, t in enumerate(test_sets):
        g = buckets[buckets.test_set == t].set_index("bucket").reindex(labels)
        ax.plot(x, g["wer"], color=SERIES[i % len(SERIES)], linewidth=2, marker="o", markersize=4, label=t)
    ax.set_xticks(x, labels)
    _finish(fig, ax, title, "WER (%)", path)


def plot_calibration(cal: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = _axes(figsize=(4.6, 4.0))
    ax.plot([0, 1], [0, 1], color=INK_2, linewidth=1, linestyle="--", label="perfect calibration")
    ax.plot(cal["mean_confidence"], cal["mean_accuracy"], color=SERIES[0], linewidth=2, marker="o",
            markersize=5, label="model")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("mean confidence", fontsize=8, color=INK_2)
    _finish(fig, ax, title, "word accuracy (1 - WER)", path)


# --------------------------------------------------------------------------- comparisons
def paired_comparison(run_a: Dict, run_b: Dict, level: str = "light") -> pd.DataFrame:
    """Per test set: WER(b) - WER(a) on the utterances both runs scored."""
    a, b = run_a["predictions"], run_b["predictions"]
    merged = a.merge(b, on=["id", "test_set"], suffixes=("_a", "_b"))
    rows = []
    for test_set, g in merged.groupby("test_set", sort=False):
        g = g[g[f"w_ref_{level}_a"] > 0]
        if g.empty:
            continue
        boot = paired_bootstrap(g[f"w_err_{level}_a"], g[f"w_err_{level}_b"], g[f"w_ref_{level}_a"])
        wil = wilcoxon_paired(g[f"w_err_{level}_a"], g[f"w_err_{level}_b"])
        rows.append({"test_set": test_set, "n_utterances": len(g),
                     f"wer_{run_a['name']}": 100 * g[f"w_err_{level}_a"].sum() / g[f"w_ref_{level}_a"].sum(),
                     f"wer_{run_b['name']}": 100 * g[f"w_err_{level}_b"].sum() / g[f"w_ref_{level}_a"].sum(),
                     "delta": boot["delta"], "delta_ci_low": boot["low"], "delta_ci_high": boot["high"],
                     "bootstrap_p": boot["p_value"], "wilcoxon_p": wil["p_value"]})
    out = pd.DataFrame(rows)
    if len(out):
        out["wilcoxon_p_holm"] = holm_bonferroni(out["wilcoxon_p"].tolist())
    return out


# --------------------------------------------------------------------------- report
def build_report(run_dirs: Sequence[Path], out_dir: Path, primary: Optional[str] = None,
                 compare: Optional[Sequence[str]] = None, level: str = "light") -> Path:
    out_dir = Path(out_dir)
    fig_dir, tab_dir = out_dir / "figures", out_dir / "tables"
    runs = [load_run(Path(d)) for d in run_dirs]
    by_name = {r["name"]: r for r in runs}
    primary_run = by_name.get(primary) if primary else runs[-1]
    md: List[str] = ["# Evaluation report", ""]
    md.append("Runs: " + ", ".join(f"`{r['name']}` ({r['summary']['model']})" for r in runs))
    md.append("")
    md.append("Normalisation levels: **raw** (as scored in Paper 3), **light** (Paper 2-style cleanup; "
              "our main metric), **strict** (light + alef/hamza, ta marbuta, alef maqsura folding).")

    # 1. Overall table
    tab_dir.mkdir(parents=True, exist_ok=True)
    table = pd.concat([ea.summary_by_test_set(r["predictions"]).assign(model=r["name"]) for r in runs])
    table.to_csv(tab_dir / "wer_cer_all.csv", index=False)
    pivot = table.pivot_table(index=["model", "test_set"], columns="level", values="wer").reset_index()
    pivot = pivot[["model", "test_set"] + [lvl for lvl in LEVELS if lvl in pivot.columns]]
    cer = table[table.level == level].set_index(["model", "test_set"])["cer"]
    ci = table[table.level == level].set_index(["model", "test_set"])[["wer_ci_low", "wer_ci_high"]]
    pivot[f"{level} 95% CI"] = [f"{ci.loc[(m, t), 'wer_ci_low']:.1f}-{ci.loc[(m, t), 'wer_ci_high']:.1f}"
                                for m, t in zip(pivot.model, pivot.test_set)]
    pivot[f"CER ({level})"] = [cer.loc[(m, t)] for m, t in zip(pivot.model, pivot.test_set)]
    md += ["", "## 1. WER (%) per test set and normalisation level", "", markdown_table(pivot)]
    plot_wer_by_model(table, level, fig_dir / f"wer_by_model_{level}.png")
    md += ["", f"![WER by model](figures/wer_by_model_{level}.png)"]

    # 2. Paper reference (raw level only)
    ref_rows = []
    for r in runs:
        for test_set, (wer, cer_ref) in CHECKPOINT_REFERENCE.get(r["summary"]["model"], {}).items():
            ours = table[(table.model == r["name"]) & (table.test_set == test_set) & (table.level == "raw")]
            if len(ours):
                ref_rows.append({"model": r["name"], "test_set": test_set, "paper_wer": wer,
                                 "our_wer_raw": float(ours["wer"].iloc[0]), "paper_cer": cer_ref,
                                 "our_cer_raw": float(ours["cer"].iloc[0])})
    if ref_rows:
        ref_df = pd.DataFrame(ref_rows)
        ref_df.to_csv(tab_dir / "paper_reproduction.csv", index=False)
        md += ["", "## 2. Reproduction of the paper's numbers (raw scoring)", "",
               "Paper MSA values for `tsize_1.0` are means over three random 80% subsets; ours use the full "
               "test set, so differences of a few tenths are expected.", "", markdown_table(ref_df)]

    # 3. Orthography
    ortho = pd.concat([ea.orthographic_share(r["predictions"]).assign(model=r["name"]) for r in runs])
    ortho = ortho[["model"] + [c for c in ortho.columns if c != "model"]]
    ortho.to_csv(tab_dir / "orthographic_share.csv", index=False)
    md += ["", "## 3. How much of the error is orthography?", "",
           "`light_drop_pts`: WER points removed by Paper 2-style cleanup; `strict_extra_drop_pts`: further points "
           "removed by letter folding (spelling variants counted as errors under raw scoring).", "",
           markdown_table(ortho)]
    plot_normalisation_effect(table, primary_run["name"], fig_dir / "normalisation_effect.png")
    md += ["", "![Normalisation effect](figures/normalisation_effect.png)"]

    # 4. Error types and hallucinations (primary run)
    p = primary_run["predictions"]
    sdi = ea.summary_by_test_set(p, levels=[level], n_boot=200)[["test_set", "wer", "sub_pct", "del_pct", "ins_pct",
                                                                    "ref_words", "n_utterances", "n_empty_refs"]]
    sdi.to_csv(tab_dir / "error_types.csv", index=False)
    hall = ea.hallucination_flags(p, level)
    hall.to_csv(tab_dir / "hallucinations.csv", index=False)
    md += ["", f"## 4. Error types and hallucinations (`{primary_run['name']}`, {level})", "",
           "Substitutions / deletions / insertions as % of reference words:", "", markdown_table(sdi), "",
           "Share of utterances (%) flagged: WER > 100%, runaway output (> 2x reference + 5 words), "
           "a 3-gram repeated 3+ times, or generation cut at the token limit:", "", markdown_table(hall)]

    # 5. Most frequent errors
    errors = ea.top_errors(p, level, top=20)
    for kind, df in errors.items():
        df.to_csv(tab_dir / f"top_{kind}.csv", index=False)
    md += ["", f"## 5. Most frequent errors (`{primary_run['name']}`, all test sets)", "",
           "Substitutions (reference -> hypothesis):", "", markdown_table(errors["substitutions"].head(15)), "",
           "Deleted words:", "", markdown_table(errors["deletions"].head(10)), "",
           "Inserted words:", "", markdown_table(errors["insertions"].head(10))]

    # 6. Length buckets
    bins = [0, 3, 7, 15, 30, 10_000]
    labels = ["1-3", "4-7", "8-15", "16-30", "31+"]
    p = p.assign(ref_words=p[f"ref_{level}"].str.split().str.len())
    buckets = ea.wer_by_bucket(p, level, "ref_words", bins, labels)
    buckets.to_csv(tab_dir / "wer_by_length.csv", index=False)
    if len(buckets):
        plot_buckets(buckets, fig_dir / "wer_by_length.png", "WER by reference length (words)")
        md += ["", "## 6. WER by utterance length", "", "![WER by length](figures/wer_by_length.png)", "",
               markdown_table(buckets.pivot(index="test_set", columns="bucket", values="wer").reset_index())]

    # 7. Bias slices
    md += ["", f"## 7. Bias and fairness slices (`{primary_run['name']}`, {level})", ""]
    for column, note in (("gender", "Common Voice self-reported gender (MSA only; most clips unlabelled)"),
                         ("age", "Common Voice self-reported age band (MSA only)"),
                         ("has_latin", "utterances whose reference contains Latin script (code-switching proxy)")):
        if column in p and p[column].astype(str).str.len().gt(0).any():
            sl = ea.slice_table(p, level, column)
            if len(sl):
                sl.to_csv(tab_dir / f"slice_{column}.csv", index=False)
                md += [f"**{column}** - {note}:", "", markdown_table(sl), ""]
    md += ["**Dialect gap**: see the WER per variety in section 1; the gap to MSA is the dialect bias "
           "Paper 3 set out to reduce.", ""]

    # 8. Calibration
    cal = ea.calibration_table(p, level)
    if len(cal):
        cal.to_csv(tab_dir / "calibration.csv", index=False)
        plot_calibration(cal, fig_dir / "calibration.png", f"Confidence calibration ({primary_run['name']})")
        md += ["## 8. Are confidence scores meaningful?", "", "![Calibration](figures/calibration.png)", "",
               markdown_table(cal)]

    # 9. Model x test-set matrix
    matrix = table[table.level == level].pivot_table(index="model", columns="test_set", values="wer")
    matrix = matrix[_ordered(list(matrix.columns))].reindex([r["name"] for r in runs])
    matrix.to_csv(tab_dir / "model_by_test_set.csv")
    plot_heatmap(matrix, fig_dir / "model_by_test_set.png", f"WER (%) - model x test set ({level})")
    md += ["", "## 9. Model x test-set matrix", "", "![Matrix](figures/model_by_test_set.png)"]

    # 10. Paired significance
    if compare and len(compare) == 2 and all(c in by_name for c in compare):
        comp = paired_comparison(by_name[compare[0]], by_name[compare[1]], level)
        comp.to_csv(tab_dir / f"paired_{compare[0]}_vs_{compare[1]}.csv", index=False)
        md += ["", f"## 10. Paired comparison: `{compare[1]}` vs `{compare[0]}` ({level})", "",
               "`delta` = WER(second) - WER(first) on identical utterances; negative favours the second model. "
               "p-values: paired bootstrap and Wilcoxon signed-rank (Holm-corrected across test sets).", "",
               markdown_table(comp, floatfmt="{:.4g}")]

    # 11. Timing
    timing_rows = []
    for r in runs:
        for test_set, t in r["summary"].get("timing", {}).items():
            timing_rows.append({"model": r["name"], "test_set": test_set, **t})
    if timing_rows:
        timing = pd.DataFrame(timing_rows)
        timing.to_csv(tab_dir / "timing.csv", index=False)
        md += ["", "## 11. Decoding time (batch evaluation, includes data loading)", "", markdown_table(timing, "{:.3f}")]

    report = out_dir / "report.md"
    report.write_text("\n".join(md) + "\n", encoding="utf-8")
    write_json({"runs": [str(d) for d in run_dirs], "primary": primary_run["name"], "level": level},
               out_dir / "report_info.json")
    return report
