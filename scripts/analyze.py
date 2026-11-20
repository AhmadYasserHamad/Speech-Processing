#!/usr/bin/env python
"""Build the evaluation report (tables, figures, Markdown) from evaluation runs.

    python scripts/analyze.py results/e0_paper_msa results/e0_paper_pooled results/e1_msa results/e2_pooled \
        --primary e2_pooled --compare paper_pooled e2_pooled --out reports/main

Sections: WER/CER at three normalisation levels with 95% CIs, reproduction of the paper's
numbers, orthographic share of the error, error types and hallucinations, most frequent
confusions, WER by length, gender/age/code-switching slices, confidence calibration, a
model x test-set matrix and a paired significance test between two runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from arabic_asr.evaluation.report import build_report
from arabic_asr.utils import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="evaluation output directories")
    parser.add_argument("--out", type=Path, default=Path("reports/main"))
    parser.add_argument("--primary", default=None, help="run name used for the detailed sections (default: last)")
    parser.add_argument("--compare", nargs=2, default=None, metavar=("BASELINE", "MODEL"))
    parser.add_argument("--level", default="light", choices=["raw", "light", "strict"])
    args = parser.parse_args()
    logger = setup_logging()
    report = build_report(args.runs, args.out, primary=args.primary, compare=args.compare, level=args.level)
    logger.info("Report written to %s", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
