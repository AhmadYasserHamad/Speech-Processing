#!/usr/bin/env python
"""Evaluate a checkpoint on the test sets (1C5).

    # E0 baselines / reproduction of the paper
    python scripts/evaluate.py --model openai/whisper-small --name zeroshot_small --out results/e0_zeroshot_small
    python scripts/evaluate.py --model otozz/whisper-small-ar_tsize_1.0 --name paper_msa --out results/e0_paper_msa
    python scripts/evaluate.py --model otozz/whisper-small-finetune_all_seed42 --name paper_pooled \
        --out results/e0_paper_pooled
    # our models
    python scripts/evaluate.py --model runs/e2_pooled/best --name e2_pooled --out results/e2_pooled
    # quick look on 200 utterances per test set
    python scripts/evaluate.py --model runs/e1_msa/best --test-sets msa --max-samples 200 --out results/tmp

Large-v3 (128 mel bins) can only be evaluated on MSA: the MASC sets ship 80-bin features.
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from arabic_asr import ALL_SOURCES  # noqa: E402
from arabic_asr.config import expand_env  # noqa: E402
from arabic_asr.evaluation.runner import evaluate_checkpoint  # noqa: E402
from arabic_asr.utils import setup_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="checkpoint directory or Hugging Face id")
    parser.add_argument("--name", default=None, help="short label used in reports")
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache-dir", default=expand_env("${ASR_CACHE:-cache}"))
    parser.add_argument("--test-sets", nargs="+", default=list(ALL_SOURCES))
    parser.add_argument("--split", default="test", choices=["test", "train"])
    parser.add_argument("--max-samples", type=int, default=None, help="random subset per test set")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-beams", type=int, default=1, help="1 = greedy (as in the paper)")
    parser.add_argument("--n-best", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=225, help="paper: generation_max_length=225")
    parser.add_argument("--lm", default=None)
    parser.add_argument("--lm-weight", type=float, default=0.0)
    parser.add_argument("--word-bonus", type=float, default=0.0)
    parser.add_argument("--save-nbest", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    setup_logging()

    evaluate_checkpoint(
        model=args.model, test_sets=args.test_sets, cache_dir=args.cache_dir, out_dir=args.out, name=args.name,
        split=args.split, max_samples=args.max_samples, batch_size=args.batch_size, num_beams=args.num_beams,
        n_best=args.n_best, max_new_tokens=args.max_new_tokens, lm_path=args.lm, lm_weight=args.lm_weight,
        word_bonus=args.word_bonus, device=args.device, precision=args.precision, num_workers=args.num_workers,
        save_nbest=args.save_nbest, seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
