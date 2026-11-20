#!/usr/bin/env python
"""Tune LM rescoring weights on the validation set (never on test).

    python scripts/tune_lm.py --config configs/e2_pooled.yaml --model runs/e2_pooled/best \
        --lm lm/arabic_3gram.pkl.gz --num-beams 5 --max-samples 300

Decodes the experiment's validation subset with beam search, keeps the n-best lists, then
grid-searches ``lm_weight`` and ``word_bonus``. Writes the grid and the best setting to
``--out`` (JSON). Use the best values with evaluate.py / transcribe.py.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402
from tqdm import tqdm  # noqa: E402

from arabic_asr.config import load_experiment_config  # noqa: E402
from arabic_asr.data.dataset import WhisperCollator, build_experiment_data  # noqa: E402
from arabic_asr.inference.lm import load_lm  # noqa: E402
from arabic_asr.inference.rescoring import grid_search  # noqa: E402
from arabic_asr.inference.transcriber import Transcriber  # noqa: E402
from arabic_asr.utils import setup_logging, write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--lm", required=True)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=225)
    parser.add_argument("--lm-weights", type=float, nargs="*", default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    parser.add_argument("--word-bonuses", type=float, nargs="*", default=[0.0, 0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=Path("results/lm_tuning.json"))
    args = parser.parse_args()
    logger = setup_logging()

    cfg = load_experiment_config(args.config)
    transcriber = Transcriber(args.model, device=args.device, num_beams=args.num_beams, n_best=args.num_beams,
                              batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)
    data = build_experiment_data(cfg.data, transcriber.tokenizer, transcriber.extractor)
    dataset = data.val_loss  # larger stratified validation subset
    if len(dataset) > args.max_samples:
        keep = np.sort(np.random.default_rng(0).choice(len(dataset), args.max_samples, replace=False))
        dataset = Subset(dataset, keep.tolist())
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=WhisperCollator(transcriber.prompt[0]))
    nbest, refs = [], []
    for batch in tqdm(loader, desc="beam search"):
        nbest.extend(transcriber.decode_features(batch["input_features"]))
        refs.extend(batch["texts"])
    lm = load_lm(args.lm)
    result = grid_search(nbest, refs, lm, args.lm_weights, args.word_bonuses)
    result.update(model=args.model, lm=args.lm, num_beams=args.num_beams, n_utterances=len(refs))
    write_json(result, args.out)
    logger.info("Beam-search WER without LM: %.2f | best with LM: %.2f (lm_weight=%.2f, word_bonus=%.2f)",
                result["baseline_wer"], result["best"]["wer"], result["best"]["lm_weight"],
                result["best"]["word_bonus"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
