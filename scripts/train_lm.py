#!/usr/bin/env python
"""Train the word n-gram LM on training transcripts only.

    python scripts/train_lm.py --config configs/e2_pooled.yaml --out lm/arabic_3gram.pkl.gz

The experiment config is used to rebuild exactly the same train/validation split as
training, so no validation (and of course no test) sentence leaks into the LM. Texts are
light-normalised, matching the model's output after normalisation. The script reports
perplexity on the validation sentences.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from arabic_asr.config import load_experiment_config
from arabic_asr.data.cache import CacheSplit
from arabic_asr.data.splits import train_val_rows
from arabic_asr.inference.lm import KneserNeyLM
from arabic_asr.utils import setup_logging, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--sources", nargs="*", default=None,
                        help="default: the config's train sources, plus msa if it uses MSA replay")
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("lm/arabic_3gram.pkl.gz"))
    args = parser.parse_args()
    logger = setup_logging()

    cfg = load_experiment_config(args.config, args.set)
    sources = args.sources or list(cfg.data.train_sources)
    if args.sources is None and cfg.data.msa_replay_fraction > 0 and "msa" not in sources:
        sources.append("msa")

    train_texts, val_texts, per_source = [], [], {}
    for source in sources:
        split = CacheSplit(cfg.data.cache_dir, source, "train")
        tr, va = train_val_rows(split, cfg.data)
        texts = split.texts("light")
        train_texts += [texts[i] for i in tr]
        val_texts += [texts[i] for i in va]
        per_source[source] = {"train_sentences": len(tr), "val_sentences": len(va)}

    lm = KneserNeyLM(order=args.order).fit(train_texts)
    ppl = lm.perplexity(val_texts)
    lm.save(args.out)
    report = {"sources": per_source, "order": args.order, "train_sentences": lm.n_sentences,
              "train_tokens": lm.n_tokens, "vocab_size": lm.vocab_size, "val_perplexity": ppl,
              "discounts": lm.discounts[1:]}
    write_json(report, args.out.with_suffix("").with_suffix(".json"))
    logger.info("Saved %d-gram LM to %s | vocab %d | validation perplexity %.1f", args.order, args.out,
                lm.vocab_size, ppl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
