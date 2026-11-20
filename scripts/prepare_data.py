#!/usr/bin/env python
"""Download and preprocess the datasets into the local cache (1C1).

    # MSA (Common Voice 16.1): ~5 GB of audio columns streamed, ~6 GB cache
    python scripts/prepare_data.py --config configs/data.yaml --sources msa
    # the five MASC dialects: ~20 GB downloaded shard by shard, a few GB cache
    python scripts/prepare_data.py --config configs/data.yaml --sources egyptian gulf iraqi levantine maghrebi
    # tiny subset for a laptop smoke test
    python scripts/prepare_data.py --config configs/data.yaml --sources msa egyptian --max-rows 48
    # change filter thresholds in configs/data.yaml, then re-apply without downloading
    python scripts/prepare_data.py --config configs/data.yaml --sources msa --refilter-only

Already prepared splits are skipped unless --overwrite is given. Prints an overview table.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from arabic_asr.config import load_prepare_config  # noqa: E402
from arabic_asr.data.prepare import cache_overview, prepare_sources, refilter_split  # noqa: E402
from arabic_asr.utils import setup_logging, write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--splits", nargs="+", default=["test", "train"], choices=["test", "train"])
    parser.add_argument("--max-rows", type=int, default=None, help="only the first N rows per split (smoke tests)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--refilter-only", action="store_true", help="re-run training filters on existing cache")
    args = parser.parse_args()

    cfg = load_prepare_config(args.config, args.set)
    logger = setup_logging(Path(cfg.cache_dir) / "prepare.log")
    if args.refilter_only:
        for source in args.sources:
            info = refilter_split(cfg, source)
            logger.info("[%s/train] kept %d rows (%.2f h): %s", source, info["rows_kept"], info["hours_kept"],
                        info["filters"])
    else:
        prepare_sources(cfg, args.sources, splits=args.splits, max_rows=args.max_rows, overwrite=args.overwrite)

    overview = cache_overview(cfg.cache_dir)
    if len(overview):
        logger.info("Cache overview (%s):\n%s", cfg.cache_dir, overview.to_string(index=False))
        write_json(overview.to_dict(orient="records"), Path(cfg.cache_dir) / "overview.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
