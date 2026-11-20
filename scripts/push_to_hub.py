#!/usr/bin/env python
"""Upload a trained checkpoint (e.g. runs/e2_pooled/best) to your Hugging Face account.

    huggingface-cli login            # or set HF_TOKEN; in a notebook: notebook_login()
    python scripts/push_to_hub.py runs/e2_pooled/best your-username/whisper-small-ar-dialects

A short model card (README.md) is generated from training_info.json. Repos are private
unless --public is given.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from arabic_asr.model.hub import push_to_hub
from arabic_asr.utils import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("repo_id")
    parser.add_argument("--public", action="store_true")
    args = parser.parse_args()
    setup_logging()
    if not (args.model_dir / "config.json").exists():
        raise SystemExit(f"{args.model_dir} does not look like a checkpoint directory")
    push_to_hub(args.model_dir, args.repo_id, private=not args.public)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
