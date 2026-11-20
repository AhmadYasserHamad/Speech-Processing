#!/usr/bin/env python
"""Transcribe audio files (batch inference, 1C4).

    python scripts/transcribe.py --model runs/e2_pooled/best audio1.wav recordings/
    python scripts/transcribe.py --model otozz/whisper-small-finetune_all_seed42 clip.mp3 \
        --num-beams 5 --n-best 3 --lm lm/arabic_3gram.pkl.gz --lm-weight 0.3 --word-bonus 0.5

Prints one line per file and optionally writes JSON Lines with confidences, alternatives
(n-best) and timing.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from arabic_asr.inference.lm import load_lm  # noqa: E402
from arabic_asr.inference.transcriber import Transcriber  # noqa: E402

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".webm"}


def collect(inputs):
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            yield from sorted(p for p in path.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS)
        else:
            yield path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="audio files and/or directories")
    parser.add_argument("--model", required=True, help="checkpoint directory or Hugging Face id")
    parser.add_argument("--num-beams", type=int, default=1, help="1 = greedy decoding")
    parser.add_argument("--n-best", type=int, default=1, help="alternatives to return (<= num-beams)")
    parser.add_argument("--max-new-tokens", type=int, default=225)
    parser.add_argument("--lm", default=None, help="n-gram LM for n-best rescoring")
    parser.add_argument("--lm-weight", type=float, default=0.3)
    parser.add_argument("--word-bonus", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--out", type=Path, default=None, help="write results as JSON Lines")
    args = parser.parse_args()

    lm = load_lm(args.lm) if args.lm else None
    transcriber = Transcriber(args.model, device=args.device, precision=args.precision, num_beams=args.num_beams,
                              n_best=args.n_best, max_new_tokens=args.max_new_tokens, lm=lm,
                              lm_weight=args.lm_weight, word_bonus=args.word_bonus)
    handle = open(args.out, "w", encoding="utf-8") if args.out else None
    try:
        for path in collect(args.inputs):
            result = transcriber.transcribe(path)
            print(f"{path.name}\t{result.confidence:.2f}\t{result.text}")
            for i, alt in enumerate(result.alternatives[1:], start=2):
                print(f"    alt {i}: {alt}")
            if handle:
                handle.write(json.dumps({"file": str(path), **result.to_dict()}, ensure_ascii=False) + "\n")
    finally:
        if handle:
            handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
