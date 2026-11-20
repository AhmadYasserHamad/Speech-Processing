#!/usr/bin/env python
"""Inference speed and size: whisper-small (fine-tuned) vs larger models (1C5 efficiency).

    python scripts/benchmark_speed.py --models runs/e1_msa/best openai/whisper-large-v3 --n 200

Uses MSA test audio (raw audio is needed so large-v3 can compute its 128-bin features).
Reports parameters, peak GPU memory, latency per utterance at batch size 1, throughput at
a larger batch size, and real-time factor. Paper 3 reports 0.12 s (small) vs 0.29 s
(large-v3) per utterance on an A100.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from arabic_asr.config import expand_env  # noqa: E402
from arabic_asr.data.cache import CacheSplit  # noqa: E402
from arabic_asr.inference.transcriber import Transcriber  # noqa: E402
from arabic_asr.model.whisper_model import count_parameters  # noqa: E402
from arabic_asr.utils import setup_logging, write_json  # noqa: E402


def time_decode(transcriber: Transcriber, waves, batch_size: int) -> float:
    feats = torch.stack([transcriber.extractor(w)[0] for w in waves])
    if transcriber.device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    for i in range(0, len(waves), batch_size):
        transcriber.decode_features(feats[i : i + batch_size])
    if transcriber.device.type == "cuda":
        torch.cuda.synchronize()
    return time.time() - start


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--cache-dir", default=expand_env("${ASR_CACHE:-cache}"))
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=Path("results/speed_benchmark.json"))
    args = parser.parse_args()
    logger = setup_logging()

    split = CacheSplit(args.cache_dir, "msa", "test")
    rows = np.sort(np.random.default_rng(0).choice(len(split), size=min(args.n, len(split)), replace=False))
    waves = [split.audio(int(r)) for r in rows]
    audio_seconds = sum(len(w) for w in waves) / 16000
    results = []
    for model in args.models:
        transcriber = Transcriber(model, device=args.device, compute_scores=False)
        if transcriber.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        time_decode(transcriber, waves[:4], 4)  # warm-up
        latency = time_decode(transcriber, waves, 1)
        throughput = time_decode(transcriber, waves, args.batch_size)
        record = {
            "model": model,
            "parameters_m": count_parameters(transcriber.model)["total"] / 1e6,
            "precision": transcriber.precision,
            "device": str(transcriber.device),
            "utterances": len(waves),
            "audio_seconds": audio_seconds,
            "latency_s_per_utt_bs1": latency / len(waves),
            "throughput_s_per_utt": throughput / len(waves),
            "rtf_bs1": latency / audio_seconds,
            "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9 if transcriber.device.type == "cuda" else None,
        }
        results.append(record)
        logger.info("%s: %.0fM params | %.3f s/utt (bs=1) | %.3f s/utt (bs=%d) | RTF %.3f", model,
                    record["parameters_m"], record["latency_s_per_utt_bs1"], record["throughput_s_per_utt"],
                    args.batch_size, record["rtf_bs1"])
        del transcriber
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json({"results": results, "paper_reference_s_per_utt": {"whisper-small": 0.12, "whisper-large-v3": 0.29}},
               args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
