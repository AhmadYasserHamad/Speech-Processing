#!/usr/bin/env python
"""Train one experiment (E1, E2, E3 or the CPU smoke test).

    python scripts/train.py --config configs/e1_msa.yaml
    python scripts/train.py --config configs/e2_pooled.yaml --set training.max_steps=6000

Re-running the same command after a disconnect resumes from ``<output_dir>/last``.
Outputs in ``output_dir``: best/ (model to use), last/ (resume state), train.log,
metrics.jsonl, tensorboard/, config.yaml, data_summary.json, splits.json, train_summary.json.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from arabic_asr.config import load_experiment_config, save_config  # noqa: E402
from arabic_asr.data.dataset import build_experiment_data  # noqa: E402
from arabic_asr.data.features import LogMelExtractor  # noqa: E402
from arabic_asr.data.prepare import load_tokenizer  # noqa: E402
from arabic_asr.model.hub import load_feature_extractor, push_to_hub  # noqa: E402
from arabic_asr.model.whisper_model import architecture_summary, build_model  # noqa: E402
from arabic_asr.training.checkpointing import find_resumable  # noqa: E402
from arabic_asr.training.trainer import Trainer  # noqa: E402
from arabic_asr.utils import get_device, set_seed, setup_logging, write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override config values, e.g. training.max_steps=100")
    parser.add_argument("--device", default=None, help="cuda, cpu, ... (default: auto)")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config, args.set)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir / "train.log")
    save_config(cfg, out_dir / "config.yaml")
    set_seed(cfg.training.seed)
    device = get_device(args.device)

    tokenizer = load_tokenizer(cfg.model.tokenizer_name, cfg.model.language, cfg.model.task)
    feature_extractor = load_feature_extractor(cfg.model.name_or_path, fallback=cfg.model.tokenizer_name)
    extractor = LogMelExtractor(mel_filters=feature_extractor.mel_filters)
    data = build_experiment_data(cfg.data, tokenizer, extractor, train_seed=cfg.training.seed)
    write_json(data.summary, out_dir / "data_summary.json")
    write_json(data.manifest, out_dir / "splits.json")

    resume_dir = find_resumable(out_dir) if cfg.training.resume else None
    model = build_model(cfg.model, source=str(resume_dir) if resume_dir else None, for_training=True)
    if model.config.num_mel_bins != extractor.n_mels:
        raise ValueError(f"Model expects {model.config.num_mel_bins} mel bins, extractor gives {extractor.n_mels}")
    write_json(architecture_summary(model), out_dir / "architecture.json")
    logger.info("Starting from %s", resume_dir or cfg.model.name_or_path)

    trainer = Trainer(cfg, model, tokenizer, feature_extractor, data, device)
    trainer.train()

    if cfg.hub.push:
        if not cfg.hub.repo_id:
            raise ValueError("hub.push is true but hub.repo_id is empty")
        push_to_hub(out_dir / "best", cfg.hub.repo_id, private=cfg.hub.private,
                    message=f"{cfg.name}: best checkpoint")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
