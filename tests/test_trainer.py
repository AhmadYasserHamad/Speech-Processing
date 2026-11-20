"""Training-loop tests with a tiny randomly initialised Whisper (no model download)."""

import json

import pytest
import torch

from arabic_asr.config import DataConfig, ExperimentConfig, ModelConfig, TrainingConfig
from arabic_asr.data.dataset import build_experiment_data
from arabic_asr.model.hub import load_feature_extractor
from arabic_asr.model.whisper_model import build_model
from arabic_asr.training.checkpointing import find_resumable
from arabic_asr.training.trainer import Trainer


def _cfg(tiny_cache, model_dir, out_dir, max_steps=4):
    return ExperimentConfig(
        name="unit",
        output_dir=str(out_dir),
        data=DataConfig(cache_dir=str(tiny_cache), train_sources=["egyptian"], msa_replay_fraction=0.25,
                        val_fraction=0.25, val_loss_samples=4, val_wer_samples=2),
        model=ModelConfig(name_or_path=str(model_dir), gradient_checkpointing=True),
        training=TrainingConfig(max_steps=max_steps, batch_size=2, eval_batch_size=2, warmup_steps=1,
                                eval_every=2, log_every=1, precision="fp32", num_workers=0,
                                val_max_new_tokens=4, early_stopping_patience=5),
    )


def _trainer(cfg, tokenizer, extractor):
    data = build_experiment_data(cfg.data, tokenizer, extractor)
    resume = find_resumable(cfg.output_dir)
    model = build_model(cfg.model, source=str(resume) if resume else None)
    fe = load_feature_extractor(cfg.model.name_or_path)
    return Trainer(cfg, model, tokenizer, fe, data, torch.device("cpu"))


def test_training_checkpoints_and_resume(tiny_cache, tiny_model_dir, tmp_path, tokenizer, extractor):
    out = tmp_path / "run"
    summary = _trainer(_cfg(tiny_cache, tiny_model_dir, out, 4), tokenizer, extractor).train()
    assert summary["steps"] == 4
    assert (out / "best" / "model.safetensors").exists()
    assert (out / "best" / "preprocessor_config.json").exists()
    assert (out / "train_summary.json").exists()
    records = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    evals = [r for r in records if r["kind"] == "eval"]
    assert [r["step"] for r in evals] == [2, 4]
    assert all(torch.isfinite(torch.tensor(r["val_loss"])) for r in evals)

    # Optimizer state is dropped after a finished run; simulate an interrupted run instead.
    out2 = tmp_path / "run2"
    cfg = _cfg(tiny_cache, tiny_model_dir, out2, 2)
    _trainer(cfg, tokenizer, extractor).train()
    # pretend the run was cut off: restore a resumable state by training 2 steps with max 6
    cfg6 = _cfg(tiny_cache, tiny_model_dir, out2, 6)
    trainer = _trainer(cfg6, tokenizer, extractor)
    # no optimizer state -> starts fresh but still runs to completion
    assert trainer.train()["steps"] == 6


def test_resume_continues_from_saved_step(tiny_cache, tiny_model_dir, tmp_path, tokenizer, extractor, monkeypatch):
    out = tmp_path / "run"
    cfg = _cfg(tiny_cache, tiny_model_dir, out, 6)
    trainer = _trainer(cfg, tokenizer, extractor)
    # Interrupt after the first evaluation (step 2) by making _after_eval raise once.
    original = trainer._save_last

    def save_then_crash():
        original()
        raise KeyboardInterrupt

    monkeypatch.setattr(trainer, "_save_last", save_then_crash)
    with pytest.raises(KeyboardInterrupt):
        trainer.train()
    assert find_resumable(out) is not None

    resumed = _trainer(cfg, tokenizer, extractor)
    assert resumed.resume_if_possible()
    assert resumed.state.step == 2 and resumed.state.epoch_offset == 4
    summary = resumed.train()
    assert summary["steps"] == 6


def test_early_stopping_logic(tiny_cache, tiny_model_dir, tmp_path, tokenizer, extractor):
    cfg = _cfg(tiny_cache, tiny_model_dir, tmp_path / "run", 4)
    cfg.training.early_stopping_patience = 2
    trainer = _trainer(cfg, tokenizer, extractor)
    assert trainer._after_eval({"val_loss": 1.0}) is False   # first value is the best
    assert trainer._after_eval({"val_loss": 0.9995}) is False  # improvement < threshold -> bad eval 1
    assert trainer._after_eval({"val_loss": 1.2}) is True      # bad eval 2 -> stop
    assert trainer.state.best_metric == 1.0 and trainer.state.stopped_early
