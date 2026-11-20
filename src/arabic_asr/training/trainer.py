"""Custom PyTorch training loop (1C3: training module).

It follows Paper 3's recipe and makes every part explicit:

* loss: token-level cross-entropy with teacher forcing. The decoder input is the label
  sequence shifted right behind <|startoftranscript|>; padded positions (-100) are ignored;
* optimiser: AdamW with beta = (0.9, 0.99) as stated in the paper (the released code left
  beta2 at 0.999), no weight decay;
* learning-rate schedule: linear warm-up over 500 steps to 1e-5, then linear decay to 0 at
  ``max_steps``;
* mixed precision (fp16 + dynamic loss scaling on T4, bf16 on A100), gradient clipping at
  1.0, optional gradient accumulation and gradient checkpointing;
* validation every ``eval_every`` steps: teacher-forced loss on a fixed validation subset
  and greedy-decoding WER/CER on a smaller fixed subset;
* early stopping: stop after ``patience`` evaluations without an improvement larger than
  ``threshold`` in the monitored metric (validation loss by default, as in the paper text);
* checkpointing: ``best/`` whenever the metric improves, ``last/`` (fully resumable) at
  every evaluation.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from transformers.models.whisper.modeling_whisper import shift_tokens_right

from arabic_asr.config import ExperimentConfig, config_to_dict
from arabic_asr.data.dataset import ExperimentData, ResumableRandomSampler, WhisperCollator
from arabic_asr.evaluation.metrics import corpus_scores
from arabic_asr.data.text_normalization import normalize_text
from arabic_asr.model.whisper_model import count_parameters, special_token_ids
from arabic_asr.training import checkpointing as ckpt
from arabic_asr.utils import (
    autocast_context,
    get_logger,
    make_grad_scaler,
    resolve_precision,
    write_json,
)


@dataclass
class TrainState:
    step: int = 0                 # optimiser updates done
    epoch: int = 0
    epoch_offset: int = 0         # samples of the current epoch already consumed
    best_metric: Optional[float] = None
    best_step: int = 0
    bad_evals: int = 0
    stopped_early: bool = False
    train_seconds: float = 0.0
    eval_seconds: float = 0.0
    history: List[Dict] = field(default_factory=list)


class Trainer:
    def __init__(self, cfg: ExperimentConfig, model, tokenizer, feature_extractor, data: ExperimentData,
                 device: torch.device):
        self.cfg = cfg
        self.tcfg = cfg.training
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.data = data
        self.device = device
        self.logger = get_logger()
        self.out_dir = Path(cfg.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        ids = special_token_ids(tokenizer)
        self.sot_id = ids["sot"]
        self.pad_id = model.config.pad_token_id if model.config.pad_token_id is not None else ids["eot"]
        self.collator = WhisperCollator(self.sot_id)

        self.precision = resolve_precision(self.tcfg.precision, device)
        self.scaler = make_grad_scaler(enabled=self.precision == "fp16")

        self.sampler = ResumableRandomSampler(len(data.train), seed=self.tcfg.seed)
        loader_kwargs = dict(collate_fn=self.collator, num_workers=self.tcfg.num_workers,
                             pin_memory=device.type == "cuda")
        self.train_loader = DataLoader(data.train, batch_size=self.tcfg.batch_size, sampler=self.sampler,
                                       drop_last=True, **loader_kwargs)
        self.val_loss_loader = DataLoader(data.val_loss, batch_size=self.tcfg.eval_batch_size, shuffle=False,
                                          **loader_kwargs)
        self.val_wer_loader = DataLoader(data.val_wer, batch_size=self.tcfg.eval_batch_size, shuffle=False,
                                         **loader_kwargs)
        if len(self.train_loader) == 0:
            raise ValueError("Training set is smaller than one batch.")

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=self.tcfg.learning_rate, betas=tuple(self.tcfg.adam_betas),
                                           eps=self.tcfg.adam_eps, weight_decay=self.tcfg.weight_decay)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, self.tcfg.warmup_steps, self.tcfg.max_steps)
        self.state = TrainState()
        self.metrics_path = self.out_dir / "metrics.jsonl"
        self.tb = self._tensorboard()

    # ------------------------------------------------------------------ utilities
    def _tensorboard(self):
        try:
            from torch.utils.tensorboard import SummaryWriter

            return SummaryWriter(log_dir=str(self.out_dir / "tensorboard"))
        except Exception:  # noqa: BLE001 - tensorboard is optional
            return None

    def _log_metrics(self, kind: str, values: Dict) -> None:
        record = {"kind": kind, "step": self.state.step, **values}
        with open(self.metrics_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self.tb is not None:
            for key, value in values.items():
                if isinstance(value, (int, float)) and math.isfinite(value):
                    self.tb.add_scalar(f"{kind}/{key}", value, self.state.step)

    def _info(self, metrics: Dict) -> Dict:
        return {
            "experiment": self.cfg.name,
            "step": self.state.step,
            "epoch": self.state.epoch,
            "metrics": metrics,
            "early_stopping_metric": self.tcfg.early_stopping_metric,
            "config": config_to_dict(self.cfg),
            "data": self.data.summary,
        }

    # ------------------------------------------------------------------ loss
    def compute_loss(self, batch: Dict, reduction: str = "mean"):
        """Teacher-forced token-level cross-entropy. Returns (loss, number of target tokens)."""
        feats = batch["input_features"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)
        decoder_input_ids = shift_tokens_right(labels, self.pad_id, self.sot_id)
        with autocast_context(self.precision, self.device):
            logits = self.model(input_features=feats, decoder_input_ids=decoder_input_ids).logits
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1),
                               ignore_index=-100, reduction=reduction)
        return loss, int((labels != -100).sum().item())

    # ------------------------------------------------------------------ evaluation
    @torch.no_grad()
    def evaluate(self) -> Dict:
        start = time.time()
        self.model.eval()
        total_loss, total_tokens = 0.0, 0
        for batch in self.val_loss_loader:
            loss, n_tokens = self.compute_loss(batch, reduction="sum")
            total_loss += float(loss.item())
            total_tokens += n_tokens
        hyps: List[str] = []
        refs: List[str] = []
        sources: List[str] = []
        for batch in self.val_wer_loader:
            feats = batch["input_features"].to(self.device)
            with autocast_context(self.precision, self.device):
                generated = self.model.generate(
                    input_features=feats,
                    language=self.cfg.model.language,
                    task=self.cfg.model.task,
                    max_new_tokens=self.tcfg.val_max_new_tokens,
                    num_beams=1,
                    use_cache=True,  # training sets config.use_cache=False for checkpointing
                )
            sequences = generated.sequences if hasattr(generated, "sequences") else generated
            hyps.extend(self.tokenizer.batch_decode(sequences, skip_special_tokens=True))
            refs.extend(batch["texts"])
            sources.extend(batch["sources"])
        self.model.train()

        light = corpus_scores([normalize_text(r, "light") for r in refs], [normalize_text(h, "light") for h in hyps])
        strict = corpus_scores([normalize_text(r, "strict") for r in refs], [normalize_text(h, "strict") for h in hyps])
        metrics = {
            "val_loss": total_loss / max(1, total_tokens),
            "val_wer": light["wer"],
            "val_cer": light["cer"],
            "val_wer_strict": strict["wer"],
        }
        for source in sorted(set(sources)):
            idx = [i for i, s in enumerate(sources) if s == source]
            per = corpus_scores([normalize_text(refs[i], "light") for i in idx],
                                [normalize_text(hyps[i], "light") for i in idx])
            metrics[f"val_wer_{source}"] = per["wer"]
        for i in range(min(3, len(hyps))):
            self.logger.info("  [%s] REF: %s", sources[i], refs[i])
            self.logger.info("  [%s] HYP: %s", sources[i], hyps[i])
        self.state.eval_seconds += time.time() - start
        return metrics

    # ------------------------------------------------------------------ checkpoints
    def _trainer_state(self) -> Dict:
        return asdict(self.state)

    def resume_if_possible(self) -> bool:
        if not self.tcfg.resume:
            return False
        last = ckpt.find_resumable(self.out_dir)
        if last is None:
            return False
        state = ckpt.load_resumable(last, self.optimizer, self.scheduler, self.scaler)
        self.state = TrainState(**state)
        self.sampler.set_state(self.state.epoch, self.state.epoch_offset)
        self.logger.info("Resumed from %s at step %d (epoch %d, offset %d)", last, self.state.step,
                         self.state.epoch, self.state.epoch_offset)
        return True

    def _save_last(self) -> None:
        if self.tcfg.save_resumable:
            ckpt.save_resumable(self.model, self.tokenizer, self.feature_extractor, self.optimizer, self.scheduler,
                                self.scaler, self._trainer_state(), self.out_dir / "last")

    def _after_eval(self, metrics: Dict) -> bool:
        """Early-stopping bookkeeping. Returns True if training should stop."""
        name = self.tcfg.early_stopping_metric
        value = metrics[name]
        best = self.state.best_metric
        improved = best is None or (best - value) > self.tcfg.early_stopping_threshold
        if improved:
            self.state.best_metric = float(value)
            self.state.best_step = self.state.step
            self.state.bad_evals = 0
            ckpt.save_model_dir(self.model, self.tokenizer, self.feature_extractor, self.out_dir / "best",
                                self._info(metrics))
            self.logger.info("New best %s = %.4f at step %d -> saved best/", name, value, self.state.step)
        else:
            self.state.bad_evals += 1
            self.logger.info("No improvement in %s (%.4f vs best %.4f); patience %d/%d", name, value,
                             best, self.state.bad_evals, self.tcfg.early_stopping_patience)
        self.state.history.append({"step": self.state.step, **metrics})
        stop = self.state.bad_evals >= self.tcfg.early_stopping_patience
        if stop:
            self.state.stopped_early = True
        return stop

    # ------------------------------------------------------------------ main loop
    def train(self) -> Dict:
        tcfg = self.tcfg
        params = count_parameters(self.model)
        self.logger.info("Experiment %s | device %s | precision %s | params %.1fM (trainable %.1fM)",
                         self.cfg.name, self.device, self.precision, params["total"] / 1e6, params["trainable"] / 1e6)
        self.logger.info("Train rows %d | batch %d x accum %d | max steps %d | eval every %d",
                         len(self.data.train), tcfg.batch_size, tcfg.grad_accum_steps, tcfg.max_steps, tcfg.eval_every)
        self.resume_if_possible()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        window_loss, window_batches, window_start = 0.0, 0, time.time()
        micro = 0
        tick = time.time()  # training wall time includes data loading, excludes evaluation
        stop = self.state.step >= tcfg.max_steps
        while not stop:
            for batch in self.train_loader:
                loss, _ = self.compute_loss(batch)
                self.scaler.scale(loss / tcfg.grad_accum_steps).backward()
                micro += 1
                self.state.epoch_offset += tcfg.batch_size
                window_loss += float(loss.item())
                window_batches += 1
                if micro % tcfg.grad_accum_steps != 0:
                    continue

                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), tcfg.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.state.step += 1
                now = time.time()
                self.state.train_seconds += now - tick
                tick = now

                if self.state.step % tcfg.log_every == 0:
                    elapsed = max(1e-6, time.time() - window_start)
                    values = {
                        "loss": window_loss / max(1, window_batches),
                        "lr": self.scheduler.get_last_lr()[0],
                        "grad_norm": float(grad_norm),
                        "samples_per_s": window_batches * tcfg.batch_size / elapsed,
                        "epoch": self.state.epoch + self.state.epoch_offset / max(1, len(self.data.train)),
                    }
                    if self.device.type == "cuda":
                        values["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
                    self._log_metrics("train", values)
                    self.logger.info("step %5d | loss %.4f | lr %.2e | grad %.2f | %.1f samples/s",
                                     self.state.step, values["loss"], values["lr"], values["grad_norm"],
                                     values["samples_per_s"])
                    window_loss, window_batches, window_start = 0.0, 0, time.time()

                is_last_step = self.state.step >= tcfg.max_steps
                if self.state.step % tcfg.eval_every == 0 or is_last_step:
                    metrics = self.evaluate()
                    self._log_metrics("eval", metrics)
                    self.logger.info("eval @ step %d | %s", self.state.step,
                                     " | ".join(f"{k} {v:.4f}" for k, v in metrics.items()))
                    stop = self._after_eval(metrics)
                    self._save_last()
                    window_start = tick = time.time()
                if stop or is_last_step:
                    stop = True
                    break
            else:
                # epoch finished without interruption
                self.state.epoch += 1
                self.state.epoch_offset = 0
                self.sampler.set_state(self.state.epoch, 0)

        return self._finish()

    def _finish(self) -> Dict:
        summary = {
            "experiment": self.cfg.name,
            "steps": self.state.step,
            "epochs": self.state.epoch + self.state.epoch_offset / max(1, len(self.data.train)),
            "best_step": self.state.best_step,
            "best_metric": self.state.best_metric,
            "early_stopping_metric": self.tcfg.early_stopping_metric,
            "stopped_early": self.state.stopped_early,
            "train_hours": round(self.state.train_seconds / 3600, 3),
            "eval_hours": round(self.state.eval_seconds / 3600, 3),
            "precision": self.precision,
            "device": str(self.device),
            "parameters": count_parameters(self.model),
            "data": self.data.summary,
            "history": self.state.history,
        }
        write_json(summary, self.out_dir / "train_summary.json")
        ckpt.drop_optimizer_state(self.out_dir / "last")
        if self.tb is not None:
            self.tb.close()
        self.logger.info("Done. Best %s %.4f at step %d (stopped early: %s). Best model: %s",
                         self.tcfg.early_stopping_metric, self.state.best_metric or float("nan"),
                         self.state.best_step, self.state.stopped_early, self.out_dir / "best")
        return summary
