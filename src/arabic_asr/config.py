"""Typed configuration loaded from YAML.

Two config families exist:

* ``PrepareConfig`` (configs/data.yaml): which Hugging Face datasets to download
  and how to clean/filter them into the local cache.
* ``ExperimentConfig`` (configs/e*.yaml): one training run (data mix, model,
  optimisation, early stopping, hub upload).

String values may reference environment variables with ``${NAME}`` or
``${NAME:-default}``, so the same YAML works on Colab, Kaggle and a laptop.
Any field can be overridden from the command line with ``--set a.b=value``.
"""

from __future__ import annotations

import os
import re
import typing
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(value: Any) -> Any:
    """Expand ``${VAR}`` / ``${VAR:-default}`` and ``~`` inside strings (recursively)."""
    if isinstance(value, str):
        def _replace(match: re.Match) -> str:
            name, default = match.group(1), match.group(2)
            if name in os.environ:
                return os.environ[name]
            return default if default is not None else match.group(0)

        expanded = _ENV_PATTERN.sub(_replace, value)
        return os.path.expanduser(expanded) if expanded.startswith("~") else expanded
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    return value


# --------------------------------------------------------------------------- #
# Data preparation config
# --------------------------------------------------------------------------- #
@dataclass
class SourceSpec:
    """One corpus on the Hugging Face Hub (the authors' preprocessed copies)."""

    kind: str  # "audio" (raw 16 kHz audio + text) or "features" (log-Mel + token ids)
    train_repo: str
    test_repo: str
    variety: str = ""  # human-readable label, e.g. "MSA" or "Egyptian"


@dataclass
class FilterConfig:
    """Training-data filters (never applied to test sets). See docs/PREPROCESSING.md."""

    min_duration_s: float = 0.1
    max_duration_s: float = 30.0
    max_label_tokens: int = 448
    drop_music: bool = True
    speaking_rate_iqr_k: float = 3.0
    leakage_min_words: int = 1


@dataclass
class PrepareConfig:
    cache_dir: str = "cache"
    tokenizer_name: str = "openai/whisper-small"
    sources: Dict[str, SourceSpec] = field(default_factory=dict)
    filters: FilterConfig = field(default_factory=FilterConfig)
    read_batch_rows: int = 32
    download_retries: int = 3


# --------------------------------------------------------------------------- #
# Experiment config
# --------------------------------------------------------------------------- #
@dataclass
class AugmentationConfig:
    enabled: bool = True
    # Waveform augmentation (only possible for sources that ship raw audio, i.e. MSA)
    speed_perturb_prob: float = 0.5
    speed_factors: List[float] = field(default_factory=lambda: [0.9, 1.1])
    gain_prob: float = 0.3
    gain_db: List[float] = field(default_factory=lambda: [-6.0, 6.0])
    noise_prob: float = 0.3
    noise_snr_db: List[float] = field(default_factory=lambda: [15.0, 40.0])
    # SpecAugment on log-Mel features (all sources)
    specaugment_prob: float = 1.0
    freq_masks: int = 2
    freq_mask_width: int = 15
    time_masks: int = 4
    time_mask_max_ratio: float = 0.05


@dataclass
class DataConfig:
    cache_dir: str = "cache"
    train_sources: List[str] = field(default_factory=lambda: ["msa"])
    msa_replay_fraction: float = 0.0
    text_level: str = "light"
    val_fraction: float = 0.2
    split_seed: int = 42
    speaker_disjoint_msa: bool = True
    max_train_samples_per_source: Optional[int] = None
    val_loss_samples: int = 2000
    val_wer_samples: int = 400
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


@dataclass
class ModelConfig:
    name_or_path: str = "openai/whisper-small"
    tokenizer_name: str = "openai/whisper-small"
    language: str = "arabic"
    task: str = "transcribe"
    gradient_checkpointing: bool = True
    freeze_encoder: bool = False


@dataclass
class TrainingConfig:
    seed: int = 42
    max_steps: int = 5000
    batch_size: int = 8
    eval_batch_size: int = 16
    grad_accum_steps: int = 1
    learning_rate: float = 1.0e-5
    warmup_steps: int = 500
    adam_betas: List[float] = field(default_factory=lambda: [0.9, 0.99])
    adam_eps: float = 1.0e-8
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    precision: str = "auto"  # auto | fp16 | bf16 | fp32
    eval_every: int = 250
    log_every: int = 25
    early_stopping_metric: str = "val_loss"  # val_loss | val_wer
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    num_workers: int = 2
    val_max_new_tokens: int = 160
    resume: bool = True
    save_resumable: bool = True


@dataclass
class HubConfig:
    push: bool = False
    repo_id: Optional[str] = None
    private: bool = True


@dataclass
class ExperimentConfig:
    name: str = "experiment"
    output_dir: str = "runs/experiment"
    description: str = ""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    hub: HubConfig = field(default_factory=HubConfig)


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
def _build(cls, data: Optional[dict]):
    """Recursively build dataclass ``cls`` from a plain dict, rejecting unknown keys."""
    data = data or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected a mapping for {cls.__name__}, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        hint = hints[name]
        origin = typing.get_origin(hint)
        args = typing.get_args(hint)
        if is_dataclass(hint):
            kwargs[name] = _build(hint, value)
        elif origin in (dict, Dict) and args and is_dataclass(args[1]):
            kwargs[name] = {k: _build(args[1], v) for k, v in (value or {}).items()}
        else:
            kwargs[name] = expand_env(value)
    return cls(**kwargs)


def _apply_overrides(raw: dict, overrides: Optional[List[str]]) -> dict:
    """Apply ``a.b.c=value`` overrides; values are parsed as YAML (numbers, bools, lists)."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.path=value, got {item!r}")
        key, value = item.split("=", 1)
        node = raw
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    return raw


def load_yaml(path: os.PathLike) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_prepare_config(path: os.PathLike, overrides: Optional[List[str]] = None) -> PrepareConfig:
    return _build(PrepareConfig, _apply_overrides(load_yaml(path), overrides))


def load_experiment_config(path: os.PathLike, overrides: Optional[List[str]] = None) -> ExperimentConfig:
    return _build(ExperimentConfig, _apply_overrides(load_yaml(path), overrides))


def config_to_dict(cfg) -> dict:
    return asdict(cfg)


def save_config(cfg, path: os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config_to_dict(cfg), handle, allow_unicode=True, sort_keys=False)
