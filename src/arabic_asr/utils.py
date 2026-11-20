"""Small shared helpers: seeding, devices, mixed precision, logging, JSON IO."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preferred: Optional[str] = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_precision(precision: str, device: torch.device) -> str:
    """Pick fp16/bf16/fp32. Mixed precision is only used on CUDA."""
    precision = (precision or "auto").lower()
    if device.type != "cuda":
        return "fp32"
    if precision == "auto":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if precision not in {"fp16", "bf16", "fp32"}:
        raise ValueError(f"Unknown precision {precision!r}")
    return precision


def autocast_context(precision: str, device: torch.device):
    if device.type != "cuda" or precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def make_grad_scaler(enabled: bool):
    """GradScaler across torch versions (torch.amp.GradScaler appeared in 2.3)."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def setup_logging(log_file: Optional[os.PathLike] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("arabic_asr")
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def get_logger() -> logging.Logger:
    logger = logging.getLogger("arabic_asr")
    if not logger.handlers:
        setup_logging()
    return logger


def write_json(obj: Any, path: os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, default=_json_default)
    os.replace(tmp, path)


def read_json(path: os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj).__name__}")
