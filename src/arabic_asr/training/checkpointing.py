"""Checkpoint saving/loading (1C3: "checkpointing and model saving").

Two checkpoints live in the run directory:

* ``best/`` - weights + tokenizer + feature extractor + ``training_info.json`` of the
  evaluation with the best early-stopping metric. Self-contained: the inference module,
  the demo and ``from_pretrained`` can load it directly.
* ``last/`` - the latest state, including optimizer, LR scheduler, gradient scaler and RNG
  states, so a Colab/Kaggle session that is cut off resumes exactly where it stopped.

Writes go to a temporary directory that is swapped in at the end, so an interrupted save
never corrupts the previous checkpoint.
"""

from __future__ import annotations

import os
import random
import shutil
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from arabic_asr.utils import read_json, write_json

STATE_FILE = "trainer_state.json"


def _swap_in(tmp: Path, final: Path) -> None:
    old = final.with_name(final.name + ".old")
    if old.exists():
        shutil.rmtree(old)
    if final.exists():
        os.replace(final, old)
    os.replace(tmp, final)
    if old.exists():
        shutil.rmtree(old)


def save_model_dir(model, tokenizer, feature_extractor, out_dir: os.PathLike, info: Optional[Dict] = None) -> Path:
    out_dir = Path(out_dir)
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    model.save_pretrained(tmp, safe_serialization=True)
    tokenizer.save_pretrained(tmp)
    if feature_extractor is not None:
        feature_extractor.save_pretrained(tmp)
    if info is not None:
        write_json(info, tmp / "training_info.json")
    _swap_in(tmp, out_dir)
    return out_dir


def rng_state() -> Dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_resumable(model, tokenizer, feature_extractor, optimizer, scheduler, scaler, trainer_state: Dict,
                   out_dir: os.PathLike) -> Path:
    out_dir = Path(out_dir)
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    model.save_pretrained(tmp, safe_serialization=True)
    tokenizer.save_pretrained(tmp)
    if feature_extractor is not None:
        feature_extractor.save_pretrained(tmp)
    torch.save(optimizer.state_dict(), tmp / "optimizer.pt")
    torch.save(scheduler.state_dict(), tmp / "scheduler.pt")
    torch.save(scaler.state_dict(), tmp / "scaler.pt")
    torch.save(rng_state(), tmp / "rng.pt")
    write_json(trainer_state, tmp / STATE_FILE)
    _swap_in(tmp, out_dir)
    return out_dir


def find_resumable(run_dir: os.PathLike) -> Optional[Path]:
    last = Path(run_dir) / "last"
    if (last / STATE_FILE).exists() and (last / "optimizer.pt").exists():
        return last
    return None


def load_resumable(last_dir: os.PathLike, optimizer, scheduler, scaler) -> Dict:
    last_dir = Path(last_dir)
    optimizer.load_state_dict(torch.load(last_dir / "optimizer.pt", map_location="cpu"))
    scheduler.load_state_dict(torch.load(last_dir / "scheduler.pt", map_location="cpu"))
    scaler.load_state_dict(torch.load(last_dir / "scaler.pt", map_location="cpu"))
    try:
        restore_rng_state(torch.load(last_dir / "rng.pt", map_location="cpu", weights_only=False))
    except TypeError:  # torch < 2.0 has no weights_only argument
        restore_rng_state(torch.load(last_dir / "rng.pt", map_location="cpu"))
    return read_json(last_dir / STATE_FILE)


def drop_optimizer_state(last_dir: os.PathLike) -> None:
    """After training finishes, keep ``last/`` loadable but free the ~2 GB of optimizer state."""
    for name in ("optimizer.pt", "scheduler.pt", "scaler.pt", "rng.pt"):
        path = Path(last_dir) / name
        if path.exists():
            path.unlink()
