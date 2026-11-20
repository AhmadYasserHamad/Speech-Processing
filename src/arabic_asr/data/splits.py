"""Train/validation splits of the authors' training sets.

* MSA (Common Voice): speaker-disjoint 80/20 split on ``client_id``. Paper 3 split utterances
  at random, so the same speaker could appear in training and validation, which makes the
  validation loss optimistic and early stopping less reliable.
* MASC dialects: seeded random 80/20 split, as in Paper 3 (no speaker ids are released).

The authors' test sets are never touched here. Splits depend only on the seed and the kept
rows, so E1, E2 and E3 all see the same MSA validation speakers.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from arabic_asr.config import DataConfig
from arabic_asr.data.cache import CacheSplit


def random_split(rows: np.ndarray, val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(rows)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(rows))
    n_val = int(round(val_fraction * len(rows)))
    if len(rows) > 1:
        n_val = min(max(n_val, 1), len(rows) - 1)
    val = np.sort(rows[perm[:n_val]])
    train = np.sort(rows[perm[n_val:]])
    return train, val


def speaker_disjoint_split(rows: np.ndarray, speakers: Sequence[str], val_fraction: float,
                           seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Assign whole speakers to validation until it holds ~``val_fraction`` of utterances.

    Speakers are visited in random order; a speaker that would push validation more than 5%
    past the target is skipped (Common Voice has a few very prolific contributors).
    Utterances with an empty speaker id are treated as separate speakers.
    """
    rows = np.asarray(rows)
    speakers = np.asarray([s if s else f"__row{r}" for s, r in zip(speakers, rows)], dtype=object)
    unique, counts = np.unique(speakers, return_counts=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    target = val_fraction * len(rows)
    chosen, total = set(), 0
    for idx in order:
        if total >= target * 0.95:
            break
        if total + counts[idx] > target * 1.05 and total > 0:
            continue
        chosen.add(unique[idx])
        total += counts[idx]
    is_val = np.array([s in chosen for s in speakers], dtype=bool)
    if len(rows) > 1 and (is_val.all() or not is_val.any()):
        return random_split(rows, val_fraction, seed)
    return np.sort(rows[~is_val]), np.sort(rows[is_val])


def train_val_rows(split: CacheSplit, cfg: DataConfig) -> Tuple[np.ndarray, np.ndarray]:
    kept = split.kept_rows()
    if split.source == "msa" and cfg.speaker_disjoint_msa:
        speakers = split.meta["client_id"].fillna("").to_numpy()[kept]
        return speaker_disjoint_split(kept, speakers, cfg.val_fraction, cfg.split_seed)
    return random_split(kept, cfg.val_fraction, cfg.split_seed)
