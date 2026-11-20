"""Evaluate one checkpoint on the prepared test sets (1C5).

Writes to ``out_dir``:

* ``predictions.csv.gz`` - one row per utterance: raw reference and hypothesis, confidence,
  duration and metadata (gender/age for MSA, Latin-script and music flags);
* ``summary.json`` / ``summary.csv`` - corpus WER/CER per test set at raw/light/strict
  levels with bootstrap 95% CIs, error-type shares, timing, and the paper's reference
  numbers for the same checkpoint when available;
* ``nbest.jsonl`` (optional) - full n-best lists with acoustic and LM scores.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from arabic_asr.data.dataset import WhisperCollator, build_eval_dataset
from arabic_asr.evaluation.error_analysis import add_level_columns, summary_by_test_set
from arabic_asr.evaluation.paper_reference import CHECKPOINT_REFERENCE
from arabic_asr.inference.lm import load_lm
from arabic_asr.inference.transcriber import Transcriber
from arabic_asr.utils import get_logger, write_json

META_COLUMNS = ("duration", "gender", "age", "accent", "has_latin", "has_music", "has_diacritics")


def evaluate_checkpoint(
    model: str,
    test_sets: Sequence[str],
    cache_dir: str,
    out_dir: str,
    name: Optional[str] = None,
    split: str = "test",
    max_samples: Optional[int] = None,
    batch_size: int = 16,
    num_beams: int = 1,
    n_best: int = 1,
    max_new_tokens: int = 225,
    lm_path: Optional[str] = None,
    lm_weight: float = 0.0,
    word_bonus: float = 0.0,
    device: Optional[str] = None,
    precision: str = "auto",
    num_workers: int = 2,
    save_nbest: bool = False,
    seed: int = 0,
    n_boot: int = 1000,
) -> Dict:
    logger = get_logger()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lm = load_lm(lm_path) if lm_path else None
    transcriber = Transcriber(model, device=device, precision=precision, num_beams=num_beams, n_best=n_best,
                              max_new_tokens=max_new_tokens, batch_size=batch_size, compute_scores=True,
                              lm=lm, lm_weight=lm_weight, word_bonus=word_bonus)
    collator = WhisperCollator(transcriber.prompt[0])
    rows: List[Dict] = []
    timing: Dict[str, Dict] = {}
    nbest_handle = open(out / "nbest.jsonl", "w", encoding="utf-8") if save_nbest else None
    try:
        for source in test_sets:
            try:
                dataset = build_eval_dataset(cache_dir, source, transcriber.tokenizer, transcriber.extractor,
                                             split=split, max_samples=max_samples, seed=seed)
            except (ValueError, FileNotFoundError) as error:
                logger.warning("Skipping %s: %s", source, error)
                continue
            meta = dataset.parts[0].meta
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                collate_fn=collator)
            start = time.time()
            for batch in tqdm(loader, desc=f"{name or model} | {source}", leave=False):
                for index, hyps in zip(batch["indices"], transcriber.decode_features(batch["input_features"])):
                    row = int(dataset.rows[index])
                    m = meta.iloc[row]
                    best = hyps[0]
                    record = {
                        "id": m["id"],
                        "test_set": source,
                        "ref": m["text_raw"],
                        "hyp": best.text,
                        "confidence": best.confidence if best.n_tokens else float("nan"),
                        "avg_logprob": best.avg_logprob,
                        "hyp_tokens": best.n_tokens,
                        "finished": best.finished,
                    }
                    record.update({c: m[c] for c in META_COLUMNS if c in m})
                    rows.append(record)
                    if nbest_handle:
                        nbest_handle.write(json.dumps({"id": m["id"], "test_set": source, "ref": m["text_raw"],
                                                       "hypotheses": [h.to_dict() for h in hyps]},
                                                      ensure_ascii=False) + "\n")
            elapsed = time.time() - start
            audio_seconds = float(meta["duration"].to_numpy()[dataset.rows].sum())
            timing[source] = {
                "utterances": len(dataset),
                "audio_hours": round(audio_seconds / 3600, 3),
                "seconds": round(elapsed, 1),
                "seconds_per_utterance": elapsed / max(1, len(dataset)),
                "real_time_factor": elapsed / max(1e-6, audio_seconds),
            }
            logger.info("%s: %d utterances in %.0fs (%.3f s/utt)", source, len(dataset), elapsed,
                        timing[source]["seconds_per_utterance"])
    finally:
        if nbest_handle:
            nbest_handle.close()

    predictions = pd.DataFrame(rows)
    predictions.to_csv(out / "predictions.csv.gz", index=False, encoding="utf-8")
    scored = add_level_columns(predictions)
    table = summary_by_test_set(scored, n_boot=n_boot)
    table.insert(0, "model", name or model)
    table.to_csv(out / "summary.csv", index=False)
    results: Dict[str, Dict] = {}
    for rec in table.to_dict(orient="records"):
        results.setdefault(rec["test_set"], {})[rec["level"]] = {k: v for k, v in rec.items()
                                                                 if k not in ("model", "test_set", "level")}
    summary = {
        "model": model,
        "name": name or model,
        "settings": {"split": split, "max_samples": max_samples, "batch_size": batch_size, "num_beams": num_beams,
                     "n_best": n_best, "max_new_tokens": max_new_tokens, "lm": lm_path, "lm_weight": lm_weight,
                     "word_bonus": word_bonus, "precision": transcriber.precision, "device": str(transcriber.device)},
        "timing": timing,
        "results": results,
        "paper_reference_raw": {k: {"wer": v[0], "cer": v[1]}
                                for k, v in CHECKPOINT_REFERENCE.get(model, {}).items()},
    }
    write_json(summary, out / "summary.json")
    print_summary(table, summary["paper_reference_raw"])
    return summary


def print_summary(table: pd.DataFrame, reference: Dict) -> None:
    logger = get_logger()
    pivot = table.pivot(index="test_set", columns="level", values="wer")
    pivot = pivot[[c for c in ("raw", "light", "strict") if c in pivot.columns]]
    lines = ["WER (%) by normalisation level" + (" | paper (raw)" if reference else "")]
    for test_set, row in pivot.iterrows():
        ref = reference.get(test_set, {}).get("wer")
        values = "  ".join(f"{lvl} {row[lvl]:6.2f}" for lvl in pivot.columns)
        lines.append(f"  {test_set:10s} {values}" + (f"  | {ref:6.2f}" if ref is not None else ""))
    logger.info("\n".join(lines))
