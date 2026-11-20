# Experiments run book (Phase 2)

All GPU work runs on Google Colab (free T4) or Kaggle (P100/T4, 30 GPU-hours a week). The
notebook `notebooks/run_on_colab_kaggle.ipynb` contains every command below in order; this
file explains what each step is for and what to record.

## Environment variables

| Variable | Colab | Kaggle | Purpose |
|---|---|---|---|
| `ASR_CACHE` | `/content/cache` (local disk, fast) | `/tmp/cache` | prepared data (~10 GB for everything) |
| `ASR_RUNS` | `/content/drive/MyDrive/arabic-asr/runs` | `/kaggle/working/runs` | checkpoints and logs; on Drive so they survive disconnects |

Re-running a training command after a disconnect resumes from `<run>/last/` (model,
optimizer, scheduler, scaler and random states). The prepared cache is rebuilt in each new
Colab session (a few minutes for MSA, roughly 15-25 min for all dialects; it depends on download speed).

## Order of work

| Step | Command (see the notebook) | Output | Rough T4 time* |
|---|---|---|---|
| 1. Tests | `pytest` | all green | 1 min |
| 2. Prepare MSA | `prepare_data.py --sources msa` | `cache/msa/{train,test}` | 10-20 min |
| 3. E0a baselines on MSA | `evaluate.py --model openai/whisper-small --test-sets msa` and `otozz/whisper-small-ar_tsize_1.0` | `results/e0_*` | 30-45 min each |
| 4. E1 train | `train.py --config configs/e1_msa.yaml` | `runs/e1_msa/best` | 2-4 h |
| 5. E1 evaluate | `evaluate.py --model $ASR_RUNS/e1_msa/best` | `results/e1_msa` | 1 h (all test sets) |
| 6. Prepare dialects | `prepare_data.py --sources egyptian gulf iraqi levantine maghrebi` | `cache/<dialect>/...` | 15-25 min |
| 7. E0b paper's pooled model | `evaluate.py --model otozz/whisper-small-finetune_all_seed42` | `results/e0_paper_pooled` | 1 h |
| 8. E2 train | `train.py --config configs/e2_pooled.yaml` | `runs/e2_pooled/best` | 3-5 h |
| 9. E2 evaluate | `evaluate.py --model $ASR_RUNS/e2_pooled/best` | `results/e2_pooled` | 1 h |
| 10. LM | `train_lm.py`, `tune_lm.py`, `evaluate.py --num-beams 5 --lm ...` (subset) | `lm/`, `results/e2_beam_lm` | 1 h |
| 11. Speed | `benchmark_speed.py --models $ASR_RUNS/e1_msa/best openai/whisper-large-v3` | `results/speed_benchmark.json` | 20 min |
| 12. Report | `analyze.py results/e0_* results/e1_msa results/e2_pooled ...` | `reports/main/report.md` + figures | 2 min |
| 13. (optional) E3 | `train.py --config configs/e3_egyptian.yaml`, then evaluate | `runs/e3_egyptian` | 3 h + 1 h |

\* Estimates, not measurements. Check the real speed in the first `train.log` lines
(`samples/s`) and adjust.

Do E0 before training anything. If our evaluation of `otozz/whisper-small-ar_tsize_1.0` at
the `raw` level does not land near the paper's 45.9% WER on MSA, something in the pipeline
differs and must be fixed before spending GPU hours.

## What each experiment answers

| ID | Question | Compare against |
|---|---|---|
| E0a | Does our evaluation reproduce the paper? How much of its MSA error is orthography? | Paper Fig. 1: zero-shot small 61.9%, fine-tuned 45.9%, large-v3 43.2% (raw) |
| E1 | Does Paper 2-style text cleanup plus augmentation improve the paper's MSA model? | E0a `paper_msa` at all three levels (paired test) |
| E0b | How far are the paper's dialect numbers inflated by spelling? | Paper pooled row: 78-95% raw WER per dialect |
| E2 | Our main model: does it beat the paper's pooled model on dialects without losing MSA? | E0b (paired test per test set), E1 on MSA |
| E3 | Specific vs. pooled under our preprocessing (paper's RQ3) | E2 on Egyptian |
| LM | Does n-gram rescoring of beam n-best lists help? | E2 greedy on the same subset |
| Speed | Is small really ~2.4x faster than large-v3? | Paper: 0.12 s vs 0.29 s per utterance on an A100 |

## Hyperparameters (paper vs. paper's code vs. ours)

| Setting | Paper text | Paper's code | Ours |
|---|---|---|---|
| Optimizer | AdamW, β = (0.9, 0.99) | AdamW, β₂ left at 0.999 | AdamW (0.9, 0.99) |
| Learning rate | 1e-5, 500 warm-up steps, linear decay | same | same |
| Max steps / batch size | 6,000 / 8 | 5,000 / 8 | 5,000 / 8 (E2: 8,000) |
| Evaluation | every 250 steps | every 250 steps, full validation set decoded | every 250 steps (E2: 500): loss on up to 2,000-2,400 validation utterances, WER on 360-400 |
| Early stopping / checkpoint selection | validation loss, patience 3 | WER, patience 3, threshold 0.001 | validation loss, patience 3, threshold 0.001 |
| Precision | not stated | fp32 (A100) | fp16 on T4, bf16 on A100 |
| Text targets | raw | raw | light-normalised |
| Augmentation | none | none | speed/gain/noise (MSA) + SpecAugment |
| Validation split (MSA) | random 20% | random 20% | speaker-disjoint 20% |
| Seeds | 3 | 3 (finetune script: 1 left) | 1 (budget); state as a limitation |

## Deliverables checklist

* Code: this repository.
* Trained weights: `runs/*/best` on Drive; push E2 (and E1) with `scripts/push_to_hub.py`.
* Results: `results/*/summary.json` and `summary.csv` (small, committed);
  `predictions.csv.gz` per run.
* Report figures and tables: `reports/main/`.
