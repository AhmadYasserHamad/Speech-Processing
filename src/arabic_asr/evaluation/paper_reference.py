"""Reference numbers from Ozyilmaz et al. (Interspeech 2025) and their released result files.

All values are WER/CER in %, scored on raw (unnormalised) text, greedy decoding, the
authors' test sets. Use them only against our ``raw`` level.

Sources:
* Paper Fig. 1 / Phase 1 report Table (RQ1): mean over three random 80% subsets of the
  Common Voice test set.
* ``results/ex_trainsize/results_common_voice_whisper-{small,large-v3}.json``: full test set.
* ``results/ex_finetune/results_whisper-small-finetune_all_seed42.json``: pooled model, seed 42.
* Paper Fig. 4 and Phase 1 report: cross-dialect WER with the MSA step, mean of 3 seeds.
"""

MSA_TRAINSIZE = {  # model -> (WER, CER), Common Voice test (mean of three 80% subsets)
    "whisper-tiny": (101.05, 54.56),
    "whisper-base": (95.92, 49.97),
    "whisper-small": (61.93, 26.83),
    "whisper-medium": (52.84, 22.25),
    "small+20% MSA": (51.66, 22.72),
    "small+40% MSA": (49.68, 22.09),
    "small+60% MSA": (48.50, 21.50),
    "small+80% MSA": (47.12, 20.68),
    "small+100% MSA": (45.93, 20.11),
    "whisper-large-v3": (43.20, 16.90),
}

MSA_FULL_TEST = {  # released result files, full 10,478-utterance test set
    "openai/whisper-small": (61.83, 26.63),
    "openai/whisper-large-v3": (43.07, 16.80),
}

# otozz/whisper-small-finetune_all_seed42 (pooled, with MSA step), released result file.
POOLED_SEED42 = {
    "egyptian": (77.96, 47.97),
    "gulf": (84.92, 60.27),
    "iraqi": (87.02, 54.03),
    "levantine": (76.88, 50.46),
    "maghrebi": (94.74, 64.99),
    "msa": (56.18, 22.62),
}

# Cross-dialect WER with MSA pre-training, mean of three seeds (rows: trained on).
CROSS_DIALECT_WER = {
    "egyptian": {"egyptian": 72.15, "gulf": 92.06, "iraqi": 97.61, "levantine": 90.35, "maghrebi": 100.40, "msa": 59.58},
    "gulf": {"egyptian": 84.47, "gulf": 81.48, "iraqi": 94.35, "levantine": 82.04, "maghrebi": 95.66, "msa": 58.14},
    "iraqi": {"egyptian": 88.41, "gulf": 93.94, "iraqi": 79.96, "levantine": 89.67, "maghrebi": 102.15, "msa": 63.67},
    "levantine": {"egyptian": 82.38, "gulf": 88.22, "iraqi": 93.74, "levantine": 74.33, "maghrebi": 92.69, "msa": 55.48},
    "maghrebi": {"egyptian": 87.29, "gulf": 94.75, "iraqi": 100.46, "levantine": 87.52, "maghrebi": 86.38, "msa": 59.26},
    "pooled": {"egyptian": 79.57, "gulf": 88.39, "iraqi": 90.31, "levantine": 79.02, "maghrebi": 97.31, "msa": 56.14},
}

INFERENCE_SECONDS_PER_UTTERANCE = {"whisper-small": 0.12, "whisper-large-v3": 0.29}  # paper Sec. 4.1, A100

# Which reference to show next to a given checkpoint in reports (raw level only).
CHECKPOINT_REFERENCE = {
    "openai/whisper-small": {"msa": MSA_FULL_TEST["openai/whisper-small"]},
    "openai/whisper-large-v3": {"msa": MSA_FULL_TEST["openai/whisper-large-v3"]},
    "otozz/whisper-small-ar_tsize_1.0": {"msa": MSA_TRAINSIZE["small+100% MSA"]},
    "otozz/whisper-small-finetune_all_seed42": POOLED_SEED42,
}
