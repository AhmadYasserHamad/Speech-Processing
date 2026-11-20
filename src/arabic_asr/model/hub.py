"""Loading feature extractors and publishing checkpoints to the Hugging Face Hub."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from arabic_asr.utils import get_logger, read_json

MODEL_CARD = """---
language: ar
license: apache-2.0
base_model: {base_model}
tags: [automatic-speech-recognition, whisper, arabic, dialectal-arabic]
---

# {name}

Whisper-small fine-tuned for Arabic speech recognition (MSA + Egyptian, Gulf, Iraqi,
Levantine and Maghrebi dialects). Reimplementation of Ozyilmaz, Coler and Valdenegro-Toro,
"Overcoming Data Scarcity in Multi-Dialectal Arabic ASR via Whisper Fine-Tuning",
Interspeech 2025, with Arabic text normalisation adapted from Grigoryan et al., ICASSP 2025.

* Training data: {data}
* Targets: light-normalised Arabic (no diacritics or punctuation)
* Best checkpoint: step {step}, {metric} = {value}

Usage:

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="{repo_id}")
print(asr("audio.wav", generate_kwargs={{"language": "arabic", "task": "transcribe"}})["text"])
```
"""


def load_feature_extractor(name_or_path: str, fallback: str = "openai/whisper-small"):
    from transformers import WhisperFeatureExtractor

    try:
        return WhisperFeatureExtractor.from_pretrained(name_or_path)
    except (OSError, ValueError):
        get_logger().info("No feature extractor in %s; using %s", name_or_path, fallback)
        return WhisperFeatureExtractor.from_pretrained(fallback)


def write_model_card(model_dir: Path, repo_id: str) -> None:
    info: Dict = {}
    info_path = Path(model_dir) / "training_info.json"
    if info_path.exists():
        info = read_json(info_path)
    cfg = info.get("config", {})
    metric = info.get("early_stopping_metric", "val_loss")
    card = MODEL_CARD.format(
        base_model=cfg.get("model", {}).get("name_or_path", "openai/whisper-small"),
        name=repo_id.split("/")[-1],
        data=", ".join(cfg.get("data", {}).get("train_sources", [])) or "see training_info.json",
        step=info.get("step", "?"),
        metric=metric,
        value=info.get("metrics", {}).get(metric, "?"),
        repo_id=repo_id,
    )
    (Path(model_dir) / "README.md").write_text(card, encoding="utf-8")


def push_to_hub(model_dir: Path, repo_id: str, private: bool = True, message: Optional[str] = None) -> str:
    """Upload a checkpoint directory. Needs a token (``huggingface-cli login`` or HF_TOKEN)."""
    from huggingface_hub import HfApi

    write_model_card(model_dir, repo_id)
    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(folder_path=str(model_dir), repo_id=repo_id,
                      commit_message=message or f"Upload {Path(model_dir).name}")
    url = f"https://huggingface.co/{repo_id}"
    get_logger().info("Pushed %s to %s", model_dir, url)
    return url
