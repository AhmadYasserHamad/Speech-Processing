"""Whisper model construction (1C2: "model architecture").

The architecture is OpenAI Whisper (Radford et al., 2023) used unchanged, exactly as in
Paper 3: a Transformer encoder over 80-bin log-Mel features (two 1-D convolutions, the
second with stride 2, then sinusoidal positions and 12 pre-norm Transformer blocks for
``small``) and an autoregressive Transformer decoder with cross-attention (12 blocks,
learned positions) whose output layer is tied to its token embedding. The decoder is
prompted with <|startoftranscript|><|ar|><|transcribe|><|notimestamps|>.

Adaptation to Arabic happens only through the weights (full fine-tuning) and the choice of
starting checkpoint: ``openai/whisper-small`` ("scratch" in the paper's terminology) or an
MSA-fine-tuned checkpoint ("with MSA pre-training").
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from transformers import GenerationConfig, WhisperForConditionalGeneration

from arabic_asr.config import ModelConfig
from arabic_asr.utils import get_logger


def special_token_ids(tokenizer) -> Dict[str, int]:
    convert = tokenizer.convert_tokens_to_ids
    return {
        "sot": convert("<|startoftranscript|>"),
        "eot": tokenizer.eos_token_id,
        "ar": convert("<|ar|>"),
        "transcribe": convert("<|transcribe|>"),
        "notimestamps": convert("<|notimestamps|>"),
    }


def configure_generation(model: WhisperForConditionalGeneration, language: str = "arabic",
                         task: str = "transcribe", base_name: str = "openai/whisper-small") -> None:
    """Make ``model.generate`` transcribe Arabic regardless of how the checkpoint was saved."""
    gc = model.generation_config
    if not getattr(gc, "lang_to_id", None) or not getattr(gc, "task_to_id", None):
        # Some fine-tuned checkpoints lose Whisper-specific generation fields; restore them.
        get_logger().info("Restoring Whisper generation config from %s", base_name)
        model.generation_config = GenerationConfig.from_pretrained(base_name)
        gc = model.generation_config
    gc.forced_decoder_ids = None
    gc.language = language
    gc.task = task
    gc.return_timestamps = False
    model.config.forced_decoder_ids = None


def build_model(cfg: ModelConfig, source: Optional[str] = None,
                for_training: bool = True) -> WhisperForConditionalGeneration:
    """Load a Whisper checkpoint (``source`` overrides ``cfg.name_or_path``, e.g. on resume)."""
    path = source or cfg.name_or_path
    model = WhisperForConditionalGeneration.from_pretrained(path)
    configure_generation(model, cfg.language, cfg.task, cfg.tokenizer_name)
    model.config.apply_spec_augment = False  # we apply our own SpecAugment in the data pipeline
    if for_training:
        model.config.use_cache = False
        if cfg.gradient_checkpointing:
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:  # very old transformers
                model.gradient_checkpointing_enable()
        if cfg.freeze_encoder:
            model.model.encoder.requires_grad_(False)
    return model


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def architecture_summary(model: WhisperForConditionalGeneration) -> Dict:
    c = model.config
    return {
        "n_mels": c.num_mel_bins,
        "d_model": c.d_model,
        "encoder_layers": c.encoder_layers,
        "decoder_layers": c.decoder_layers,
        "attention_heads": c.encoder_attention_heads,
        "ffn_dim": c.encoder_ffn_dim,
        "vocab_size": c.vocab_size,
        "max_source_positions": c.max_source_positions,
        "max_target_positions": c.max_target_positions,
        **{f"params_{k}": v for k, v in count_parameters(model).items()},
    }
