"""Inference / decoding module (1C4).

``Transcriber`` turns audio into text with any Whisper checkpoint (ours, the authors', or
OpenAI's):

* batch transcription of files or arrays; audio is converted to 16 kHz mono first;
* greedy (``num_beams=1``) or beam search, with an n-best list of alternatives;
* a confidence score per hypothesis: exp(mean token log-probability), obtained by
  re-scoring the decoded tokens with one teacher-forced pass (version-independent and
  exact, unlike reading scores back out of ``generate``);
* optional n-gram LM rescoring of the n-best list;
* audio longer than 30 s is split at low-energy points into <= 30 s chunks.

The demo UI (1C6) only needs to wrap :meth:`Transcriber.transcribe`.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from arabic_asr.data.audio import TARGET_SR, load_audio, prepare_waveform
from arabic_asr.data.features import LogMelExtractor
from arabic_asr.data.prepare import load_tokenizer
from arabic_asr.inference.rescoring import rescore
from arabic_asr.model.hub import load_feature_extractor
from arabic_asr.model.whisper_model import configure_generation, special_token_ids
from arabic_asr.utils import autocast_context, get_device, resolve_precision

AudioInput = Union[str, Path, np.ndarray]


@dataclass
class Hypothesis:
    text: str
    am_logprob: float = 0.0          # sum of token log-probs (text tokens + end-of-text)
    n_tokens: int = 0
    lm_logprob: Optional[float] = None
    score: Optional[float] = None
    finished: bool = True             # False if generation hit max_new_tokens

    @property
    def avg_logprob(self) -> float:
        return self.am_logprob / max(1, self.n_tokens)

    @property
    def confidence(self) -> float:
        return float(math.exp(self.avg_logprob))

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(avg_logprob=self.avg_logprob, confidence=self.confidence)
        return d


@dataclass
class ChunkResult:
    start_s: float
    end_s: float
    hypotheses: List[Hypothesis]


@dataclass
class Transcription:
    text: str
    confidence: float
    duration_s: float
    processing_s: float
    chunks: List[ChunkResult] = field(default_factory=list)

    @property
    def alternatives(self) -> List[str]:
        """Distinct alternative transcripts (n-best) when the audio fits in one chunk.

        Different beams can decode to the same text (different token splits), so duplicates
        are removed, keeping the best-scored order."""
        if len(self.chunks) == 1:
            return list(dict.fromkeys(h.text for h in self.chunks[0].hypotheses))
        return [self.text]

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "duration_s": self.duration_s,
            "processing_s": self.processing_s,
            "real_time_factor": self.processing_s / max(1e-6, self.duration_s),
            "alternatives": self.alternatives,
            "chunks": [
                {"start_s": c.start_s, "end_s": c.end_s, "hypotheses": [h.to_dict() for h in c.hypotheses]}
                for c in self.chunks
            ],
        }


def split_long_audio(audio: np.ndarray, sr: int = TARGET_SR, max_s: float = 30.0,
                     search_s: float = 5.0) -> List[Tuple[int, int]]:
    """Cut audio into <= ``max_s`` chunks, choosing each cut at the quietest 25 ms frame in
    the last ``search_s`` seconds before the limit (so words are rarely split)."""
    max_len = int(max_s * sr)
    if len(audio) <= max_len:
        return [(0, len(audio))]
    frame, hop = int(0.025 * sr), int(0.010 * sr)
    bounds, start = [], 0
    while len(audio) - start > max_len:
        lo = start + int((max_s - search_s) * sr)
        window = audio[lo : start + max_len]
        n_frames = max(1, (len(window) - frame) // hop + 1)
        energy = np.array([np.mean(window[i * hop : i * hop + frame] ** 2) for i in range(n_frames)])
        cut = lo + int(np.argmin(energy)) * hop + frame // 2
        bounds.append((start, cut))
        start = cut
    bounds.append((start, len(audio)))
    return bounds


class Transcriber:
    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        precision: str = "auto",
        tokenizer_name: Optional[str] = None,
        language: str = "arabic",
        task: str = "transcribe",
        num_beams: int = 1,
        n_best: int = 1,
        max_new_tokens: int = 225,
        batch_size: int = 8,
        compute_scores: bool = True,
        lm=None,
        lm_weight: float = 0.0,
        word_bonus: float = 0.0,
        base_model: str = "openai/whisper-small",
    ):
        self.device = get_device(device)
        self.precision = resolve_precision(precision, self.device)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path)
        configure_generation(self.model, language, task, base_model)
        self.model.config.use_cache = True
        self.model.to(self.device).eval()
        tok_source = tokenizer_name or self._tokenizer_source(model_name_or_path, base_model)
        self.tokenizer = load_tokenizer(tok_source, language, task)
        fe = load_feature_extractor(model_name_or_path, fallback=base_model)
        self.extractor = LogMelExtractor(mel_filters=fe.mel_filters)
        if self.extractor.n_mels != self.model.config.num_mel_bins:
            raise ValueError("Feature extractor and model disagree on the number of mel bins")
        ids = special_token_ids(self.tokenizer)
        self.eot_id = ids["eot"]
        self.prompt = [ids["sot"], ids["ar"], ids["transcribe"], ids["notimestamps"]]
        self.language, self.task = language, task
        self.num_beams = max(1, num_beams)
        self.n_best = max(1, min(n_best, self.num_beams))
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.compute_scores = compute_scores or lm is not None
        self.lm, self.lm_weight, self.word_bonus = lm, lm_weight, word_bonus
        self.model_name = str(model_name_or_path)

    @staticmethod
    def _tokenizer_source(path: str, fallback: str) -> str:
        local = Path(path)
        if local.exists():
            return path if (local / "tokenizer_config.json").exists() else fallback
        return fallback if not path.startswith("openai/") else path

    # ------------------------------------------------------------------ decoding
    def _content(self, sequence: Sequence[int]) -> Tuple[List[int], bool]:
        """Text tokens after the prompt, cut at the first end-of-text."""
        content, finished = [], False
        for tok in sequence:
            if tok == self.eot_id:  # end of text (later positions are padding)
                finished = True
                break
            if tok > self.eot_id:  # <|startoftranscript|>, language, task, timestamps, ...
                continue
            content.append(int(tok))
        return content, finished

    @torch.inference_mode()
    def _score(self, encoder_states: torch.Tensor, contents: List[List[int]], per_input: int,
               chunk: int = 8) -> List[Tuple[float, int]]:
        """Teacher-forced log-probability of each hypothesis (text tokens + end-of-text)."""
        results: List[Tuple[float, int]] = []
        p = len(self.prompt)
        for start in range(0, len(contents), chunk):
            block = contents[start : start + chunk]
            enc = encoder_states.repeat_interleave(per_input, dim=0)[start : start + chunk]
            max_len = max(p + len(c) for c in block)
            inputs = torch.full((len(block), max_len), self.eot_id, dtype=torch.long)
            targets = torch.full((len(block), max_len), -100, dtype=torch.long)
            for i, c in enumerate(block):
                seq = self.prompt + c
                inputs[i, : len(seq)] = torch.tensor(seq)
                tgt = c + [self.eot_id]
                targets[i, p - 1 : p - 1 + len(tgt)] = torch.tensor(tgt)
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            with autocast_context(self.precision, self.device):
                logits = self.model(encoder_outputs=BaseModelOutput(last_hidden_state=enc),
                                    decoder_input_ids=inputs).logits
            logp = torch.log_softmax(logits.float(), dim=-1)
            mask = targets != -100
            picked = logp.gather(-1, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1) * mask
            for i in range(len(block)):
                results.append((float(picked[i].sum().item()), int(mask[i].sum().item())))
        return results

    @torch.inference_mode()
    def decode_features(self, features: torch.Tensor) -> List[List[Hypothesis]]:
        """features: (batch, n_mels, 3000) -> n-best hypotheses per input (best first)."""
        feats = features.to(self.device)
        per_input = self.n_best
        with autocast_context(self.precision, self.device):
            generated = self.model.generate(
                input_features=feats,
                language=self.language,
                task=self.task,
                num_beams=self.num_beams,
                num_return_sequences=per_input,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )
        sequences = generated.sequences if hasattr(generated, "sequences") else generated
        parsed = [self._content(seq.tolist()) for seq in sequences.cpu()]
        texts = self.tokenizer.batch_decode([c for c, _ in parsed], skip_special_tokens=True)
        scores = [(0.0, 0)] * len(parsed)
        if self.compute_scores:
            with autocast_context(self.precision, self.device):
                encoder_states = self.model.model.encoder(feats).last_hidden_state
            scores = self._score(encoder_states, [c for c, _ in parsed], per_input)
        out: List[List[Hypothesis]] = []
        for b in range(feats.shape[0]):
            hyps = []
            for k in range(per_input):
                i = b * per_input + k
                hyps.append(Hypothesis(text=texts[i].strip(), am_logprob=scores[i][0], n_tokens=scores[i][1],
                                       finished=parsed[i][1]))
            if self.lm is not None:
                hyps = rescore(hyps, self.lm, self.lm_weight, self.word_bonus)
            out.append(hyps)
        return out

    # ------------------------------------------------------------------ public API
    def _waveform(self, audio: AudioInput, sample_rate: Optional[int]) -> np.ndarray:
        if isinstance(audio, (str, Path)):
            return load_audio(audio)
        if sample_rate is None:
            raise ValueError("sample_rate is required when passing a raw array")
        return prepare_waveform(np.asarray(audio), sample_rate)

    def transcribe(self, audio: AudioInput, sample_rate: Optional[int] = None) -> Transcription:
        start = time.time()
        wav = self._waveform(audio, sample_rate)
        bounds = split_long_audio(wav)
        feats = torch.stack([self.extractor(wav[s:e])[0] for s, e in bounds])
        nbest: List[List[Hypothesis]] = []
        for i in range(0, len(bounds), self.batch_size):
            nbest.extend(self.decode_features(feats[i : i + self.batch_size]))
        chunks = [ChunkResult(s / TARGET_SR, e / TARGET_SR, hyps) for (s, e), hyps in zip(bounds, nbest)]
        text = " ".join(c.hypotheses[0].text for c in chunks if c.hypotheses[0].text).strip()
        total_lp = sum(c.hypotheses[0].am_logprob for c in chunks)
        total_tok = sum(c.hypotheses[0].n_tokens for c in chunks)
        confidence = math.exp(total_lp / total_tok) if total_tok else float("nan")
        return Transcription(text=text, confidence=confidence, duration_s=len(wav) / TARGET_SR,
                             processing_s=time.time() - start, chunks=chunks)

    def transcribe_many(self, inputs: Sequence[AudioInput]) -> List[Transcription]:
        return [self.transcribe(x) for x in inputs]
