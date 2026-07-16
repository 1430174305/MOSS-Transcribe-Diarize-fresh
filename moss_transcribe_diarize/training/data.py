"""Dataset, collator, and meeting-domain augmentation for fine-tuning."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import soxr
import torch
from torch.utils.data import Dataset

from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT, build_transcription_messages
from moss_transcribe_diarize.processing_moss_transcribe_diarize import (
    MossTranscribeDiarizeProcessor,
)


@dataclass
class ScriptArguments:
    train_jsonl: str = field(metadata={"help": "Conversation-format training manifest."})
    eval_jsonl: Optional[str] = field(default=None, metadata={"help": "Optional eval manifest."})
    model_name_or_path: str = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
    max_length: int = 131072
    attn_implementation: str = "flash_attention_2"
    # multi-prompt mixing
    prompt_pool_file: Optional[str] = field(default=None, metadata={"help": "One prompt per line."})
    prompt_pool_prob: float = 0.5
    # token-weighted loss
    loss_weight_digits: float = 3.0
    loss_weight_struct: float = 2.0
    loss_weight_base: float = 1.0
    # memory: chunk the cross-entropy over time to avoid materializing [T, V] softmax
    loss_chunk_size: int = 8192
    # parameter strategy
    freeze_whisper_encoder: bool = field(default=True, metadata={"help": "Freeze the Whisper audio encoder."})
    unfreeze_encoder_layers: int = field(default=0, metadata={"help": "If >0, unfreeze the last N encoder layers (0 = fully frozen)."})
    train_vq_adaptor: bool = field(default=True, metadata={"help": "Keep the VQAdaptor trainable."})
    use_lora: bool = field(default=False, metadata={"help": "Apply LoRA to the language model."})
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,up_proj,gate_proj,down_proj"
    # meeting-domain augmentation
    noise_dir: Optional[str] = field(default=None, metadata={"help": "Dir of background noise wavs."})
    rir_dir: Optional[str] = field(default=None, metadata={"help": "Dir of room impulse response wavs."})
    aug_noise_prob: float = 0.3
    aug_noise_snr_db: float = 10.0
    aug_rir_prob: float = 0.3
    aug_gain_prob: float = 0.5
    aug_gain_db: float = 6.0
    seed: int = 42


def build_prompt_pool(path: Optional[str]) -> list[str]:
    if not path:
        return []
    prompts = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


class ConversationDataset(Dataset):
    """Parse conversation JSONL into {audio, prompt, target, length} samples.

    Each line: ``{"conversation": [user/text, user/audio, assistant/text], "length"?: seconds}``.
    An optional top-level ``prompt`` overrides the prompt; otherwise the dataset
    draws from the prompt pool (or ``DEFAULT_PROMPT``).
    """

    def __init__(
        self,
        path: str,
        prompt_pool: Optional[list[str]] = None,
        prompt_pool_prob: float = 0.5,
        seed: int = 42,
    ):
        manifest = Path(path).expanduser().resolve()
        if not manifest.exists():
            raise FileNotFoundError(manifest)
        self.prompt_pool = prompt_pool or []
        self.prompt_pool_prob = prompt_pool_prob
        self._rng = np.random.default_rng(seed)
        self.samples: list[dict] = []
        with manifest.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                self.samples.append(self._parse(json.loads(line), manifest.parent, line_no))
        if not self.samples:
            raise ValueError(f"No samples found in {manifest}")
        self.lengths: Optional[list[int]] = self._collect_lengths()

    @staticmethod
    def _parse(row: dict, root: Path, line_no: int) -> dict:
        conversation = row.get("conversation") or []
        expected = [("user", "text"), ("user", "audio"), ("assistant", "text")]
        if not isinstance(conversation, list) or not all(isinstance(item, dict) for item in conversation):
            raise ValueError(f"Line {line_no}: conversation must be a list of messages")
        actual = [(item.get("role"), item.get("message_type")) for item in conversation]
        if actual != expected:
            raise ValueError(f"Line {line_no}: expected user/text, user/audio, assistant/text")

        prompt, audio_path, target = (item.get("content") for item in conversation)
        if not all(isinstance(value, str) and value.strip() for value in (prompt, audio_path, target)):
            raise ValueError(f"Line {line_no}: prompt, audio path, and target must be non-empty strings")

        audio = Path(audio_path).expanduser()
        if not audio.is_absolute():
            audio = (root / audio).resolve()
        return {
            "audio": str(audio),
            "prompt": prompt.strip(),
            "target": target.strip(),
            "length": float(row["length"]) if "length" in row else None,
        }

    def _collect_lengths(self) -> Optional[list[int]]:
        if any(s["length"] is None for s in self.samples):
            return None
        # audio token count ~= seconds * 12.5 (after 4x merge); used only for grouping.
        return [max(1, int(s["length"] * 12.5)) + len(s["target"]) for s in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        # re-sample prompt per access so multi-prompt mixing actually varies across epochs
        if self.prompt_pool and float(self._rng.random()) < self.prompt_pool_prob:
            return {**sample, "prompt": str(self._rng.choice(self.prompt_pool))}
        return sample


class MeetingAugmentor:
    """Timing-preserving augmentations for meeting audio.

    Reverberation and additive noise do not shift word timing, so existing
    timestamp labels stay valid. Speed perturbation is intentionally avoided.
    """

    def __init__(
        self,
        noise_dir: Optional[str] = None,
        rir_dir: Optional[str] = None,
        noise_prob: float = 0.3,
        noise_snr_db: float = 10.0,
        rir_prob: float = 0.3,
        gain_prob: float = 0.5,
        gain_db: float = 6.0,
        seed: int = 42,
    ):
        self.noise_files = self._list_audio(noise_dir)
        self.rir_files = self._list_audio(rir_dir)
        self.noise_prob = noise_prob
        self.noise_snr_db = noise_snr_db
        self.rir_prob = rir_prob
        self.gain_prob = gain_prob
        self.gain_db = gain_db
        self._rng = np.random.default_rng(seed)

    @staticmethod
    def _list_audio(directory: Optional[str]) -> list[Path]:
        if not directory:
            return []
        root = Path(directory).expanduser().resolve()
        if not root.exists():
            return []
        exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
        return [p for p in sorted(root.rglob("*")) if p.suffix.lower() in exts]

    def __call__(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        audio = audio.copy()
        if self.rir_files and self._rng.random() < self.rir_prob:
            audio = self._apply_rir(audio, sample_rate)
        if self.noise_files and self._rng.random() < self.noise_prob:
            audio = self._add_noise(audio, sample_rate)
        if self._rng.random() < self.gain_prob:
            audio = self._apply_gain(audio)
        # guard against clipping
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.0:
            audio = audio / peak
        return audio.astype(np.float32, copy=False)

    def _apply_rir(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        try:
            import scipy.signal as signal
        except ImportError:
            return audio
        rir_path = str(self.rir_files[int(self._rng.integers(0, len(self.rir_files)))])
        rir, sr = sf.read(rir_path, dtype="float32", always_2d=True)
        rir = rir.mean(axis=1)
        if sr != sample_rate:
            rir = soxr.resample(rir, sr, sample_rate)
        rir = rir / (np.max(np.abs(rir)) + 1e-9)
        if rir.size > audio.size:
            rir = rir[: audio.size]
        reverbed = signal.fftconvolve(audio, rir, mode="full")[: audio.size]
        # mix dry/wet to keep presence
        return 0.7 * audio + 0.3 * reverbed

    def _add_noise(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        noise_path = str(self.noise_files[int(self._rng.integers(0, len(self.noise_files)))])
        noise, sr = sf.read(noise_path, dtype="float32", always_2d=True)
        noise = noise.mean(axis=1)
        if sr != sample_rate:
            noise = soxr.resample(noise, sr, sample_rate)
        if noise.size < audio.size:
            reps = int(math.ceil(audio.size / noise.size))
            noise = np.tile(noise, reps)
        noise = noise[: audio.size]
        noise = noise - float(np.mean(noise))
        sig_power = float(np.mean(audio ** 2)) + 1e-9
        noise_power = float(np.mean(noise ** 2)) + 1e-9
        scale = math.sqrt(sig_power / (noise_power * 10 ** (self.noise_snr_db / 10.0)))
        return audio + scale * noise

    def _apply_gain(self, audio: np.ndarray) -> np.ndarray:
        gain_db = float(self._rng.uniform(-self.gain_db, self.gain_db))
        return audio * (10.0 ** (gain_db / 20.0))


class DataCollator:
    """Build model inputs, labels, and per-token loss weights.

    Keeps the label-masking strategy from the original minimal collator: the
    prompt span (chat template + audio placeholders) is masked with ``-100``,
    the assistant target tokens are supervised.
    """

    def __init__(self, processor: MossTranscribeDiarizeProcessor, max_length: int, args: ScriptArguments):
        self.processor = processor
        self.max_length = max_length
        self.sample_rate = int(processor.feature_extractor.sampling_rate)
        self.args = args
        self.augmentor = MeetingAugmentor(
            noise_dir=args.noise_dir,
            rir_dir=args.rir_dir,
            noise_prob=args.aug_noise_prob,
            noise_snr_db=args.aug_noise_snr_db,
            rir_prob=args.aug_rir_prob,
            gain_prob=args.aug_gain_prob,
            gain_db=args.aug_gain_db,
            seed=args.seed,
        )
        self.use_weights = args.loss_weight_digits != 1.0 or args.loss_weight_struct != 1.0 or args.loss_weight_base != 1.0
        self.augment = True
        self._struct_ids = self._build_struct_token_ids()

    def _build_struct_token_ids(self) -> torch.Tensor:
        tok = self.processor.tokenizer
        ids: set[int] = set()
        # digit token ids are guaranteed single-token by the processor
        ids.update(int(v) for v in self.processor.digit_token_ids.values())
        for ch in (".", "[", "]", "S"):
            enc = tok.encode(ch, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(int(enc[0]))
        # also catch multi-token decimal forms like "0." is not needed; digits covered.
        return torch.tensor(sorted(ids), dtype=torch.long) if ids else torch.empty(0, dtype=torch.long)

    def _load_audio(self, path: str) -> np.ndarray:
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if sample_rate != self.sample_rate:
            audio = soxr.resample(audio, sample_rate, self.sample_rate)
        if not self.augment:
            return audio
        return self.augmentor(audio, self.sample_rate)

    def __call__(self, samples: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        prompts, texts, audios = [], [], []
        for sample in samples:
            prompt = self.processor.apply_chat_template(
                build_transcription_messages(sample["audio"], sample["prompt"]),
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)
            texts.append(prompt + sample["target"] + self.processor.tokenizer.eos_token)
            audios.append(self._load_audio(sample["audio"]))

        batch = self.processor(
            text=texts,
            audio=audios,
            max_length=self.max_length,
            return_tensors="pt",
        )
        device = batch["input_ids"].device

        audio_lengths = torch.zeros(len(samples), dtype=torch.long, device=device)
        audio_lengths.scatter_add_(
            0,
            batch["audio_chunk_mapping"].to(device),
            batch["audio_feature_lengths"].to(device),
        )

        labels = batch["input_ids"].clone()
        weights = torch.zeros_like(labels, dtype=torch.float)
        struct_ids = self._struct_ids.to(device)
        base = self.args.loss_weight_base

        for index, (prompt, audio_length) in enumerate(zip(prompts, audio_lengths.tolist())):
            prompt_ids = self.processor.expand_audio_token(
                prompt,
                audio_length,
                self.max_length,
            )
            plen = len(prompt_ids)
            labels[index, :plen] = -100
            if self.use_weights:
                target_ids = batch["input_ids"][index, plen:]
                w = torch.full_like(target_ids, base, dtype=torch.float, device=device)
                if struct_ids.numel() > 0:
                    mask = torch.isin(target_ids, struct_ids)
                    # distinguish digits (in digit set) vs other struct tokens
                    digit_ids = torch.tensor(
                        [int(v) for v in self.processor.digit_token_ids.values()],
                        dtype=torch.long,
                        device=device,
                    )
                    w[mask] = self.args.loss_weight_struct
                    if digit_ids.numel() > 0:
                        w[torch.isin(target_ids, digit_ids)] = self.args.loss_weight_digits
                weights[index, plen:] = w

        pad_mask = batch["attention_mask"] == 0
        labels[pad_mask] = -100
        weights[pad_mask] = 0.0
        batch["labels"] = labels
        if self.use_weights:
            batch["loss_weights"] = weights
        return dict(batch)
