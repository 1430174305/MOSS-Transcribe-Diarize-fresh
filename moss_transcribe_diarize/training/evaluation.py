"""Generation-based evaluator: run the model on a manifest and score it."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from moss_transcribe_diarize.inference_utils import (
    generate_transcription,
    resolve_device,
)
from moss_transcribe_diarize.training import metrics
from moss_transcribe_diarize.training.data import ConversationDataset, build_prompt_pool


def load_eval_dataset(eval_jsonl: str, prompt_pool_file: str | None) -> ConversationDataset:
    pool = build_prompt_pool(prompt_pool_file)
    return ConversationDataset(eval_jsonl, prompt_pool=pool, prompt_pool_prob=0.0)


@torch.inference_mode()
def evaluate(
    model,
    processor,
    dataset: ConversationDataset,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    max_length: int = 131072,
    max_new_tokens: int = 65536,
    output_path: str | None = None,
) -> dict:
    """Run greedy generation on every eval sample and compute corpus metrics.

    Because evaluation is offline-only, we use a large ``max_new_tokens`` and
    greedy decoding for deterministic, high-quality outputs.
    """
    device = resolve_device("auto") if device is None else device
    if dtype is None:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    references: list[str] = []
    hypotheses: list[str] = []
    records: list[dict] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": sample["audio"]},
                    {"type": "text", "text": sample["prompt"].strip()},
                ],
            }
        ]
        result = generate_transcription(
            model,
            processor,
            messages,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            device=device,
            dtype=dtype,
        )
        hyp = result["text"]
        ref = sample["target"]
        references.append(ref)
        hypotheses.append(hyp)
        records.append({"audio": sample["audio"], "reference": ref, "hypothesis": hyp})

    summary = metrics.summarize(references, hypotheses)
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({"summary": summary, "predictions": records}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return summary
