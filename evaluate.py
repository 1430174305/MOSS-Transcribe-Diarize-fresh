#!/usr/bin/env python3
"""Offline generation-based evaluator for fine-tuned checkpoints.

Run after (or during) training to score a checkpoint with the same metrics used
in the README: CER, cpCER, Δcp, timestamp MAE, and approximate DER.

Example:
    python evaluate.py --model outputs/finetuned --eval_jsonl data/eval.jsonl \\
        --max_new_tokens 65536 --output runs/eval.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, HfArgumentParser

from moss_transcribe_diarize.training import evaluate, load_eval_dataset


@dataclass
class EvalArguments:
    model: str = field(metadata={"help": "Path or id of the fine-tuned model/processor."})
    eval_jsonl: str = field(metadata={"help": "Eval manifest."})
    prompt_pool_file: str | None = field(default=None)
    max_length: int = 131072
    max_new_tokens: int = 65536
    output: str | None = field(default=None, metadata={"help": "Write detailed JSON here."})
    dtype: str = "auto"


def load_model(model_path: str, dtype: torch.dtype):
    """Load a checkpoint, auto-detecting a LoRA adapter directory."""
    base_dir = Path(model_path)
    adapter_cfg = base_dir / "adapter_config.json"
    if adapter_cfg.exists():
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        config = json.loads(adapter_cfg.read_text(encoding="utf-8"))
        base_name = config.get("base_model_name_or_path") or "OpenMOSS-Team/MOSS-Transcribe-Diarize"
        base = AutoModelForCausalLM.from_pretrained(base_name, trust_remote_code=True, dtype=dtype)
        return PeftModel.from_pretrained(base, model_path)
    return AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, dtype=dtype)


def main() -> None:
    parser = HfArgumentParser(EvalArguments)
    args = parser.parse_args_into_dataclass()[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dtype == "auto":
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
    else:
        dtype = getattr(torch, args.dtype)

    model = load_model(args.model, dtype).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    dataset = load_eval_dataset(args.eval_jsonl, args.prompt_pool_file)
    summary = evaluate(
        model,
        processor,
        dataset,
        device=torch.device(device),
        dtype=dtype,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        output_path=args.output,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
