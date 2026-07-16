#!/usr/bin/env python3
"""Fine-tune MOSS-Transcribe-Diarize with quality-focused optimizations.

This is an enhanced replacement for the minimal `finetune.py` shipped in the
upstream repo. It adds, on top of the original workflow:

- selective parameter freezing (Whisper encoder) + optional LoRA on the LM
- FlashAttention-2 and gradient checkpointing for memory-efficient long audio
- token-weighted causal LM loss (timestamps / speaker tags upweighted)
- meeting-domain, timing-preserving audio augmentation (RIR / noise / gain)
- length-bucketed sampling to cut padding waste
- multi-prompt mixing for stronger instruction following
- teacher-forced eval during training + offline generation eval via evaluate.py

Multi-GPU: run with `torchrun --nproc_per_node=<N> finetune.py ...` (DDP).
FSDP is optional for memory-tight long-sequence runs.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    HfArgumentParser,
    TrainingArguments,
)

from moss_transcribe_diarize.processing_moss_transcribe_diarize import MossTranscribeDiarizeProcessor
from moss_transcribe_diarize.training import (
    ConversationDataset,
    DataCollator,
    ScriptArguments,
    WeightedTrainer,
    build_prompt_pool,
)
from moss_transcribe_diarize.training.metrics import compute_token_accuracy


def freeze_module(model, attribute_path: str, freeze: bool) -> None:
    """Toggle requires_grad on a submodule reached by dotted path."""
    obj = model
    for part in attribute_path.split("."):
        obj = getattr(obj, part)
    for param in obj.parameters():
        param.requires_grad_(freeze)


def apply_parameter_strategy(model, args: ScriptArguments) -> list:
    """Freeze encoder / VQAdaptor according to the selected strategy.

    Returns the list of encoder-tail params to re-enable *after* peft wrapping
    (so ``--unfreeze_encoder_layers`` survives LoRA's base freeze).
    """
    tail_params: list = []
    if args.freeze_whisper_encoder:
        freeze_module(model, "model.whisper_encoder", freeze=True)
        n = int(args.unfreeze_encoder_layers)
        if n > 0:
            encoder = model.model.whisper_encoder
            layers = getattr(encoder, "layers", None)
            if layers is not None:
                n = min(n, len(layers))
                tail_params = [p for layer in layers[-n:] for p in layer.parameters()]
    if not args.train_vq_adaptor and not args.use_lora:
        freeze_module(model, "model.vq_adaptor", freeze=True)
    return tail_params


def apply_lora(model, args: ScriptArguments):
    """Wrap the LM with LoRA adapters; keep VQAdaptor trainable if requested."""
    from peft import LoraConfig, get_peft_model

    target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # hold adaptor params so we can re-enable them after peft freezes the base
    adaptor_params = list(model.model.vq_adaptor.parameters())
    model = get_peft_model(model, config)
    if args.train_vq_adaptor:
        for param in adaptor_params:
            param.requires_grad_(True)
    model.print_trainable_parameters()
    return model


def preprocess_logits_for_metrics(logits, labels):
    """Argmax to int to avoid materializing the full [B, T, V] float tensor."""
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def main() -> None:
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()
    training_args.remove_unused_columns = False
    training_args.label_names = ["labels"]

    processor = MossTranscribeDiarizeProcessor.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=True,
    )
    prompt_pool = build_prompt_pool(script_args.prompt_pool_file)

    train_dataset = ConversationDataset(
        script_args.train_jsonl,
        prompt_pool=prompt_pool,
        prompt_pool_prob=script_args.prompt_pool_prob,
        seed=script_args.seed,
    )
    eval_dataset = None
    if script_args.eval_jsonl:
        eval_dataset = ConversationDataset(
            script_args.eval_jsonl,
            prompt_pool=prompt_pool,
            prompt_pool_prob=0.0,
            seed=script_args.seed,
        )

    collator = DataCollator(processor, script_args.max_length, script_args)

    # length bucketing only helps (and is safest) in single-process training
    if train_dataset.lengths is not None and training_args.world_size == 1:
        training_args.group_by_length = True

    dtype = (
        torch.bfloat16 if training_args.bf16 else torch.float16 if training_args.fp16 else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=True,
        dtype=dtype,
        attn_implementation=script_args.attn_implementation,
    )
    model.tie_weights()
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False

    encoder_tail_params = apply_parameter_strategy(model, script_args)
    if script_args.use_lora:
        model = apply_lora(model, script_args)
    # re-enable encoder tail after (optional) LoRA wrap, so it survives peft's base freeze
    for param in encoder_tail_params:
        param.requires_grad_(True)

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=processor,
        compute_metrics=compute_token_accuracy if eval_dataset is not None else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if eval_dataset is not None else None,
    )
    trainer.loss_chunk_size = script_args.loss_chunk_size

    result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
    trainer.save_state()
    trainer.save_metrics("train", result.metrics)


if __name__ == "__main__":
    main()
