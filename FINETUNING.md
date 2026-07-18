# Fine-tuning

This repo ships an enhanced fine-tuning stack on top of the model implementation:

- `finetune.py` — training entrypoint (selective freezing, LoRA, FlashAttention-2, token-weighted loss, meeting-domain augmentation, length bucketing, eval)
- `prepare_data.py` — normalize raw transcripts into the conversation manifest (speaker renumbering, timestamp validation, duration tagging, train/eval split)
- `evaluate.py` — offline generation-based scoring (CER / cpCER / Δcp / timestamp MAE / DER)
- `moss_transcribe_diarize/training/` — dataset, collator, augmentor, weighted Trainer, metrics

## Installation

Follow the environment setup in the main README, then install the training extras:

```bash
uv pip install -e ".[torch-runtime,train,flash-attn]" --torch-backend=auto
```

## Data format

Training data is a JSONL with one conversation per line (text prompt, audio path, reference transcript, in that order):

```json
{"conversation":[{"role":"user","message_type":"text","content":"Transcribe the audio with timestamps and speaker labels."},{"role":"user","message_type":"audio","content":"audio/example.wav"},{"role":"assistant","message_type":"text","content":"[0.00][S01]Welcome[0.72]"}], "length": 12.34}
```

The optional `length` field (audio seconds) enables length-bucketed sampling to reduce padding. Build it from raw transcripts with `prepare_data.py`, which also renumbers speakers by first appearance (`[S01]`, `[S02]`, ...) and validates timestamps:

```bash
python prepare_data.py \
  --input_jsonl raw.jsonl \
  --output_dir data \
  --train_split 0.95
```

`raw.jsonl` is one record per line: `{"audio": "path.wav", "transcript": "[0.5][S03]hi[1.2]...", "prompt"?: "..."}`.

## Training (recommended meeting-domain recipe)

Single GPU:

```bash
python finetune.py \
  --train_jsonl data/train.jsonl \
  --eval_jsonl  data/eval.jsonl \
  --output_dir outputs/finetuned \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --num_train_epochs 4 \
  --learning_rate 2e-5 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --weight_decay 0.1 \
  --adam_beta1 0.9 --adam_beta2 0.95 \
  --label_smoothing_factor 0.05 \
  --bf16 --gradient_checkpointing \
  --attn_implementation flash_attention_2 \
  --max_length 65536 \
  --freeze_whisper_encoder true \
  --use_lora true --lora_r 64 --lora_alpha 128 \
  --noise_dir data/noise --rir_dir data/rir \
  --prompt_pool_file examples/prompts.txt \
  --loss_weight_digits 3.0 --loss_weight_struct 2.0 \
  --eval_strategy epoch \
  --save_total_limit 3 \
  --load_best_model_at_end \
  --metric_for_best_model eval_loss --greater_is_better false
```

Multi-GPU (DDP) — preferred for full-parameter fine-tuning:

```bash
torchrun --nproc_per_node=8 finetune.py \
  --train_jsonl data/train.jsonl --eval_jsonl data/eval.jsonl \
  --output_dir outputs/finetuned \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
  --num_train_epochs 4 --learning_rate 2e-5 --bf16 --gradient_checkpointing \
  --attn_implementation flash_attention_2 --max_length 65536 \
  --freeze_whisper_encoder true --use_lora false
```

For memory-tight long-sequence runs, add FSDP: `--fsdp "full_shard auto_cpu_offload" --fsdp_config configs/fsdp.json`. The default DDP path is simpler and sufficient for 0.9B on 80G cards.

## Parameter strategy

**Recommended: single-stage, freeze encoder, LoRA + VQAdaptor trainable.** The encoder and VQAdaptor are already aligned with the LM in the pretrained checkpoint, and the model is already SOTA on meeting benchmarks (AISHELL-4, Alimeeting), so there is no need to stage "adaptor-warmup" first.

- `--freeze_whisper_encoder true` (default): the Whisper encoder is already strong; freezing it avoids catastrophic forgetting and saves memory/compute.
- `--unfreeze_encoder_layers N` (default 0): escalation lever only. When the frozen-encoder run plateaus *and* your acoustics differ from Whisper's training distribution (far-field mics, heavy reverb, strong overlap, non-English-heavy), unfreeze the **last N** encoder layers (e.g. 4–8) — not the whole encoder. This survives LoRA's base freeze automatically.
- `--use_lora true` (recommended): LoRA on `q/k/v/o/up/gate/down_proj` of the Qwen3 LM; the small `VQAdaptor` bridge is kept fully trainable. Use this first; only switch to `--use_lora false` (selective full fine-tuning, low LR `5e-6`) if LoRA plateaus.
- `--train_vq_adaptor true` (default): the audio-text bridge is small but critical and should always be trainable.

## Token-weighted loss

Timestamp digits and `[Sxx]` speaker tags are structural and error-prone. `--loss_weight_digits 3.0` upweights digit tokens, `--loss_weight_struct 2.0` upweights brackets/`S`, keeping plain text at `1.0`. This sharpens boundary and speaker accuracy without changing the decoder architecture.

## Offline evaluation

The in-loop metric (`token_accuracy`) is a cheap teacher-forced proxy. For real quality, run the generation-based evaluator on the eval set (large `max_new_tokens`, greedy, deterministic):

```bash
python evaluate.py \
  --model outputs/finetuned \
  --eval_jsonl data/eval.jsonl \
  --max_new_tokens 65536 \
  --output runs/eval.json
```

It reports CER, cpCER, Δcp, timestamp MAE (seconds), and approximate DER, mirroring the README's evaluation scale.

## Prompt pool (multi-task mixing)

Provide one prompt per line via `--prompt_pool_file` (e.g. concatenate `examples/prompts.md` snippets into `prompts.txt`). Each sample draws from the pool with probability `--prompt_pool_prob` to improve instruction following across transcribe/diarize, hotword, and event-awareness tasks.

## Quick smoke test (does fine-tuning help at all?)

Before committing to a long run, prove the direction on a tiny subset. The base model is already SOTA, so it is easy to make it *worse* — always compare against a baseline.

1. **Baseline** the pretrained model on your eval split first (no fine-tuning):

   ```bash
   python evaluate.py --model OpenMOSS-Team/MOSS-Transcribe-Diarize \
     --eval_jsonl data/eval.jsonl --max_new_tokens 65536 --output runs/baseline.json
   ```

2. **Smoke train** on ~50–200 short samples, single GPU, a few minutes:

   ```bash
   python finetune.py \
     --train_jsonl data/train.small.jsonl --eval_jsonl data/eval.jsonl \
     --output_dir outputs/smoke --max_length 8192 \
     --freeze_whisper_encoder true --use_lora true --lora_r 16 --lora_alpha 32 \
     --learning_rate 1e-4 --num_train_epochs 1 --bf16 --gradient_checkpointing \
     --attn_implementation flash_attention_2 --eval_strategy epoch
   ```

3. **Evaluate** the smoke checkpoint the same way as the baseline:

   ```bash
   python evaluate.py --model outputs/smoke --eval_jsonl data/eval.jsonl \
     --max_new_tokens 65536 --output runs/smoke.json
   ```

4. **Decision**: if `cpCER` and `CER` are both down vs baseline, scale up (more data, `max_length` to the 95th percentile, `lora_r 64`, lower LR to `2e-5`, 3–4 epochs). If worse, the usual culprits are: speaker labels not renumbered to first-appearance order (run `prepare_data.py`), timestamp granularity mismatch between your labels and model output, or LR too high / too few samples (overfit). Do not ship a checkpoint that regresses the base.

## Memory and long sequences

The model itself supports ~90 min of audio at the default `max_length=131072` (131072 / 12.5 audio tokens/s). This is an **inference-time** capability: `generate` is incremental and never materializes the full logits tensor. Training is different — the forward pass must compute a `[B, T, V]` logits tensor (V=151936) for the cross-entropy, which is the real memory cost of long-audio training:

- `max_length=65536` → ~20 GB logits (bf16) + ~20 GB grad ≈ 40 GB.
- `max_length=131072` (~90 min) → ~40 GB logits + ~40 GB grad ≈ 80 GB → OOM on a single 80 GB card even with LoRA, and DDP does not help (each rank replicates the full logits).

Mitigations, in order of effectiveness for full-length training:

1. **Windowed loss (recommended, now built-in)**: `--loss_window 8192` (or 16384). The decoder still runs the **full** long forward (long-range attention is exercised, so long-context diarization is actually trained), but the `lm_head` + CE only operate on a random sub-window of W tokens per step, so only `[W, V]` logits are materialized instead of `[T, V]`. Since `hidden_states` is 1024-dim (90 min ≈ 0.27 GB), a single 80 GB card fits 90-min+ audio easily. Different windows across steps cover the whole sequence. This is the single-card fix for the 90-min OOM — no multi-GPU needed for the duration problem.
2. **Liger fused linear-CE** (`pip install liger-kernel`, `--use_liger_kernel true`): fuses `lm_head` + CE so the `[T, V]` logits are never materialized. Alternative to windowed loss but needs wiring into the custom model's forward and is incompatible with per-token weighting.
3. **Shorter training clips**: train on ≤ `max_length` windows (e.g. 32768, ~26 min) via `prepare_data.py` segmentation. Domain adaptation transfers from shorter clips; the model keeps its 90-min inference ability. Use this if you do not specifically need long-context adaptation.
4. The cross-entropy is already chunked over time (`--loss_chunk_size 8192`), which kills the ~40 GB softmax buffer but not the logits tensor itself (windowed loss above handles that).
5. `--gradient_checkpointing` trades ~25% speed for activation savings.
6. **FSDP with vocab sharding**: shards `lm_head`/embeddings so each rank computes `[T, V/n]` logits — scales duration ~linearly with cards. High-risk for this custom tied-weight model; prefer windowed loss on a single card.

**Multi-GPU note**: plain DDP (`torchrun`) does **not** extend single-sample duration — each rank replicates the full `[T, V]` logits. DDP only adds throughput (parallel samples). To train a single 90-min+ audio, use `--loss_window` (single card) or FSDP with vocab sharding (multi-card).

For most meeting-domain adaptation, `--loss_window 8192` on a single 80 GB card is the simplest path to 90-min training. Use multi-GPU (DDP) to scale throughput once one sample fits.

## Notes

- `max_length` 65536 covers ~52 minutes of meeting audio in one pass (12.5 audio tokens/sec after 4x merge); offline you can afford full-length, unchunked training.
- The effective batch size is `nproc * per_device_train_batch_size * gradient_accumulation_steps`.
- Any standard `TrainingArguments` option can be passed on the command line.
- The final model, processor, and intermediate checkpoints are saved under `output_dir`.
