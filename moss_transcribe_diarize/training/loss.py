"""Weighted causal-LM Trainer with chunked and windowed cross-entropy.

Two memory knobs:

- ``loss_chunk_size``: chunk the softmax over the time dim so the [T, V]
  softmax buffer stays small. The full [T, V] logits are still materialized.
- ``loss_window`` (>0): supervise a random sub-window of W tokens per step.
  The decoder still runs the FULL long forward (so long-range attention is
  exercised), but the lm_head only sees [W, V] logits — this is what lets a
  90-min+ single audio fit on one card. Different windows across steps cover
  the whole sequence.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import Trainer

try:
    from transformers.trainer_pt_utils import LengthGroupedSampler
except ImportError:  # pragma: no cover - older transformers
    LengthGroupedSampler = None


class WeightedTrainer(Trainer):
    """Trainer applying per-token loss weights via chunked/windowed CE."""

    loss_chunk_size: int = 8192
    loss_window: int = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._window_gen = torch.Generator(device="cpu")
        self._window_gen.manual_seed(int(self.args.seed))

    # ------------------------------------------------------------------ #
    def _inner_and_head(self, model):
        """Return (inner MossTranscribeDiarizeModel, lm_head) for the windowed path.

        Walks down ``.model`` until the module that owns ``language_model``
        (the inner model that returns last_hidden_state). Works for both the
        plain model and peft-wrapped models.
        """
        inner = getattr(model, "model", None)
        while inner is not None and not hasattr(inner, "language_model") and hasattr(inner, "model"):
            inner = getattr(inner, "model")
        lm_head = getattr(model, "lm_head", None)
        if lm_head is None and inner is not None and hasattr(inner, "lm_head"):
            lm_head = getattr(inner, "lm_head")
        if inner is None or not hasattr(inner, "language_model") or lm_head is None:
            raise RuntimeError("Windowed loss could not locate inner model / lm_head.")
        return inner, lm_head

    # ------------------------------------------------------------------ #
    def _chunked_ce(self, logits, labels, weights):
        """Weighted mean CE over already-shifted (logits, labels) [B, T, V]/[B, T]."""
        B, Tm1, V = logits.shape
        flat_logits = logits.reshape(B * Tm1, V)
        flat_labels = labels.reshape(-1)
        valid = flat_labels != -100
        clamped = flat_labels.clamp(min=0)
        flat_weights = weights.reshape(-1).float() if weights is not None else None

        chunk = max(1, int(self.loss_chunk_size))
        total_loss = logits.new_zeros(())
        total_weight = logits.new_zeros(())
        for start in range(0, B * Tm1, chunk):
            end = min(start + chunk, B * Tm1)
            per = F.cross_entropy(
                flat_logits[start:end],
                clamped[start:end],
                reduction="none",
                ignore_index=-100,
            )
            mask = valid[start:end].float()
            if flat_weights is not None:
                w = flat_weights[start:end]
                total_loss = total_loss + (per * w).sum()
                total_weight = total_weight + (w * mask).sum()
            else:
                total_loss = total_loss + (per * mask).sum()
                total_weight = total_weight + mask.sum()
        return total_loss / total_weight.clamp_min(1.0)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        loss_weights = inputs.pop("loss_weights", None)

        # Windowed path is training-only; eval (return_outputs=True) falls back
        # to the full path so in-loop metrics keep working (use --eval_strategy no
        # for very long sequences to avoid eval-time OOM).
        use_window = self.loss_window and not return_outputs and labels is not None
        if not use_window:
            outputs = model(**inputs)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = loss_weights[..., 1:].contiguous() if loss_weights is not None else None
            loss = self._chunked_ce(shift_logits, shift_labels, shift_weights)
            return (loss, outputs) if return_outputs else loss

        # ---- windowed path ----
        inner, lm_head = self._inner_and_head(model)
        outputs = inner(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            input_features=inputs.get("input_features"),
            audio_feature_lengths=inputs.get("audio_feature_lengths"),
            audio_chunk_mapping=inputs.get("audio_chunk_mapping"),
        )
        hidden = outputs.last_hidden_state  # [B, T, 1024] — tiny even at 90 min
        B, T, _ = hidden.shape
        if T < 2:
            loss = hidden.new_zeros(())
            return (loss, outputs) if return_outputs else loss

        W = min(int(self.loss_window), T - 1)
        if T - 1 <= W:
            a, b = 0, T - 1
        else:
            a = int(torch.randint(0, T - W, (1,), generator=self._window_gen).item())
            b = a + W

        # hidden[:, a:b] predicts labels[:, a+1:b+1] (causal offset baked in)
        hidden_w = hidden[:, a:b, :]
        labels_w = labels[:, a + 1 : b + 1]
        weights_w = loss_weights[:, a + 1 : b + 1] if loss_weights is not None else None
        logits_w = lm_head(hidden_w)  # [B, W, V] — only [W, V] materialized
        loss = self._chunked_ce(logits_w, labels_w, weights_w)
        return (loss, outputs) if return_outputs else loss

    # ------------------------------------------------------------------ #
    def evaluate(self, *args, **kwargs):
        """Disable audio augmentation while scoring, then restore it."""
        collator = self.data_collator
        previous = getattr(collator, "augment", True)
        if collator is not None:
            collator.augment = False
        try:
            return super().evaluate(*args, **kwargs)
        finally:
            if collator is not None:
                collator.augment = previous

    def _get_train_sampler(self):
        lengths = getattr(self.train_dataset, "lengths", None)
        if (
            LengthGroupedSampler is not None
            and self.args.group_by_length
            and lengths is not None
            and self.args.world_size == 1
        ):
            return LengthGroupedSampler(
                batch_size=self.args.train_batch_size,
                lengths=lengths,
                model_input_name=None,
            )
        return super()._get_train_sampler()
