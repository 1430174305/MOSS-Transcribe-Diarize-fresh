"""Weighted causal-LM Trainer with chunked cross-entropy and length-bucketed sampling.

The cross-entropy is chunked over the time dimension so the softmax buffer never
exceeds ``chunk_size * vocab`` elements. This is what makes long-audio training
(T up to 65536, V=151936) fit in memory: a full [T, V] softmax would be ~40 GB,
while chunking keeps the peak at a few GB. The logits tensor itself is still
produced by the forward pass, so set ``max_length`` to your data's 95th percentile
to bound that, or use Liger kernels if available.
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
    """Trainer applying per-token loss weights via a memory-chunked CE."""

    #: tokens per CE chunk; override on the instance to tune memory.
    loss_chunk_size: int = 8192

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        loss_weights = inputs.pop("loss_weights", None)
        outputs = model(**inputs)
        logits = outputs.logits

        # causal shift: predict token t+1 from position t
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        B, Tm1, V = shift_logits.shape
        flat_logits = shift_logits.view(B * Tm1, V)
        flat_labels = shift_labels.view(-1)
        valid = flat_labels != -100
        clamped = flat_labels.clamp(min=0)

        if loss_weights is not None:
            flat_weights = loss_weights[..., 1:].contiguous().view(-1).float()
        else:
            flat_weights = None

        chunk = max(1, int(self.loss_chunk_size))
        total_loss = logits.new_zeros(())
        total_weight = logits.new_zeros(())
        for start in range(0, B * Tm1, chunk):
            end = min(start + chunk, B * Tm1)
            seg_logits = flat_logits[start:end]
            seg_labels = clamped[start:end]
            seg_valid = valid[start:end]
            per_token = F.cross_entropy(
                seg_logits,
                seg_labels,
                reduction="none",
                ignore_index=-100,
            )
            if flat_weights is not None:
                seg_weights = flat_weights[start:end]
                mask = seg_valid.float()
                total_loss = total_loss + (per_token * seg_weights).sum()
                total_weight = total_weight + (seg_weights * mask).sum()
            else:
                mask = seg_valid.float()
                total_loss = total_loss + (per_token * mask).sum()
                total_weight = total_weight + mask.sum()

        loss = total_loss / total_weight.clamp_min(1.0)
        return (loss, outputs) if return_outputs else loss

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
