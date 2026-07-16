"""Training utilities for MOSS-Transcribe-Diarize.

This subpackage extends the minimal `finetune.py` workflow with:

- speaker-aware data loading and length-bucketed batching
- optional meeting-domain audio augmentation (timing-preserving)
- token-weighted causal LM loss (timestamps / speaker tags upweighted)
- generation-based evaluation (CER / cpCER / DER / timestamp MAE)
"""

from .data import ConversationDataset, DataCollator, MeetingAugmentor, ScriptArguments, build_prompt_pool
from .evaluation import evaluate, load_eval_dataset
from .loss import WeightedTrainer
from .metrics import compute_cer, compute_cpcer, compute_timestamp_mae, compute_der, summarize

__all__ = [
    "ConversationDataset",
    "DataCollator",
    "MeetingAugmentor",
    "ScriptArguments",
    "build_prompt_pool",
    "evaluate",
    "load_eval_dataset",
    "WeightedTrainer",
    "compute_cer",
    "compute_cpcer",
    "compute_timestamp_mae",
    "compute_der",
    "summarize",
]
