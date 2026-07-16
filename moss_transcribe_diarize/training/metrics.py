"""Evaluation metrics for transcription + diarization.

Implements CER (character error rate), cpCER (concatenated minimum-permutation
CER), approximate DER (diarization error rate) and timestamp MAE. These mirror
the objective metrics reported in the project README so fine-tuning progress can
be compared on the same scale.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from moss_transcribe_diarize.transcript_parser import TranscriptSegment, parse_transcript


# --------------------------------------------------------------------------- #
# character error rate
# --------------------------------------------------------------------------- #
def _edit_distance(ref: list[str], hyp: list[str]) -> int:
    m, n = len(ref), len(hyp)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n]


def _cer_pair(ref: str, hyp: str) -> tuple[int, int]:
    """Return (edits, ref_char_count)."""
    ref_chars = list(ref)
    hyp_chars = list(hyp)
    edits = _edit_distance(ref_chars, hyp_chars)
    return edits, len(ref_chars)


def compute_cer(reference: str, hypothesis: str) -> float:
    """Plain character error rate between two raw transcript strings."""
    edits, ref_len = _cer_pair(reference, hypothesis)
    return edits / max(ref_len, 1)


# --------------------------------------------------------------------------- #
# cpCER: concatenated minimum-permutation CER
# --------------------------------------------------------------------------- #
def _group_by_speaker(segments: list[TranscriptSegment]) -> dict[str, str]:
    grouped: dict[str, list[str]] = {}
    for seg in segments:
        grouped.setdefault(seg.speaker, []).append(seg.text)
    return {spk: "".join(parts) for spk, parts in grouped.items()}


def compute_cpcer(reference: str, hypothesis: str) -> float:
    """cpCER: per-speaker concatenated CER minimized over speaker assignment.

    Uses the Hungarian algorithm when ``scipy`` is available, otherwise falls
    back to brute force over permutations (feasible for <=7 speakers).
    """
    ref_segs = parse_transcript(reference)
    hyp_segs = parse_transcript(hypothesis)
    if not ref_segs:
        return compute_cer(reference, hypothesis)

    ref_groups = _group_by_speaker(ref_segs)
    hyp_groups = _group_by_speaker(hyp_segs) if hyp_segs else {}

    ref_keys = list(ref_groups)
    hyp_keys = list(hyp_groups)
    n_ref, n_hyp = len(ref_keys), len(hyp_keys)

    # edit-distance matrix between every ref speaker and every hyp speaker
    matrix = np.zeros((n_ref, n_hyp), dtype=np.float64)
    for i, rk in enumerate(ref_keys):
        for j, hk in enumerate(hyp_keys):
            edits, _ = _cer_pair(ref_groups[rk], hyp_groups[hk])
            matrix[i, j] = edits

    total_edits = _assign_min_total(matrix)

    # unassigned ref speakers -> all deletions; unassigned hyp speakers -> insertions
    matched_ref = min(n_ref, n_hyp)
    if n_ref > matched_ref:
        for i in range(matched_ref, n_ref):
            total_edits += len(ref_groups[ref_keys[i]])
    if n_hyp > matched_ref:
        for j in range(matched_ref, n_hyp):
            total_edits += len(hyp_groups[hyp_keys[j]])

    ref_chars = sum(len(v) for v in ref_groups.values())
    return total_edits / max(ref_chars, 1)


def _assign_min_total(matrix: np.ndarray) -> float:
    n_ref, n_hyp = matrix.shape
    if n_ref == 0 or n_hyp == 0:
        return 0.0
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(matrix)
        return float(matrix[rows, cols].sum())
    except ImportError:
        # brute force over the smaller side
        from itertools import permutations

        if n_ref <= n_hyp:
            best = None
            for perm in permutations(range(n_hyp), n_ref):
                total = sum(matrix[i, perm[i]] for i in range(n_ref))
                if best is None or total < best:
                    best = total
            return float(best or 0.0)
        else:
            best = None
            for perm in permutations(range(n_ref), n_hyp):
                total = sum(matrix[perm[j], j] for j in range(n_hyp))
                if best is None or total < best:
                    best = total
            return float(best or 0.0)


# --------------------------------------------------------------------------- #
# timestamp MAE
# --------------------------------------------------------------------------- #
def compute_timestamp_mae(reference: str, hypothesis: str) -> float:
    """Mean absolute error of segment boundaries, paired by sorted rank."""
    ref_segs = parse_transcript(reference)
    hyp_segs = parse_transcript(hypothesis)
    if not ref_segs or not hyp_segs:
        return 0.0

    ref_starts = sorted(s.start for s in ref_segs)
    hyp_starts = sorted(s.start for s in hyp_segs)
    ref_ends = sorted(s.end for s in ref_segs)
    hyp_ends = sorted(s.end for s in hyp_segs)

    n = min(len(ref_starts), len(hyp_starts))
    errors = [abs(ref_starts[i] - hyp_starts[i]) for i in range(n)]
    m2 = min(len(ref_ends), len(hyp_ends))
    errors.extend(abs(ref_ends[i] - hyp_ends[i]) for i in range(m2))
    return float(np.mean(errors)) if errors else 0.0


# --------------------------------------------------------------------------- #
# approximate DER (diarization error rate)
# --------------------------------------------------------------------------- #
def compute_der(reference: str, hypothesis: str, step: float = 0.01) -> float:
    """Grid-based approximate DER over the union timeline."""
    ref_segs = parse_transcript(reference)
    hyp_segs = parse_transcript(hypothesis)
    if not ref_segs:
        return 0.0

    end = max((s.end for s in ref_segs + hyp_segs), default=0.0)
    if end <= 0.0:
        return 0.0

    ref_spans = [(s.start, s.end, s.speaker) for s in ref_segs]
    hyp_spans = [(s.start, s.end, s.speaker) for s in hyp_segs]

    def active(spans, t):
        return {spk for (st, en, spk) in spans if st <= t < en}

    ref_total = confusion = false_alarm = missed = 0.0
    t = 0.0
    while t < end:
        r = active(ref_spans, t)
        h = active(hyp_spans, t)
        ref_total += len(r)
        confusion += len(r - h)
        false_alarm += len(h - r)
        missed += len(r - h)
        t += step

    return (confusion + false_alarm + missed) / max(ref_total, 1.0)


# --------------------------------------------------------------------------- #
# teacher-forced proxy metric (free, from logits)
# --------------------------------------------------------------------------- #
def compute_token_accuracy(eval_pred) -> dict:
    """Token-level accuracy on non-ignored positions (teacher-forced proxy).

    Handles both raw logits ([B,T,V]) and pre-argmaxed ids ([B,T]).
    """
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    if logits.ndim == labels.ndim + 1:
        preds = np.argmax(logits, axis=-1)
    else:
        preds = logits
    mask = labels != -100
    if mask.sum() == 0:
        return {"token_accuracy": 0.0}
    correct = (preds == labels) & mask
    return {"token_accuracy": float(correct.sum() / mask.sum())}


def summarize(references: Iterable[str], hypotheses: Iterable[str]) -> dict:
    """Aggregate metrics over a corpus; also report Δcp = CER - cpCER."""
    refs = list(references)
    hyps = list(hypotheses)
    n = min(len(refs), len(hyps))
    if n == 0:
        return {}
    cers = [compute_cer(refs[i], hyps[i]) for i in range(n)]
    cpcers = [compute_cpcer(refs[i], hyps[i]) for i in range(n)]
    ts = [compute_timestamp_mae(refs[i], hyps[i]) for i in range(n)]
    ders = [compute_der(refs[i], hyps[i]) for i in range(n)]
    cer = float(np.mean(cers))
    cpcer = float(np.mean(cpcers))
    return {
        "n": n,
        "CER": cer,
        "cpCER": cpcer,
        "delta_cp": cer - cpcer,
        "timestamp_mae_s": float(np.mean(ts)),
        "DER": float(np.mean(ders)),
    }
