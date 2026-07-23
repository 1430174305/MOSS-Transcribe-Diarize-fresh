#!/usr/bin/env python3
"""Long-audio inference: VAD windowing + per-window generation + front-wins merge.

For audio longer than the model's ~90-min context, this script:
1. plans windows with ABSOLUTE nominal anchors (0, T, 2T, ...), no drift;
   each head is the EARLIEST VAD silence in [nom-L, nom] and each tail is the
   LATEST VAD silence in [nom, nom+L] (extremal picks, max overlap ~2L);
2. if the remaining audio <= --max_remaining_minutes, takes the whole tail as
   one final window (no point cutting a 25-min remainder);
3. cuts each window (ffmpeg, 16 kHz mono) and runs greedy generation;
4. merges per-window transcripts via merge_windows.merge (front-window-wins:
   boundary at the VAD silence, overlap text of the later window is dropped).

Cross-window SPEAKER consistency is NOT done here -- it needs a speaker encoder
(global_diarize.py); merge keeps unique (window, local) labels.

Usage:
    python scripts/windowed_inference.py --model outputs/finetuned \
        --audio meeting.wav --output_dir runs/long \
        --target_minutes 20 --tolerance_minutes 2 --max_remaining_minutes 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

# reuse the VAD helpers from segment_long (same scripts/ dir on sys.path)
from segment_long import cut_audio, probe_duration, silence_gaps
from merge_windows import merge

from moss_transcribe_diarize.inference_utils import (
    build_transcription_messages,
    generate_transcription,
)


def _gap_midpoints(gaps):
    """silence gaps -> list of midpoints (seconds). gaps may be None (rigid)."""
    if not gaps:
        return []
    return [(g0 + g1) / 2.0 for (g0, g1) in gaps]


def pick_head(gaps, nominal_start: float, L: float) -> float:
    """Earliest VAD silence in [nominal_start - L, nominal_start]; rigid nom-L."""
    lo, hi = nominal_start - L, nominal_start
    mids = [m for m in _gap_midpoints(gaps) if lo <= m <= hi]
    if mids:
        return min(mids, key=lambda m: abs(m - lo))  # closest to lo = earliest
    return lo  # rigid


def pick_tail(gaps, nominal_end: float, L: float) -> float:
    """Latest VAD silence in [nominal_end, nominal_end + L]; rigid nom+L."""
    lo, hi = nominal_end, nominal_end + L
    mids = [m for m in _gap_midpoints(gaps) if lo <= m <= hi]
    if mids:
        return min(mids, key=lambda m: abs(m - hi))  # closest to hi = latest
    return hi  # rigid


def pick_nearest(gaps, target: float, L: float) -> float:
    """VAD silence nearest `target` within [target-L, target+L]; rigid target.

    Used for back-to-back (no-overlap) cutting: cut at the silence closest to
    the nominal end, next window starts exactly there (no overlap).
    """
    lo, hi = target - L, target + L
    mids = [m for m in _gap_midpoints(gaps) if lo <= m <= hi]
    if mids:
        return min(mids, key=lambda m: abs(m - target))
    return target  # rigid


def plan_windows(duration, T, L, max_remaining, min_seg, gaps, no_overlap=False):
    """Return [(start, end), ...].

    - overlap mode (default, for old same-time 1:1 scheme): absolute nominal
      anchors, head earliest / tail latest, ~2L overlap between windows.
    - no_overlap mode (for montage-voting scheme): back-to-back, each window
      ends at the VAD silence nearest (start + T), next starts exactly there.
    """
    windows = []
    if no_overlap:
        cursor = 0.0
        while cursor < duration:
            start = cursor
            if duration - start <= max_remaining:
                windows.append((start, duration))
                break
            end = pick_nearest(gaps, start + T, L)
            if end >= duration or end - start < min_seg:
                windows.append((start, duration))
                break
            windows.append((start, end))
            cursor = end
        return windows

    N = 0
    while True:
        nominal_start = N * T
        if nominal_start >= duration:
            break
        start = 0.0 if N == 0 else pick_head(gaps, nominal_start, L)
        start = max(0.0, min(start, duration))
        # max_remaining: take the whole remainder as the final window
        if duration - start <= max_remaining:
            windows.append((start, duration))
            break
        nominal_end = (N + 1) * T
        end = pick_tail(gaps, nominal_end, L)
        if end >= duration or end - start < min_seg:
            windows.append((start, duration))
            break
        windows.append((start, end))
        N += 1
    return windows


@torch.inference_mode()
def infer_window(model, processor, audio_path, device, dtype, max_length, max_new_tokens):
    messages = build_transcription_messages(audio_path)
    result = generate_transcription(
        model, processor, messages,
        max_length=max_length, max_new_tokens=max_new_tokens,
        do_sample=False, device=device, dtype=dtype,
    )
    return result["text"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--target_minutes", type=float, default=20.0)
    ap.add_argument("--tolerance_minutes", type=float, default=2.0)
    ap.add_argument("--max_remaining_minutes", type=float, default=30.0)
    ap.add_argument("--min_gap_seconds", type=float, default=1.0)
    ap.add_argument("--min_segment_seconds", type=float, default=10.0)
    ap.add_argument("--max_new_tokens", type=int, default=65536)
    ap.add_argument("--max_length", type=int, default=131072)
    ap.add_argument("--no_overlap", action="store_true",
                    help="back-to-back cuts at VAD silence (for montage-voting scheme; default overlaps for old 1:1)")
    ap.add_argument("--dtype", default="auto")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoProcessor

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    audio = Path(args.audio).expanduser().resolve()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (args.dtype == "auto" and device.type == "cuda") else (
        getattr(torch, args.dtype) if args.dtype != "auto" else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, dtype=dtype).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    T = args.target_minutes * 60.0
    L = args.tolerance_minutes * 60.0
    max_rem = args.max_remaining_minutes * 60.0
    min_seg = args.min_segment_seconds

    duration = probe_duration(audio)
    gaps = silence_gaps(audio, min_gap=args.min_gap_seconds)
    if gaps is None:
        print("[warn] silero-vad unavailable, using rigid cuts")
        gaps = []
    else:
        print(f"[info] {len(gaps)} silence gaps (>= {args.min_gap_seconds}s) found")

    windows = plan_windows(duration, T, L, max_rem, min_seg, gaps, no_overlap=args.no_overlap)
    print(f"[info] planned {len(windows)} windows for {duration:.1f}s audio")

    # per-window inference
    win_records = []
    for i, (start, end) in enumerate(windows):
        clip = out_dir / f"window_{i:03d}.wav"
        cut_audio(audio, start, end - start, clip)
        text = infer_window(model, processor, str(clip), device, dtype, args.max_length, args.max_new_tokens)
        win_records.append({"offset": start, "end": end, "transcript": text})
        print(f"[info] window {i}: [{start:.1f}, {end:.1f}] ({end-start:.1f}s) -> {len(text)} chars")

    # front-wins merge
    merged, mapping = merge(win_records)
    merged_text = "".join(f"[{s.start:.2f}][{s.speaker}]{s.text}[{s.end:.2f}]" for s in merged)

    (out_dir / "merged.txt").write_text(merged_text, encoding="utf-8")
    (out_dir / "windows.json").write_text(json.dumps(win_records, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "speaker_mapping.json").write_text(json.dumps(
        {"mapping": {f"{wi}:{lspk}": g for (wi, lspk), g in mapping.items()},
         "note": "unique per (window,local). Refine same-person with global_diarize.py."},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] merged {len(merged)} segments -> {out_dir/'merged.txt'}")


if __name__ == "__main__":
    main()
