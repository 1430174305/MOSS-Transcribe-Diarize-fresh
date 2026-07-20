#!/usr/bin/env python3
"""Merge overlapping per-window MOSS transcripts into one global transcript.

Each window is an audio chunk (~30 min, optionally overlapping neighbors). Each
window's transcript has LOCAL timestamps (start at 0.0). This script:

1. shifts each window's segments to global time (add the window offset),
2. computes a "trusted range" [L_i, R_i] per window: the midpoint of each
   overlap goes to the window whose interior is on that side,
3. keeps a segment from window i iff its global START falls in [L_i, R_i]
   (so every global time is owned by exactly one window -- no text is cut,
   no duplicates),
4. sorts by global start,
5. renumbers speakers GLOBALLY but UNIQUELY per (window, local_label) -- this
   is a SAFE default that never false-merges two different people. To collapse
   same-person across windows, feed the output mapping into a speaker-encoder
   step (global_diarize.py) that refines (window, local) -> true global ID.

Input JSON:
{
  "windows": [
    {"offset": 0.0,    "transcript": "[0.0][S01]hello[1.2]..."},
    {"offset": 1770.0, "transcript": "[0.0][S01]world[2.0]..."}
  ]
}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from moss_transcribe_diarize.transcript_parser import TranscriptSegment, parse_transcript


def render(seg: TranscriptSegment) -> str:
    return f"[{seg.start:.2f}][{seg.speaker}]{seg.text}[{seg.end:.2f}]"


def window_end(w: dict) -> float:
    """Global end time of a window: explicit `end`, else offset + last seg end."""
    if "end" in w:
        return float(w["end"])
    segs = parse_transcript(w["transcript"])
    return float(w["offset"]) + (max((s.end for s in segs), default=0.0))


def trusted_ranges(windows: list[dict]) -> list[tuple[float, float]]:
    """[L_i, R_i]: keep a segment from window i iff its start is in here.

    - L_i = midpoint of the head overlap (with window i-1), else S_i.
    - R_i = midpoint of the tail overlap (with window i+1), else E_i.
    The LEFT half of an overlap is owned by the LEFT window, the RIGHT half by
    the RIGHT window -- each side goes to the window whose interior is closer.
    """
    n = len(windows)
    ends = [window_end(w) for w in windows]
    ranges = []
    for i, w in enumerate(windows):
        S = float(w["offset"])
        E = ends[i]
        L = S
        R = E
        if i > 0:  # head overlap with previous window
            prev_E = ends[i - 1]
            if S < prev_E:
                L = (S + prev_E) / 2.0
        if i < n - 1:  # tail overlap with next window
            nxt_S = float(windows[i + 1]["offset"])
            if nxt_S < E:
                R = (nxt_S + E) / 2.0
        ranges.append((L, R))
    return ranges


def merge(windows: list[dict]) -> tuple[list[TranscriptSegment], dict]:
    ranges = trusted_ranges(windows)
    kept: list[tuple[TranscriptSegment, int, str]] = []
    for i, w in enumerate(windows):
        L, R = ranges[i]
        offset = float(w["offset"])
        for s in parse_transcript(w["transcript"]):
            g_start = s.start + offset
            g_end = s.end + offset
            if L <= g_start <= R:  # own this segment
                seg = TranscriptSegment(start=g_start, end=g_end, speaker=s.speaker, text=s.text)
                kept.append((seg, i, s.speaker))
    kept.sort(key=lambda x: (x[0].start, x[0].end))

    # unique global renumber per (window, local_label) -- no false merges
    mapping: dict[tuple[int, str], str] = {}
    g = 0
    out: list[TranscriptSegment] = []
    for seg, wi, lspk in kept:
        key = (wi, lspk)
        if key not in mapping:
            g += 1
            mapping[key] = f"S{g:02d}"
        out.append(TranscriptSegment(start=seg.start, end=seg.end, speaker=mapping[key], text=seg.text))
    return out, mapping


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSON with {windows: [{offset, transcript, end?}]}")
    ap.add_argument("--output_transcript", default="merged.txt")
    ap.add_argument("--output_mapping", default="speaker_mapping.json")
    args = ap.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    windows = data["windows"]
    segs, mapping = merge(windows)

    Path(args.output_transcript).write_text("".join(render(s) for s in segs), encoding="utf-8")
    Path(args.output_mapping).write_text(
        json.dumps({"mapping": {f"{wi}:{lspk}": g for (wi, lspk), g in mapping.items()},
                    "note": "unique per (window,local). Refine same-person with a speaker encoder."},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"merged {len(segs)} segments, {len(mapping)} (window,local) speakers -> {args.output_transcript}")


if __name__ == "__main__":
    main()
