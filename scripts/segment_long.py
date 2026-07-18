#!/usr/bin/env python3
"""Split long audio + transcripts into <=N-minute segments for training/inference.

The model's context is ~128k tokens (~90 min). Audio longer than that CANNOT be
fed as one sequence regardless of GPU count -- it must be segmented. This script
cuts each (audio, transcript) row into consecutive --segment_minutes windows,
writing new per-segment audio files (16 kHz mono via ffmpeg) and time-shifted
transcripts so each segment starts at 0.0.

Input JSONL (one per line): {"audio": "path.wav", "transcript": "[0.5][S03]...[...]"}
Output: a new JSONL where each line is one segment, ready for prepare_data.py.

Usage:
    python scripts/segment_long.py --input raw.jsonl --segment_minutes 60 --output raw_seg.jsonl --audio_out data/seg
    python prepare_data.py --input_jsonl raw_seg.jsonl --output_dir data --train_split 0.95
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from moss_transcribe_diarize.transcript_parser import TranscriptSegment, parse_transcript

DEFAULT_PROMPT = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
    "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
    "并在段末标注结束时间戳，以清晰标明该段语音范围。"
)


def render(seg: TranscriptSegment) -> str:
    return f"[{seg.start:.2f}][{seg.speaker}]{seg.text}[{seg.end:.2f}]"


def split_transcript(segments: list[TranscriptSegment], start: float, end: float) -> str:
    """Return transcript text for [start, end), shifted so start -> 0.0."""
    out: list[TranscriptSegment] = []
    for s in segments:
        if s.end <= start or s.start >= end:
            continue
        new_start = max(s.start, start) - start
        new_end = min(s.end, end) - start
        if new_end > new_start:
            out.append(TranscriptSegment(start=new_start, end=new_end, speaker=s.speaker, text=s.text))
    return "".join(render(s) for s in out)


def probe_duration(path: Path) -> float:
    import soundfile as sf
    info = sf.info(str(path))
    return info.frames / info.samplerate


def cut_audio(src: Path, start: float, duration: float, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(src),
            "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
            "-ac", "1", "-ar", "16000", str(dest),
        ],
        check=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="raw.jsonl with {audio, transcript}")
    ap.add_argument("--segment_minutes", type=float, default=60.0)
    ap.add_argument("--output", default="raw_seg.jsonl")
    ap.add_argument("--audio_out", default="data/seg")
    ap.add_argument("--min_segment_seconds", type=float, default=10.0,
                    help="drop segments shorter than this (tail scraps)")
    args = ap.parse_args()

    src = Path(args.input).expanduser().resolve()
    out_jsonl = Path(args.output)
    audio_out = Path(args.audio_out).expanduser().resolve()
    seg_sec = args.segment_minutes * 60.0
    n_written = 0

    with out_jsonl.open("w", encoding="utf-8") as w:
        for line_no, line in enumerate(src.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            audio = Path(row["audio"]).expanduser()
            if not audio.is_absolute():
                audio = (src.parent / audio).resolve()
            transcript = row.get("transcript", "")
            segments = parse_transcript(transcript)

            try:
                duration = probe_duration(audio)
            except Exception as exc:
                print(f"[line {line_no}] skip, cannot probe {audio}: {exc}")
                continue

            n_segs = max(1, int((duration + seg_sec - 1) // seg_sec))
            for i in range(n_segs):
                start = i * seg_sec
                end = min((i + 1) * seg_sec, duration)
                if end - start < args.min_segment_seconds:
                    continue
                dest = audio_out / f"{audio.stem}_seg{i:03d}.wav"
                try:
                    cut_audio(audio, start, end - start, dest)
                except subprocess.CalledProcessError as exc:
                    print(f"[line {line_no} seg {i}] ffmpeg failed: {exc}")
                    continue
                seg_text = split_transcript(segments, start, end)
                if not seg_text.strip():
                    continue
                w.write(json.dumps(
                    {"audio": str(dest), "transcript": seg_text, "prompt": row.get("prompt", DEFAULT_PROMPT)},
                    ensure_ascii=False,
                ) + "\n")
                n_written += 1

    print(f"wrote {n_written} segments (<= {args.segment_minutes} min each) to {out_jsonl}")


if __name__ == "__main__":
    main()
