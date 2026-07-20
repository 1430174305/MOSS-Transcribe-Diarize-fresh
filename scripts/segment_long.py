#!/usr/bin/env python3
"""Split long audio + transcripts into <=N-minute segments for training/inference.

The model's context is ~128k tokens (~90 min). Audio longer than that CANNOT be
fed as one sequence regardless of GPU count -- it must be segmented. This script
cuts each (audio, transcript) row into consecutive windows of ~--segment_minutes,
preferring to cut at SILENCE (via silero-vad) so speaker turns are not truncated.
Writes per-segment audio (16 kHz mono via ffmpeg) and time-shifted transcripts
(each starts at 0.0).

Input JSONL (one per line): {"audio": "path.wav", "transcript": "[0.5][S03]...[...]"}
Output: a new JSONL where each line is one segment, ready for prepare_data.py.

Usage:
    python scripts/segment_long.py --input raw.jsonl --segment_minutes 30 \
        --tolerance_minutes 5 --output raw_seg.jsonl --audio_out data/seg
    python prepare_data.py --input_jsonl raw_seg.jsonl --output_dir data --train_split 0.95

VAD: requires `silero-vad` (pip install silero-vad). If unavailable, the script
prints a warning and falls back to rigid cuts at exactly --segment_minutes.
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


def silence_gaps(audio_path: Path, min_gap: float = 0.3):
    """Return list of (start, end) silence gaps in seconds via silero-vad.

    Returns None if silero-vad is unavailable (caller falls back to rigid cuts).
    """
    try:
        import soundfile as sf
        import soxr
        from silero_vad import get_speech_timestamps, load_silero_vad
    except ImportError:
        return None
    wav, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != 16000:
        wav = soxr.resample(wav, sr, 16000)
        sr = 16000
    model = load_silero_vad()
    ts = get_speech_timestamps(wav, model, return_seconds=True, min_speech_duration_ms=250)
    gaps = []
    for i in range(len(ts) - 1):
        gs, ge = ts[i]["end"], ts[i + 1]["start"]
        if ge - gs >= min_gap:
            gaps.append((gs, ge))
    return gaps


def pick_cut(gaps, target: float, tol: float, duration: float, floor: float) -> float:
    """Pick a silence-gap midpoint nearest `target` within +-tol; else rigid."""
    if gaps:
        lo, hi = target - tol, target + tol
        cands = [((g[0] + g[1]) / 2.0) for g in gaps if lo <= (g[0] + g[1]) / 2.0 <= hi]
        if cands:
            return min(cands, key=lambda m: abs(m - target))
    cut = min(target, duration)
    return max(cut, floor)


def plan_cuts(duration: float, seg_sec: float, tol_sec: float, min_sec: float, gaps):
    """Return list of (start, end) consecutive segments covering [0, duration]."""
    cuts = []
    cursor = 0.0
    while cursor < duration - min_sec:
        target = cursor + seg_sec
        if target >= duration:
            cuts.append((cursor, duration))
            return cuts
        end = pick_cut(gaps, target, tol_sec, duration, cursor + min_sec)
        if end - cursor < min_sec:  # safety: avoid tiny segments
            end = min(target, duration)
        cuts.append((cursor, end))
        cursor = end
    if cursor < duration:
        cuts.append((cursor, duration))
    return cuts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="raw.jsonl with {audio, transcript}")
    ap.add_argument("--segment_minutes", type=float, default=30.0)
    ap.add_argument("--tolerance_minutes", type=float, default=5.0,
                    help="cut at the silence nearest the target within +-this")
    ap.add_argument("--min_segment_seconds", type=float, default=10.0)
    ap.add_argument("--min_gap_seconds", type=float, default=0.3,
                    help="minimum silence gap length to consider a cut point")
    ap.add_argument("--no_vad", action="store_true", help="disable VAD, use rigid cuts")
    ap.add_argument("--output", default="raw_seg.jsonl")
    ap.add_argument("--audio_out", default="data/seg")
    args = ap.parse_args()

    src = Path(args.input).expanduser().resolve()
    out_jsonl = Path(args.output)
    audio_out = Path(args.audio_out).expanduser().resolve()
    seg_sec = args.segment_minutes * 60.0
    tol_sec = args.tolerance_minutes * 60.0
    n_written = 0
    vad_on = not args.no_vad

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

            gaps = None
            if vad_on:
                gaps = silence_gaps(audio, min_gap=args.min_gap_seconds)
                if gaps is None:
                    print(f"[line {line_no}] silero-vad unavailable, falling back to rigid cuts")
                    gaps = []
                    vad_on = False  # warn once, then rigid for the rest

            cuts = plan_cuts(duration, seg_sec, tol_sec, args.min_segment_seconds, gaps)
            for i, (start, end) in enumerate(cuts):
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

    print(f"wrote {n_written} segments to {out_jsonl}")


if __name__ == "__main__":
    main()
