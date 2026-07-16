#!/usr/bin/env python3
"""Prepare and normalize a manifest for fine-tuning.

Takes a simple input manifest and emits the conversation-format JSONL that
`finetune.py` consumes. It:

- parses raw transcripts with the shipped parser
- renumbers speakers by first appearance ([S01], [S02], ...) for label stability
- validates timestamps (non-decreasing starts, end >= start) and drops bad rows
- renders a canonical ``[start][Sxx]text[end]`` target
- writes audio duration as ``length`` for length-bucketed sampling
- optionally splits into train / eval sets

Input JSONL (one record per line):
    {"audio": "path.wav", "transcript": "[0.5][S03]hi[1.2]...", "prompt"?: "..."}

Output JSONL record:
    {"conversation":[{user/text},{user/audio},{assistant/text}], "length": 12.34}
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import soundfile as sf
from transformers import HfArgumentParser

from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT
from moss_transcribe_diarize.transcript_parser import (
    TranscriptSegment,
    parse_transcript,
)


@dataclass
class PrepArguments:
    input_jsonl: str = field(metadata={"help": "Raw manifest with audio + transcript."})
    output_dir: str = field(metadata={"help": "Where to write train/eval JSONL."})
    train_split: float = 0.95
    prompt: str | None = field(default=None, metadata={"help": "Override prompt for all rows."})
    seed: int = 42
    min_segments: int = 1


def render_segment(seg: TranscriptSegment) -> str:
    return f"[{seg.start:.2f}][{seg.speaker}]{seg.text}[{seg.end:.2f}]"


def renumber(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    mapping: dict[str, str] = {}
    out: list[TranscriptSegment] = []
    for seg in segments:
        if seg.speaker not in mapping:
            mapping[seg.speaker] = f"S{len(mapping) + 1:02d}"
        out.append(
            TranscriptSegment(start=seg.start, end=seg.end, speaker=mapping[seg.speaker], text=seg.text)
        )
    return out


def validate(segments: list[TranscriptSegment], min_segments: int) -> bool:
    if len(segments) < min_segments:
        return False
    prev_start = -1.0
    for seg in segments:
        if seg.end < seg.start:
            return False
        if seg.start < prev_start:
            return False
        prev_start = seg.start
    return True


def audio_duration(path: str) -> float:
    info = sf.info(path)
    return info.frames / info.samplerate


def normalize_row(row: dict, root: Path, prompt: str, min_segments: int) -> dict | None:
    audio = row.get("audio")
    transcript = row.get("transcript")
    if not audio or not transcript:
        return None
    audio_path = Path(audio).expanduser()
    if not audio_path.is_absolute():
        audio_path = (root / audio_path).resolve()
    if not audio_path.exists():
        return None

    segments = parse_transcript(str(transcript))
    if not validate(segments, min_segments):
        return None
    segments = renumber(segments)
    target = "".join(render_segment(s) for s in segments)
    used_prompt = (prompt or row.get("prompt") or DEFAULT_PROMPT).strip()

    try:
        length = audio_duration(str(audio_path))
    except Exception:
        length = 0.0

    return {
        "conversation": [
            {"role": "user", "message_type": "text", "content": used_prompt},
            {"role": "user", "message_type": "audio", "content": str(audio_path)},
            {"role": "assistant", "message_type": "text", "content": target},
        ],
        "length": length,
    }


def main() -> None:
    parser = HfArgumentParser(PrepArguments)
    args = parser.parse_args_into_dataclass()[0]

    src = Path(args.input_jsonl).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt = args.prompt.strip() if args.prompt else None

    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    skipped = 0
    with src.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            normalized = normalize_row(row, src.parent, prompt or DEFAULT_PROMPT, args.min_segments)
            if normalized is None:
                skipped += 1
                continue
            key = (normalized["conversation"][1]["content"], normalized["conversation"][2]["content"])
            if key in seen:
                skipped += 1
                continue
            seen.add(key)
            rows.append(normalized)

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_train = int(len(rows) * args.train_split)
    train_rows, eval_rows = rows[:n_train], rows[n_train:]

    def write_jsonl(path: Path, data: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for item in data:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "eval.jsonl", eval_rows)
    print(f"prepared {len(train_rows)} train + {len(eval_rows)} eval rows, skipped {skipped} bad/dup rows")


if __name__ == "__main__":
    main()
