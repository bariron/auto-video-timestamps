import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_MODEL = "coriollon/whisper-large-v3-turbo-russian-codeswitch"
SAMPLE_RATE = 16000


def format_timestamp(seconds: float | None) -> str:
    if seconds is None:
        return "00:00:00"

    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def build_pipeline(model_name: str):
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    return pipeline(
        task="automatic-speech-recognition",
        model=model_name,
        torch_dtype=torch_dtype,
        device=device,
    )


def get_ffmpeg_executable() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def run_command(command: list[str], check: bool = True) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        command,
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed


def get_media_duration(input_path: Path) -> float:
    completed = run_command(
        [
            get_ffmpeg_executable(),
            "-hide_banner",
            "-i",
            str(input_path),
        ],
        check=False,
    )
    match = re.search(
        r"Duration:\s*(\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)",
        completed.stderr,
    )
    if not match:
        raise RuntimeError(f"Could not read media duration: {input_path}")

    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def extract_audio_chunk(
    input_path: Path,
    output_path: Path,
    start_seconds: float,
    duration_seconds: float,
) -> None:
    run_command(
        [
            get_ffmpeg_executable(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            str(start_seconds),
            "-t",
            str(duration_seconds),
            "-i",
            str(input_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            str(output_path),
        ]
    )


def transcribe_chunk(recognizer, chunk_path: Path) -> dict:
    return recognizer(
        str(chunk_path),
        return_timestamps=True,
        generate_kwargs={"language": "ru", "task": "transcribe"},
    )


def normalize_chunk_result(result: dict, offset_seconds: float, skip_before: float) -> list[dict]:
    chunks = result.get("chunks") or []
    if not chunks:
        text = result.get("text", "").strip()
        if not text:
            return []

        return [
            {
                "timestamp": [offset_seconds + skip_before, None],
                "text": text,
            }
        ]

    normalized = []
    for chunk in chunks:
        start, end = chunk.get("timestamp", (None, None))
        text = chunk.get("text", "").strip()
        if not text:
            continue

        if end is not None and end <= skip_before:
            continue

        absolute_start = None if start is None else offset_seconds + max(start, skip_before)
        absolute_end = None if end is None else offset_seconds + end
        normalized.append(
            {
                "timestamp": [absolute_start, absolute_end],
                "text": text,
            }
        )

    return normalized


def transcribe_long_media(
    input_path: Path,
    model_name: str,
    chunk_seconds: int,
    overlap_seconds: int,
) -> dict:
    if overlap_seconds >= chunk_seconds:
        raise ValueError("--overlap-seconds must be smaller than --chunk-seconds")

    recognizer = build_pipeline(model_name)
    duration = get_media_duration(input_path)
    step_seconds = chunk_seconds - overlap_seconds
    chunk_count = max(1, math.ceil(duration / step_seconds))
    all_chunks = []

    with tempfile.TemporaryDirectory(prefix="auto-video-timestamps-") as tmp_dir:
        tmp_path = Path(tmp_dir)

        for index in range(chunk_count):
            start = index * step_seconds
            if start >= duration:
                break

            remaining = duration - start
            current_duration = min(chunk_seconds, remaining)
            chunk_path = tmp_path / f"chunk_{index:04d}.wav"

            print(
                f"Processing chunk {index + 1}/{chunk_count}: "
                f"{format_timestamp(start)} - {format_timestamp(start + current_duration)}",
                file=sys.stderr,
            )
            extract_audio_chunk(input_path, chunk_path, start, current_duration)
            result = transcribe_chunk(recognizer, chunk_path)
            skip_before = overlap_seconds if index > 0 else 0
            all_chunks.extend(normalize_chunk_result(result, start, skip_before))

    return {
        "text": " ".join(chunk["text"] for chunk in all_chunks),
        "chunks": all_chunks,
        "metadata": {
            "duration_seconds": duration,
            "chunk_seconds": chunk_seconds,
            "overlap_seconds": overlap_seconds,
            "model": model_name,
        },
    }


def render_timestamps(result: dict) -> str:
    chunks = result.get("chunks") or []
    if not chunks:
        return result.get("text", "").strip()

    lines = []
    for chunk in chunks:
        start, end = chunk.get("timestamp", (None, None))
        text = chunk.get("text", "").strip()
        if text:
            lines.append(f"{format_timestamp(start)} - {format_timestamp(end)} {text}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create speech timestamps for video and audio files with Whisper."
    )
    parser.add_argument("input", type=Path, help="Path to a video or audio file.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Hugging Face model name. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for a text file with generated timestamps.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Optional path for the raw Whisper result as JSON.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=600,
        help="Lecture chunk size in seconds. Default: 600.",
    )
    parser.add_argument(
        "--overlap-seconds",
        type=int,
        default=5,
        help="Overlap between lecture chunks in seconds. Default: 5.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    result = transcribe_long_media(
        args.input,
        args.model,
        args.chunk_seconds,
        args.overlap_seconds,
    )
    text = render_timestamps(result)

    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)

    if args.json:
        args.json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
