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


def build_chunk_plan(
    duration: float,
    chunk_seconds: int,
    overlap_seconds: int,
) -> list[dict]:
    step_seconds = chunk_seconds - overlap_seconds
    chunk_count = max(1, math.ceil(duration / step_seconds))
    chunks = []

    for index in range(chunk_count):
        start = index * step_seconds
        if start >= duration:
            break

        remaining = duration - start
        current_duration = min(chunk_seconds, remaining)
        chunks.append(
            {
                "index": index,
                "start": start,
                "duration": current_duration,
                "skip_before": overlap_seconds if index > 0 else 0,
            }
        )

    return chunks


def validate_processing_args(
    chunk_seconds: int,
    overlap_seconds: int,
    max_duration_seconds: int | None,
) -> None:
    if chunk_seconds <= 0:
        raise ValueError("--chunk-seconds must be greater than 0")
    if overlap_seconds < 0:
        raise ValueError("--overlap-seconds must be 0 or greater")
    if overlap_seconds >= chunk_seconds:
        raise ValueError("--overlap-seconds must be smaller than --chunk-seconds")
    if max_duration_seconds is not None and max_duration_seconds <= 0:
        raise ValueError("--max-duration-seconds must be greater than 0")


def transcribe_long_media(
    input_path: Path,
    model_name: str,
    chunk_seconds: int,
    overlap_seconds: int,
    max_duration_seconds: int | None,
) -> dict:
    validate_processing_args(chunk_seconds, overlap_seconds, max_duration_seconds)

    source_duration = get_media_duration(input_path)
    duration = (
        min(source_duration, max_duration_seconds)
        if max_duration_seconds is not None
        else source_duration
    )
    chunk_plan = build_chunk_plan(duration, chunk_seconds, overlap_seconds)
    all_chunks = []
    recognizer = build_pipeline(model_name)

    with tempfile.TemporaryDirectory(prefix="auto-video-timestamps-") as tmp_dir:
        tmp_path = Path(tmp_dir)

        for chunk in chunk_plan:
            index = chunk["index"]
            start = chunk["start"]
            current_duration = chunk["duration"]
            chunk_path = tmp_path / f"chunk_{index:04d}.wav"

            print(
                f"Processing chunk {index + 1}/{len(chunk_plan)}: "
                f"{format_timestamp(start)} - {format_timestamp(start + current_duration)}",
                file=sys.stderr,
            )
            extract_audio_chunk(input_path, chunk_path, start, current_duration)
            result = transcribe_chunk(recognizer, chunk_path)
            all_chunks.extend(normalize_chunk_result(result, start, chunk["skip_before"]))

    return {
        "text": " ".join(chunk["text"] for chunk in all_chunks),
        "chunks": all_chunks,
        "metadata": {
            "duration_seconds": duration,
            "source_duration_seconds": source_duration,
            "chunk_seconds": chunk_seconds,
            "overlap_seconds": overlap_seconds,
            "max_duration_seconds": max_duration_seconds,
            "model": model_name,
        },
    }


def render_chunk_plan(
    input_path: Path,
    chunk_seconds: int,
    overlap_seconds: int,
    max_duration_seconds: int | None,
) -> str:
    validate_processing_args(chunk_seconds, overlap_seconds, max_duration_seconds)

    source_duration = get_media_duration(input_path)
    duration = (
        min(source_duration, max_duration_seconds)
        if max_duration_seconds is not None
        else source_duration
    )
    lines = [
        f"Input: {input_path}",
        f"Source duration: {format_timestamp(source_duration)}",
        f"Processed duration: {format_timestamp(duration)}",
        f"Chunk size: {chunk_seconds}s",
        f"Overlap: {overlap_seconds}s",
        "",
        "Chunk plan:",
    ]

    for chunk in build_chunk_plan(duration, chunk_seconds, overlap_seconds):
        start = chunk["start"]
        end = start + chunk["duration"]
        skip_before = chunk["skip_before"]
        lines.append(
            f"{chunk['index'] + 1:02d}. "
            f"{format_timestamp(start)} - {format_timestamp(end)} "
            f"(skip first {skip_before}s after overlap)"
        )

    return "\n".join(lines)


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
    parser.add_argument(
        "--max-duration-seconds",
        type=int,
        help="Only process the first N seconds of the input file.",
    )
    parser.add_argument(
        "--prototype-5min",
        action="store_true",
        help="Process the first 5 minutes with 60-second chunks.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print the chunk plan without loading Whisper or transcribing.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    if args.prototype_5min:
        if args.max_duration_seconds is None:
            args.max_duration_seconds = 300
        if "--chunk-seconds" not in sys.argv:
            args.chunk_seconds = 60

    if args.plan_only:
        print(
            render_chunk_plan(
                args.input,
                args.chunk_seconds,
                args.overlap_seconds,
                args.max_duration_seconds,
            )
        )
        return

    result = transcribe_long_media(
        args.input,
        args.model,
        args.chunk_seconds,
        args.overlap_seconds,
        args.max_duration_seconds,
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
