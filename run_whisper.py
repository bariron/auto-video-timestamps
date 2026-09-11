import argparse
import json
from pathlib import Path

import torch
from transformers import pipeline


DEFAULT_MODEL = "coriollon/whisper-large-v3-turbo-russian-codeswitch"


def format_timestamp(seconds: float | None) -> str:
    if seconds is None:
        return "00:00:00"

    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def build_pipeline(model_name: str):
    device = 0 if torch.cuda.is_available() else -1
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    return pipeline(
        task="automatic-speech-recognition",
        model=model_name,
        torch_dtype=torch_dtype,
        device=device,
    )


def transcribe(input_path: Path, model_name: str) -> dict:
    recognizer = build_pipeline(model_name)
    return recognizer(
        str(input_path),
        return_timestamps=True,
        generate_kwargs={"language": "ru", "task": "transcribe"},
    )


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
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    result = transcribe(args.input, args.model)
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
