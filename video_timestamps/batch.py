import argparse
import json
from pathlib import Path

from .paths import INPUT_DIR as DEFAULT_INPUT_DIR
from .paths import OUTPUT_DIR as DEFAULT_OUTPUT_DIR
from .transcription import render_timestamps, transcribe_long_media

MEDIA_EXTENSIONS = {
    ".avi",
    ".m4a",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".wav",
    ".webm",
}


def find_media_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS
    )


def output_base_name(input_path: Path) -> str:
    return input_path.stem.replace(" ", "_")


def process_file(
    input_path: Path,
    output_dir: Path,
    model_name: str,
    chunk_seconds: int,
    overlap_seconds: int,
    max_duration_seconds: int | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = output_base_name(input_path)
    timestamps_path = output_dir / f"{base_name}_timestamps.txt"
    json_path = output_dir / f"{base_name}_result.json"

    result = transcribe_long_media(
        input_path=input_path,
        model_name=model_name,
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds,
        max_duration_seconds=max_duration_seconds,
    )
    timestamps_path.write_text(render_timestamps(result), encoding="utf-8")
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved: {timestamps_path}")
    print(f"Saved: {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process all lecture media files from the data directory."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory with lecture videos/audio. Default: data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated files. Default: outputs.",
    )
    parser.add_argument(
        "--model",
        default="coriollon/whisper-large-v3-turbo-russian-codeswitch",
        help="Hugging Face ASR model name.",
    )
    parser.add_argument("--chunk-seconds", type=int, default=600)
    parser.add_argument("--overlap-seconds", type=int, default=5)
    parser.add_argument(
        "--max-duration-seconds",
        type=int,
        help="Only process the first N seconds from each media file.",
    )
    args = parser.parse_args()

    media_files = find_media_files(args.input_dir)
    if not media_files:
        print(f"No media files found in {args.input_dir}")
        return

    for index, media_file in enumerate(media_files, start=1):
        print(f"[{index}/{len(media_files)}] Processing {media_file}")
        process_file(
            input_path=media_file,
            output_dir=args.output_dir,
            model_name=args.model,
            chunk_seconds=args.chunk_seconds,
            overlap_seconds=args.overlap_seconds,
            max_duration_seconds=args.max_duration_seconds,
        )


if __name__ == "__main__":
    main()
