"""Process a YouTube or public Yandex Disk media link with Whisper."""

import argparse
import hashlib
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yandex_disk


def normalize_youtube_url(value: str) -> str:
    parsed = urlparse(value.strip())
    if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
        raise ValueError("Expected an http(s) YouTube video URL.")
    host = parsed.hostname
    parts = parsed.path.strip("/").split("/")
    video_id = None
    if host == "youtu.be" and len(parts) == 1:
        video_id = parts[0]
    elif host in {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [None])[0]
        elif len(parts) == 2 and parts[0] in {"shorts", "live", "embed"}:
            video_id = parts[1]
    if not video_id or not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError("Expected a single YouTube video link (watch, youtu.be, shorts or live).")
    return f"https://www.youtube.com/watch?v={video_id}"


def reject_live(info, *, incomplete=False):
    if info.get("is_live") or info.get("live_status") in {"is_live", "is_upcoming"}:
        return "Live and upcoming streams are unsupported; use a completed video."
    return None


def download_youtube_audio(url: str, directory: Path) -> Path:
    try:
        from yt_dlp import YoutubeDL
        from yt_dlp.utils import DownloadError
    except ImportError as exc:
        raise RuntimeError('Install YouTube support: python -m pip install -U "yt-dlp[default]"') from exc

    options = {
        "format": "bestaudio/best",
        "outtmpl": str(directory / "audio.%(ext)s"),
        "noplaylist": True,
        "match_filter": reject_live,
        "js_runtimes": {"deno": {}, "node": {}},
    }
    try:
        with YoutubeDL(options) as downloader:
            info = downloader.extract_info(url, download=True)
            if not info or info.get("_type", "video") != "video":
                raise RuntimeError("No downloadable completed video was returned.")
            path = Path(downloader.prepare_filename(info))
    except DownloadError as exc:
        raise RuntimeError(
            f"YouTube download failed: {exc}\n"
            'Try updating with python -m pip install -U "yt-dlp[default]" '
            "and ensure Deno or Node.js is installed."
        ) from exc
    if not path.is_file():
        raise RuntimeError("YouTube audio was not downloaded (the video may be live or unavailable).")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="YouTube or public Yandex Disk video/audio URL.")
    parser.add_argument("--disk-path", help="File path within a public Yandex Disk folder, e.g. /video.mp4.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model", help="Override the existing Whisper model.")
    parser.add_argument("--chunk-seconds", type=int, default=600)
    parser.add_argument("--overlap-seconds", type=int, default=5)
    parser.add_argument("--max-duration-seconds", type=int)
    parser.add_argument("--plan-only", action="store_true", help="Download audio and print the plan without loading Whisper.")
    args = parser.parse_args()
    try:
        is_disk = urlparse(args.url.strip()).hostname in yandex_disk.HOSTS
        url = yandex_disk.normalize_public_url(args.url) if is_disk else normalize_youtube_url(args.url)
        if args.disk_path is not None and (not is_disk or not args.disk_path.startswith("/")):
            raise ValueError("--disk-path requires a Yandex Disk link and a path starting with '/'.")
        if args.chunk_seconds <= 0 or not 0 <= args.overlap_seconds < args.chunk_seconds:
            raise ValueError("Require --chunk-seconds > 0 and 0 <= --overlap-seconds < --chunk-seconds.")
        if args.max_duration_seconds is not None and args.max_duration_seconds <= 0:
            raise ValueError("--max-duration-seconds must be greater than 0.")
        if is_disk:
            source_id = "yandex_" + hashlib.sha256((url + "\n" + (args.disk_path or "")).encode()).hexdigest()[:16]
        else:
            source_id = "youtube_" + parse_qs(urlparse(url).query)["v"][0]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        base = args.output_dir / source_id
        with tempfile.TemporaryDirectory(prefix="video-source-") as directory:
            if is_disk:
                audio_path = yandex_disk.download_public_media(url, Path(directory), args.disk_path)
            else:
                audio_path = download_youtube_audio(url, Path(directory))
            command = [
                sys.executable, str(Path(__file__).with_name("run_whisper.py")), str(audio_path),
                "--output", str(base.with_name(base.name + "_timestamps.txt")),
                "--json", str(base.with_name(base.name + "_result.json")),
                "--chunk-seconds", str(args.chunk_seconds),
                "--overlap-seconds", str(args.overlap_seconds),
            ]
            if args.model:
                command.extend(["--model", args.model])
            if args.max_duration_seconds is not None:
                command.extend(["--max-duration-seconds", str(args.max_duration_seconds)])
            if args.plan_only:
                command.append("--plan-only")
            subprocess.run(command, check=True)
        if not args.plan_only:
            print(f"Saved: {base}_timestamps.txt\nSaved: {base}_result.json")
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    except subprocess.CalledProcessError as exc:
        parser.exit(exc.returncode, "Whisper processing failed; see the error above.\n")


if __name__ == "__main__":
    main()
