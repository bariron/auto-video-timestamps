# auto-video-timestamps

Automatic timestamp generation for video and audio files using Whisper.

The project starts as a small command-line tool and will grow into a workflow for creating video chapters, subtitles, and structured summaries.

## Setup

The app uses FFmpeg to read long videos and split them into audio chunks. On Windows, `imageio-ffmpeg` usually provides a bundled FFmpeg binary through `requirements.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

```powershell
python run_whisper.py path\to\video.mp4 --output timestamps.txt
```

Long lectures are split into 10-minute chunks by default:

```powershell
python run_whisper.py lecture.mp4 --chunk-seconds 600 --overlap-seconds 5 --output timestamps.txt
```

For a quick 5-minute prototype run:

```powershell
python run_whisper.py lecture.mp4 --prototype-5min --plan-only
python run_whisper.py lecture.mp4 --prototype-5min --output prototype_timestamps.txt --json prototype_result.json
```

This processes only the first 5 minutes and uses 60-second chunks with the default 5-second overlap.

You can also save the raw Whisper output:

```powershell
python run_whisper.py path\to\video.mp4 --output timestamps.txt --json result.json
```

## Roadmap

- Generate speech-based timestamps.
- Group transcript segments into meaningful video chapters.
- Export YouTube-ready chapter descriptions.
- Export subtitles in SRT/VTT formats.
- Add a simple UI for local video processing.
