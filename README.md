# auto-video-timestamps

Automatic timestamp generation for video and audio files using Whisper.

The project starts as a small command-line tool and will grow into a workflow for creating video chapters, subtitles, and structured summaries.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

```powershell
python run_whisper.py path\to\video.mp4 --output timestamps.txt
```

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
