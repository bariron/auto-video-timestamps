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

### Local web application

Install `requirements.txt`, then run:

```powershell
python web_app.py
```

Open http://127.0.0.1:7860. The Russian-language interface accepts YouTube and public Yandex Disk links, including full links to files inside shared folders. It queues videos, shows processing stages and logs, and provides chapter/transcript previews and TXT/JSON downloads. The optional preview mode transcribes only the first five minutes; it still downloads the entire source.

Whisper and chapter generation run sequentially in a background worker on this computer. Keep the server running while processing; closing the browser tab is fine. The server is local only, with one process and one video processed at a time. Change the port with `python web_app.py --port 7861`.

Jobs and their results are stored in `outputs/web_jobs/`. Existing results under `outputs/` also appear in the library. Queued jobs resume after a server restart; interrupted running jobs are marked as failed. Submit the link again to retry. Transcripts remain available if chapter generation fails. Automatically generated chapters still need review.

### Command line

Put lecture videos or audio files into `data/`. Generated transcripts and timestamps go to `outputs/`.

Process every lecture from `data/`:

```powershell
python process_data.py
```

For a quick test on the first 5 minutes of every file:

```powershell
python process_data.py --max-duration-seconds 300 --chunk-seconds 60
```

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

## Process a YouTube link

Install the requirements above (Python 3.10+) and [Deno](https://docs.deno.com/runtime/getting_started/installation/), or a supported Node.js version. YouTube downloads use the [yt-dlp Python API](https://github.com/yt-dlp/yt-dlp#embedding-yt-dlp); its current [JavaScript runtime requirements](https://github.com/yt-dlp/yt-dlp/wiki/EJS) apply. No API key is needed.

On Windows, install Deno with `winget install --id DenoLand.Deno -e` and restart the app, or put the official `deno.exe` in `.tools/` in the project. The wrapper also checks `PATH` and `~/.deno/bin`. A missing runtime now produces an explicit setup error before downloading. Downloads use a 60-second socket timeout, bounded retries and 4 MiB HTTP chunks. Network timeouts are reported separately from runtime setup problems.

```sh
python process_url.py "https://www.youtube.com/watch?v=VIDEO_ID"
python process_url.py "https://youtu.be/VIDEO_ID" --max-duration-seconds 300 --chunk-seconds 60
```

Results are saved as `outputs/youtube_VIDEO_ID_timestamps.txt` and `outputs/youtube_VIDEO_ID_result.json`. Use `--output-dir` to change the destination. Running the same video again overwrites those results. The JSON works with `generate_chapters.py` below.

The wrapper downloads audio to a temporary directory, runs the existing Whisper processor, and removes the download even if processing fails. The duration limit applies to transcription; the entire audio is still downloaded. `--plan-only` downloads the audio and prints the chunk plan without loading Whisper.

Supports individual watch, short, embed and completed livestream links. Playlist parameters are ignored when a video ID is present; playlist-only URLs and active livestreams are rejected. Private, restricted or unavailable videos may fail to download.

If YouTube downloads stop working, update the downloader:

```sh
python -m pip install -U "yt-dlp[default]"
```

## Process a Yandex Disk link

The same command accepts a public video or audio file link from Yandex Disk:

```sh
python process_url.py "https://disk.yandex.ru/i/PUBLIC_KEY"
python process_url.py "https://yadi.sk/d/PUBLIC_KEY" --max-duration-seconds 300
```

For a file inside a public folder, specify its path relative to the shared folder:

```sh
python process_url.py "https://disk.yandex.ru/d/PUBLIC_KEY" --disk-path "/Lectures/lecture.mp4"
```

Uses the [Yandex Disk REST API](https://yandex.ru/dev/disk/rest/) to read public resource metadata and obtain a download URL (`/v1/disk/public/resources/download`). Public downloads require no OAuth token or additional Python packages. Deno and yt-dlp are only needed for YouTube.

The entire source file is downloaded in chunks to temporary storage and removed after processing, including on failure. `--max-duration-seconds` limits transcription, not download size; `--plan-only` still downloads the file. Outputs are `outputs/yandex_<link-and-path-hash>_timestamps.txt` and `outputs/yandex_<link-and-path-hash>_result.json`. Repeating the same link and path overwrites these results.

Folders require `--disk-path`; non-media files are rejected. Private or password-protected links and files with downloading disabled are unsupported.

## Generate Chapters

`process_url.py` automatically generates topic chapters after Whisper for both YouTube and Yandex Disk. In addition to the transcript, it saves `<source>_chapters.txt` (chapter timestamps and titles) and `<source>_chapters.json` in the output directory. Repeating a run overwrites these files.

The chapter step uses `Qwen/Qwen3-4B-Instruct-2507`; about 8 GB of model weights are downloaded on first use. Override it with `--chapter-model`. Use `--skip-chapters` for transcription only. `--plan-only` skips both models. If chapter generation fails, the command reports an error and keeps the Whisper results so you can retry just this step. Generated titles and topic boundaries still need review.

To generate chapters separately from an existing transcript (including results from `process_data.py` or `run_whisper.py`):

```powershell
python generate_chapters.py result.json --output chapters.txt --json chapters.json
```

The default `detailed` mode reads the **entire transcript** in batches of up to eight minutes / 14,000 characters, loading the model once. Each batch gets several concrete topic headings, typically one per 2–4 minutes. There is no global ten-chapter cap and no discarded text. Timestamp IDs are checked against source segments; missing batch coverage, invalid JSON and vague short titles trigger retries. A failed batch fails the run without replacing existing chapter files. The JSON includes source segment IDs and text for review; `.generation.json` records the model responses.

Tune batch size with `--batch-seconds` and `--batch-max-chars`. The old `exact` and `compressed` single-prompt modes remain available for comparisons; `--min-chapters`, `--max-chapters` and `--min-gap-seconds` apply only to those legacy modes. Compressed mode discards part of the transcript and is unsuitable for detailed chapters of long lectures:

Repeating chapter generation with the same input, model and prompts reuses validated batches from `.generation.json`. Changes to the input, model or prompts invalidate that checkpoint. The web library notices updated chapter files without having to resubmit the video.

```powershell
python generate_chapters.py result.json --prompt-mode compressed --output chapters.txt --json chapters.json
```

For a cheap first check without loading the language model:

```powershell
python generate_chapters.py result.json --dry-run --prompt-output chapter_prompt.txt
```

## Roadmap

- Generate speech-based timestamps.
- Group transcript segments into meaningful video chapters.
- Export YouTube-ready chapter descriptions.
- Export subtitles in SRT/VTT formats.
- Add a simple UI for local video processing.
