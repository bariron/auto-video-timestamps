import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from video_timestamps import pipeline


class ProcessUrlTests(unittest.TestCase):
    def test_supported_links_ignore_playlist_and_time(self):
        for url in [
            "https://youtu.be/abcdefghijk?t=30",
            "https://www.youtube.com/watch?v=abcdefghijk&list=playlist",
            "https://m.youtube.com/shorts/abcdefghijk",
            "https://youtube.com/live/abcdefghijk",
        ]:
            with self.subTest(url=url):
                self.assertEqual(
                    pipeline.normalize_youtube_url(url),
                    "https://www.youtube.com/watch?v=abcdefghijk",
                )

    def test_rejects_other_sources_and_playlist_only(self):
        for url in [
            "https://youtube.com.evil.org/watch?v=abcdefghijk",
            "https://youtube.com/playlist?list=abc",
            "file:///tmp/video.mp4",
            "https://disk.yandex.ru/i/abcdefghijk",
        ]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                pipeline.normalize_youtube_url(url)

    def test_live_filter(self):
        self.assertIsNotNone(pipeline.reject_live({"is_live": True}))
        self.assertIsNotNone(pipeline.reject_live({"live_status": "is_upcoming"}))
        self.assertIsNone(pipeline.reject_live({"live_status": "was_live"}))

    def test_youtube_runtime_discovery_and_missing_runtime_error(self):
        with (
            patch.object(pipeline.shutil, "which", return_value=None),
            patch.object(Path, "is_file", return_value=False),
        ):
            with self.assertRaisesRegex(RuntimeError, "Deno"):
                pipeline.youtube_js_runtimes()
        with (
            patch.object(pipeline.shutil, "which", return_value=None),
            patch.object(Path, "is_file", side_effect=lambda: True),
        ):
            self.assertIn("deno", pipeline.youtube_js_runtimes())

    def test_youtube_download_options_and_network_error(self):
        from yt_dlp.utils import DownloadError

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                pipeline, "youtube_js_runtimes", return_value={"deno": {"path": "deno.exe"}}
            ),
            patch("yt_dlp.YoutubeDL") as downloader,
        ):
            downloader.return_value.__enter__.return_value.extract_info.side_effect = DownloadError(
                "Read timed out"
            )
            with self.assertRaisesRegex(RuntimeError, "сетевое соединение"):
                pipeline.download_youtube_audio("https://youtu.be/abcdefghijk", Path(directory))
            options = downloader.call_args.args[0]
            self.assertEqual(options["socket_timeout"], 60)
            self.assertEqual(options["retries"], 5)
            self.assertEqual(options["js_runtimes"]["deno"]["path"], "deno.exe")

    def test_pipeline_passes_options_and_cleans_audio_on_success_and_failure(self):
        for fail in [False, True]:
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as output:
                paths = []

                def download(url, directory):
                    path = directory / "audio.webm"
                    path.write_bytes(b"audio")
                    paths.append(path)
                    return path

                def run(command, check):
                    self.assertTrue(paths[0].exists())
                    self.assertTrue(check)
                    self.assertEqual(command[command.index("--max-duration-seconds") + 1], "300")
                    self.assertIn(str(Path(output) / "youtube_abcdefghijk_result.json"), command)
                    if fail:
                        raise subprocess.CalledProcessError(2, command)

                argv = [
                    "pipeline.py",
                    "https://youtu.be/abcdefghijk",
                    "--output-dir",
                    output,
                    "--max-duration-seconds",
                    "300",
                    "--skip-chapters",
                ]
                with (
                    patch("sys.argv", argv),
                    patch.object(pipeline, "download_youtube_audio", download),
                    patch.object(pipeline.subprocess, "run", run),
                ):
                    if fail:
                        with self.assertRaises(SystemExit) as error:
                            pipeline.main()
                        self.assertEqual(error.exception.code, 2)
                    else:
                        pipeline.main()
                self.assertFalse(paths[0].parent.exists())

    def test_chapters_follow_transcription_for_both_sources(self):
        for url in ["https://youtu.be/abcdefghijk", "https://disk.yandex.ru/d/key"]:
            for mode in ["success", "whisper_failure", "chapter_failure", "plan", "skip"]:
                with self.subTest(url=url, mode=mode), tempfile.TemporaryDirectory() as output:
                    calls = []
                    paths = []

                    def download(url, directory, *args):
                        media = directory / "media.mp4"
                        media.write_bytes(b"video")
                        paths.append(media)
                        return media

                    def run(command, check):
                        calls.append(command)
                        if len(calls) == 1:
                            self.assertTrue(paths[0].exists())
                            if mode == "whisper_failure":
                                raise subprocess.CalledProcessError(2, command)
                            if mode != "plan":
                                Path(command[command.index("--json") + 1]).write_text(
                                    '{"chunks": []}'
                                )
                        else:
                            self.assertFalse(paths[0].exists())
                            self.assertEqual(command[1:3], ["-m", "video_timestamps.chapters"])
                            self.assertEqual(command[3], calls[0][calls[0].index("--json") + 1])
                            self.assertTrue(Path(command[3]).exists())
                            self.assertEqual(command[command.index("--model") + 1], "test-model")
                            if mode == "chapter_failure":
                                raise subprocess.CalledProcessError(3, command)

                    argv = [
                        "pipeline.py",
                        url,
                        "--output-dir",
                        output,
                        "--chapter-model",
                        "test-model",
                    ]
                    argv += {"plan": ["--plan-only"], "skip": ["--skip-chapters"]}.get(mode, [])
                    with (
                        patch("sys.argv", argv),
                        patch.object(pipeline, "download_youtube_audio", download),
                        patch.object(pipeline.yandex_disk, "download_public_media", download),
                        patch.object(pipeline.subprocess, "run", run),
                        patch("sys.stderr", new_callable=io.StringIO) as stderr,
                    ):
                        if mode.endswith("failure"):
                            with self.assertRaises(SystemExit) as error:
                                pipeline.main()
                            self.assertEqual(
                                error.exception.code, 2 if mode == "whisper_failure" else 3
                            )
                            if mode == "chapter_failure":
                                self.assertIn("Whisper results are saved", stderr.getvalue())
                                self.assertTrue(Path(calls[1][3]).exists())
                        else:
                            pipeline.main()
                    self.assertEqual(len(calls), 2 if mode in {"success", "chapter_failure"} else 1)
                    self.assertFalse(paths[0].parent.exists())


if __name__ == "__main__":
    unittest.main()
