import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import process_url


class ProcessUrlTests(unittest.TestCase):
    def test_supported_links_ignore_playlist_and_time(self):
        for url in [
            "https://youtu.be/abcdefghijk?t=30",
            "https://www.youtube.com/watch?v=abcdefghijk&list=playlist",
            "https://m.youtube.com/shorts/abcdefghijk",
            "https://youtube.com/live/abcdefghijk",
        ]:
            with self.subTest(url=url):
                self.assertEqual(process_url.normalize_youtube_url(url),
                                 "https://www.youtube.com/watch?v=abcdefghijk")

    def test_rejects_other_sources_and_playlist_only(self):
        for url in ["https://youtube.com.evil.org/watch?v=abcdefghijk",
                    "https://youtube.com/playlist?list=abc", "file:///tmp/video.mp4",
                    "https://disk.yandex.ru/i/abcdefghijk"]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                process_url.normalize_youtube_url(url)

    def test_live_filter(self):
        self.assertIsNotNone(process_url.reject_live({"is_live": True}))
        self.assertIsNotNone(process_url.reject_live({"live_status": "is_upcoming"}))
        self.assertIsNone(process_url.reject_live({"live_status": "was_live"}))

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

                argv = ["process_url.py", "https://youtu.be/abcdefghijk",
                        "--output-dir", output, "--max-duration-seconds", "300"]
                with patch("sys.argv", argv), patch.object(process_url, "download_youtube_audio", download), \
                        patch.object(process_url.subprocess, "run", run):
                    if fail:
                        with self.assertRaises(SystemExit) as error:
                            process_url.main()
                        self.assertEqual(error.exception.code, 2)
                    else:
                        process_url.main()
                self.assertFalse(paths[0].parent.exists())


if __name__ == "__main__":
    unittest.main()
