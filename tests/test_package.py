"""Exercise the packaged entry points without downloading media or models."""

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from video_timestamps import chapters
from video_timestamps.web import create_app


class PackageTests(unittest.TestCase):
    def test_all_module_entry_points_from_another_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            for module in (
                "video_timestamps",
                "video_timestamps.pipeline",
                "video_timestamps.transcription",
                "video_timestamps.chapters",
                "video_timestamps.batch",
            ):
                with self.subTest(module=module):
                    result = subprocess.run(
                        [sys.executable, "-m", module, "--help"],
                        cwd=directory,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("usage:", result.stdout)

    def test_packaged_page_and_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            client = create_app(directory, start_worker=False).test_client()
            for url in ("/", "/static/app.css", "/static/app.js"):
                with self.subTest(url=url):
                    response = client.get(url)
                    self.assertEqual(response.status_code, 200)
                    self.assertGreater(len(response.data), 100)
                    response.close()

    def test_chapter_cli_writes_results_and_restores_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "transcript.json"
            source.write_text(
                json.dumps(
                    {
                        "chunks": [
                            {"timestamp": [0, 20], "text": "First precise lecture concept"},
                            {"timestamp": [500, 520], "text": "Final precise lecture concept"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output = root / "nested" / "chapters.txt"
            result = output.with_suffix(".json")
            argv = ["chapters", str(source), "--output", str(output), "--json", str(result)]
            with (
                patch("sys.argv", argv),
                patch.object(chapters, "load_chapter_model", return_value=(None, None)),
                patch.object(
                    chapters,
                    "generate_with_model",
                    side_effect=[
                        '[{"segment_id": 0, "title": "First precise lecture concept"}]',
                        '[{"segment_id": 1, "title": "Final precise lecture concept"}]',
                    ],
                ) as generate,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                chapters.main()
                self.assertEqual(generate.call_count, 2)
                generate.reset_mock()
                chapters.main()
                generate.assert_not_called()
            self.assertEqual(len(json.loads(result.read_text(encoding="utf-8"))), 2)
            self.assertIn("00:08:20", output.read_text(encoding="utf-8"))
            self.assertTrue(result.with_suffix(".generation.json").is_file())


if __name__ == "__main__":
    unittest.main()
