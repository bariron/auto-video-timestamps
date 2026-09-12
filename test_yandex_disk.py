import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

import yandex_disk
import process_url


class YandexDiskTests(unittest.TestCase):
    def test_public_urls(self):
        for host in ["disk.yandex.ru", "disk.yandex.com", "yadi.sk"]:
            self.assertEqual(yandex_disk.normalize_public_url(f"http://{host}/d/abc-123/?x=1"),
                             f"https://{host}/d/abc-123")
        for url in ["https://disk.yandex.ru.evil.org/i/abc", "file:///i/abc",
                    "https://user@disk.yandex.ru/i/abc", "https://disk.yandex.ru/client/disk"]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                yandex_disk.normalize_public_url(url)

    def test_api_download_and_encoded_folder_path(self):
        responses = [io.BytesIO(json.dumps({"type": "file", "name": "../../video.mp4", "size": 5}).encode()),
                     io.BytesIO(b'{"href":"https://download.example/media", "method":"GET"}'),
                     io.BytesIO(b"video")]
        with tempfile.TemporaryDirectory() as directory, patch.object(yandex_disk, "urlopen", side_effect=responses) as request:
            result = yandex_disk.download_public_media("https://disk.yandex.ru/d/key", Path(directory), "/Лекция 1.mp4")
            self.assertEqual(result, Path(directory) / "media.mp4")
            self.assertEqual(result.read_bytes(), b"video")
            for call in request.call_args_list[:2]:
                params = parse_qs(urlparse(call.args[0]).query)
                self.assertEqual(params["path"], ["/Лекция 1.mp4"])
                self.assertEqual(params["public_key"], ["https://disk.yandex.ru/d/key"])

    def test_rejects_folders_and_non_media_before_download(self):
        for metadata in [{"type": "dir"}, {"type": "file", "name": "report.pdf"}]:
            with patch.object(yandex_disk, "api_request", return_value=metadata) as api:
                with self.assertRaises(ValueError):
                    yandex_disk.download_public_media("https://yadi.sk/d/key", Path("unused"))
                self.assertEqual(api.call_count, 1)

    def test_rejects_incomplete_download(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(yandex_disk, "api_request", side_effect=[
                    {"type": "file", "name": "video.mp4", "size": 100},
                    {"href": "https://download.example/media"}]), \
                patch.object(yandex_disk, "urlopen", return_value=io.BytesIO(b"short")):
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                yandex_disk.download_public_media("https://yadi.sk/d/key", Path(directory))

    def test_http_error_is_readable(self):
        with patch.object(yandex_disk, "urlopen", side_effect=HTTPError("url", 404, "missing", {}, None)):
            with self.assertRaisesRegex(RuntimeError, "Public file not found"):
                yandex_disk.download_public_media("https://yadi.sk/d/key", Path("unused"))

    def test_cli_routes_disk_and_cleans_temporary_file(self):
        paths = []

        def download(url, directory, resource_path):
            self.assertEqual(resource_path, "/lecture.mp4")
            media = directory / "media.mp4"
            media.write_bytes(b"video")
            paths.append(media)
            return media

        with tempfile.TemporaryDirectory() as output, \
                patch("sys.argv", ["process_url.py", "https://yadi.sk/d/key", "--disk-path", "/lecture.mp4",
                                   "--output-dir", output, "--plan-only"]), \
                patch.object(yandex_disk, "download_public_media", side_effect=download), \
                patch.object(process_url, "download_youtube_audio") as youtube, \
                patch.object(process_url.subprocess, "run") as run:
            process_url.main()
            youtube.assert_not_called()
            self.assertIn("--plan-only", run.call_args.args[0])
            self.assertFalse(paths[0].parent.exists())


if __name__ == "__main__":
    unittest.main()
