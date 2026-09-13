"""Resolve and download a public Yandex Disk media file via REST API."""

import json
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlencode, urlparse
from urllib.request import urlopen


HOSTS = {"disk.yandex.ru", "disk.yandex.com", "disk.yandex.by",
         "disk.yandex.kz", "disk.yandex.com.tr", "yadi.sk"}
API = "https://cloud-api.yandex.net/v1/disk/public/resources"
MEDIA_EXTENSIONS = {".avi", ".m4a", ".mkv", ".mov", ".mp3", ".mp4", ".wav",
                    ".webm", ".flac", ".ogg", ".opus", ".aac", ".m4v", ".wma"}


def split_public_url(value: str, resource_path: str | None = None) -> tuple[str, str | None]:
    """Accept a shared root URL or a browser URL to a file within it."""
    parsed = urlparse(value.strip())
    parts = parsed.path.split("/", 3)
    if len(parts) == 4 and parts[3]:
        embedded_path = "/" + unquote(parts[3])
        if resource_path and resource_path != embedded_path:
            raise ValueError("The URL path and --disk-path refer to different files.")
        resource_path = embedded_path
        parsed = parsed._replace(path="/".join(parts[:3]))
    if resource_path and not resource_path.startswith("/"):
        raise ValueError("Yandex Disk file path must start with '/'.")
    return normalize_public_url(parsed.geturl()), resource_path


def normalize_public_url(value: str) -> str:
    parsed = urlparse(value.strip())
    if (parsed.scheme not in {"http", "https"} or parsed.hostname not in HOSTS
            or parsed.username or parsed.password or parsed.port is not None
            or not re.fullmatch(r"/[di]/[A-Za-z0-9_-]+/?", parsed.path)):
        raise ValueError("Expected a public Yandex Disk file link, such as https://disk.yandex.ru/i/KEY.")
    return f"https://{parsed.hostname}{parsed.path.rstrip('/')}"


def api_request(endpoint: str, params: dict) -> dict:
    with urlopen(f"{API}{endpoint}?{urlencode(params)}", timeout=30) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise RuntimeError("Yandex Disk returned an invalid API response.")
    return result


def download_public_media(url: str, directory: Path, resource_path: str | None = None) -> Path:
    params = {"public_key": normalize_public_url(url)}
    if resource_path:
        params["path"] = resource_path
    try:
        metadata = api_request("", params)
        if metadata.get("type") == "dir":
            raise ValueError("This link points to a folder. Select a file with --disk-path '/video.mp4'.")
        if metadata.get("type") != "file":
            raise RuntimeError("Yandex Disk did not return a file.")
        suffix = Path(str(metadata.get("name", ""))).suffix.lower()
        mime = str(metadata.get("mime_type", ""))
        if suffix not in MEDIA_EXTENSIONS and not mime.startswith(("video/", "audio/")):
            raise ValueError("The selected Yandex Disk resource is not a video or audio file.")
        link = api_request("/download", params)
        href = link.get("href")
        if (not isinstance(href, str) or urlparse(href).scheme != "https"
                or not urlparse(href).hostname or link.get("method", "GET") != "GET"):
            raise RuntimeError("Yandex Disk returned an invalid download link.")
        # Never use the remote filename as a local path.
        target = directory / ("media" + (suffix if suffix in MEDIA_EXTENSIONS else ".bin"))
        size = 0
        with urlopen(href, timeout=60) as response, target.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                size += len(chunk)
        expected_size = metadata.get("size")
        if not size or (isinstance(expected_size, int) and size != expected_size):
            raise RuntimeError("Yandex Disk download is empty or incomplete; retry the download.")
        return target
    except HTTPError as exc:
        hints = {403: "Access or downloading is forbidden.", 404: "Public file not found.",
                 429: "Too many requests; try again later."}
        raise RuntimeError(f"Yandex Disk HTTP {exc.code}: "
                           + hints.get(exc.code, "Could not download the public file.")) from exc
    except URLError as exc:
        raise RuntimeError(f"Yandex Disk connection failed: {exc.reason}") from exc
