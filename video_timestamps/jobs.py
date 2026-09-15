"""Persistent jobs, background processing, and result discovery."""

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from .paths import PROJECT_ROOT as ROOT
from .pipeline import normalize_youtube_url
from .sources import yandex_disk

ARTIFACTS = {
    "chapters": "_chapters.txt",
    "chapters_json": "_chapters.json",
    "transcript": "_timestamps.txt",
    "result_json": "_result.json",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def validate_submission(data):
    if not isinstance(data, dict):
        raise ValueError("Ожидается ссылка на видео.")
    url = data.get("url", "")
    disk_path = data.get("disk_path", "")
    if not isinstance(url, str) or not isinstance(disk_path, str) or len(url) > 8192:
        raise ValueError("Проверьте ссылку и путь к файлу.")
    disk_path = disk_path.strip() or None
    if urlparse(url.strip()).hostname in yandex_disk.HOSTS:
        try:
            url, disk_path = yandex_disk.split_public_url(url, disk_path)
        except ValueError as exc:
            raise ValueError(
                "Проверьте публичную ссылку Яндекс Диска. Путь должен начинаться с / и совпадать с файлом в ссылке."
            ) from exc
        provider = "Яндекс Диск"
    else:
        try:
            url = normalize_youtube_url(url)
        except ValueError as exc:
            raise ValueError(
                "Вставьте ссылку на отдельное видео YouTube или публичный файл Яндекс Диска."
            ) from exc
        if disk_path:
            raise ValueError("Путь внутри папки используется только для Яндекс Диска.")
        provider = "YouTube"
    title = Path(disk_path).name if disk_path else url.rsplit("/", 1)[-1]
    return dict(url=url, disk_path=disk_path, title=title, provider=provider)


class JobQueue:
    """One persistent queue and one worker per local server process."""

    def __init__(self, output_dir, start_worker=True):
        self.output_dir = Path(output_dir).resolve()
        self.jobs_dir = self.output_dir / "web_jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.condition = threading.Condition()
        self.jobs = {}
        for path in self.jobs_dir.glob("*/job.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
                if job["id"] != path.parent.name:
                    continue
                if job["status"] == "running":
                    job.update(
                        status="failed",
                        stage="Прервано",
                        error="Сервер был остановлен во время обработки. Сохранённые результаты доступны в истории.",
                    )
                    self._save(job)
                self.jobs[job["id"]] = job
            except (OSError, ValueError, KeyError):
                continue
        if start_worker:
            threading.Thread(target=self._worker, daemon=True, name="video-worker").start()

    def _save(self, job):
        path = self.jobs_dir / job["id"] / "job.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)

    def update(self, job_id, **values):
        with self.condition:
            self.jobs[job_id].update(values)
            self._save(self.jobs[job_id])

    def submit(self, data):
        values = validate_submission(data)
        with self.condition:
            if sum(j["status"] in {"queued", "running"} for j in self.jobs.values()) >= 20:
                raise ValueError("В очереди уже 20 видео. Дождитесь завершения обработки.")
            job = dict(
                id=uuid.uuid4().hex,
                created=now(),
                status="queued",
                stage="В очереди",
                error=None,
                **values,
            )
            self._save(job)
            self.jobs[job["id"]] = job
            self.condition.notify()
            return dict(job)

    def list_jobs(self):
        with self.condition:
            return [
                dict(j)
                for j in sorted(self.jobs.values(), key=lambda j: j["created"], reverse=True)
            ]

    def _worker(self):
        while True:
            with self.condition:
                pending = [j for j in self.jobs.values() if j["status"] == "queued"]
                if not pending:
                    self.condition.wait()
                    continue
                job = min(pending, key=lambda j: j["created"])
                job.update(status="running", stage="Скачивание видео")
                self._save(job)
            self.run_job(dict(job))

    def run_job(self, job):
        directory = self.jobs_dir / job["id"]
        command = [
            sys.executable,
            "-u",
            "-m",
            "video_timestamps.pipeline",
            job["url"],
            "--output-dir",
            str(directory),
        ]
        if job["disk_path"]:
            command += ["--disk-path", job["disk_path"]]
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        failure_message = None
        try:
            with (directory / "run.log").open("w", encoding="utf-8") as log:
                with subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                ) as proc:
                    for line in proc.stdout:
                        log.write(line)
                        log.flush()
                        if line.startswith("Error:"):
                            failure_message = line.removeprefix("Error:").strip()
                        if "Transcribing with Whisper" in line:
                            self.update(job["id"], stage="Распознавание речи")
                        elif match := re.search(r"Processing chunk (\d+)/(\d+)", line):
                            self.update(
                                job["id"],
                                stage=f"Распознавание речи · фрагмент {match[1]} из {match[2]}",
                            )
                        elif "Generating topic chapters" in line:
                            self.update(job["id"], stage="Создание смысловых глав")
                        elif match := re.search(r"Chapter batch (\d+)/(\d+)", line):
                            self.update(
                                job["id"], stage=f"Создание глав · блок {match[1]} из {match[2]}"
                            )
                    code = proc.wait()
            if code:
                raise RuntimeError(
                    failure_message
                    or f"Обработка завершилась с ошибкой (код {code}). Подробности — в журнале. Готовая расшифровка, если она есть, сохранена."
                )
            if not any(
                path.read_text(encoding="utf-8").strip()
                for path in directory.glob("*_chapters.txt")
            ):
                raise RuntimeError(
                    "Генератор не создал файл с главами. Проверьте журнал обработки."
                )
            self.update(job["id"], status="done", stage="Готово", finished=now())
        except Exception as exc:
            self.update(
                job["id"], status="failed", stage="Ошибка обработки", error=str(exc), finished=now()
            )

    def results(self):
        results = []
        jobs = {j["id"]: j for j in self.list_jobs()}
        for path in self.output_dir.rglob("*_result.json"):
            if not path.resolve().is_relative_to(self.output_dir):
                continue
            base = path.name.removesuffix("_result.json")
            job = jobs.get(path.parent.name, {})
            relative = path.relative_to(self.output_dir).as_posix()
            result_id = hashlib.sha256(relative.encode()).hexdigest()[:24]
            files = {
                kind: path.with_name(base + suffix)
                for kind, suffix in ARTIFACTS.items()
                if path.with_name(base + suffix).is_file()
                and path.with_name(base + suffix).resolve().is_relative_to(self.output_dir)
            }
            results.append(
                dict(
                    id=result_id,
                    title=job.get("title", base),
                    job_id=job.get("id"),
                    created=job.get(
                        "created",
                        datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                    ),
                    preview=job.get("preview", False),
                    updated=max(p.stat().st_mtime_ns for p in files.values()),
                    files=files,
                )
            )
        return sorted(results, key=lambda r: r["created"], reverse=True)
