# Auto Video Timestamps

Локальное веб-приложение, которое превращает длинные видео в расшифровку и подробные главы с таймкодами.

Вставьте ссылку на **YouTube** или **Яндекс Диск** — приложение скачает запись, распознает речь через Whisper и создаст главы через Qwen. Результаты можно просматривать, копировать и скачивать; ход обработки виден в очереди и журнале.

## Запуск

Нужен **Python 3.10+**. Рекомендуется NVIDIA GPU с CUDA-сборкой PyTorch. Для YouTube также нужен **Deno** или **Node.js** в PATH. При первом запуске модели скачиваются с Hugging Face.

```sh
git clone https://github.com/bariron/auto-video-timestamps.git
cd auto-video-timestamps
```

На Windows:

```powershell
.\scripts\setup.ps1
.\scripts\start.ps1
```

В готовом Python-окружении, в том числе на Linux и macOS:

```sh
python -m pip install -e .
python -m video_timestamps
```

Откройте **http://127.0.0.1:7860**. Распознавание и генерация выполняются на вашем компьютере.

## Файлы проекта

| Папка | Содержимое |
| --- | --- |
| `video_timestamps/` | Код приложения, HTML, CSS и JavaScript |
| `scripts/` | Установка, запуск и проверки |
| `tests/` | Автоматические тесты |
| `data/` | Локальные исходные записи |
| `outputs/` | Расшифровки, главы и история обработки |

Записи и результаты не попадают в Git. Для другой папки данных задайте `VIDEO_TIMESTAMPS_HOME`. Запускайте один сервер на папку результатов.

Обработка локальных файлов и отдельные этапы описаны в [справке по CLI](docs/CLI.md).

## Разработка

```sh
python -m pip install -e ".[dev]"
python -m ruff check video_timestamps tests
python -m ruff format --check video_timestamps tests
python -m unittest discover -s tests -v
```

На Windows все проверки также запускаются командой `.\scripts\test.ps1`.
