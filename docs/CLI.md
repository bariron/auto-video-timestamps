# Командная строка

Все команды выполняются в окружении, где установлен пакет (`python -m pip install -e .`). Для каждой команды доступен `--help`.

## Полная обработка ссылки

```sh
video-timestamps-url "https://www.youtube.com/watch?v=VIDEO_ID"
video-timestamps-url "https://disk.yandex.ru/d/PUBLIC_KEY" --disk-path "/lecture.mp4"
```

Эквивалент без консольного ярлыка: `python -m video_timestamps.pipeline URL`.

По умолчанию результат попадает в `outputs/`. Можно задать `--output-dir PATH`, `--model NAME` для Whisper или `--chapter-model NAME` для Qwen. `--skip-chapters` оставляет только расшифровку. Чтобы повторить только генерацию глав после сбоя, используйте сохранённый JSON расшифровки в отдельной команде ниже.

## Локальный файл

```sh
video-timestamps-transcribe data/lecture.mp4 --output outputs/lecture_timestamps.txt --json outputs/lecture_result.json
video-timestamps-chapters outputs/lecture_result.json --output outputs/lecture_chapters.txt --json outputs/lecture_chapters.json
```

Эти же команды доступны как `python -m video_timestamps.transcription` и `python -m video_timestamps.chapters`.

Распознавание делит запись на части по 600 секунд с перекрытием 5 секунд. Настройки: `--chunk-seconds` и `--overlap-seconds`. Для диагностики доступны `--plan-only` (план без загрузки Whisper) и `--max-duration-seconds N` (ограничение длительности). Веб-приложение обрабатывает запись полностью.

Для просмотра промптов без запуска Qwen:

```sh
video-timestamps-chapters outputs/lecture_result.json --dry-run --prompt-output outputs/lecture_prompt.txt
```

Генератор использует весь текст, разбивая его на порции. Настройки: `--batch-seconds`, `--batch-max-chars`, `--max-new-tokens`. Рядом с JSON глав сохраняется файл `.generation.json` с ответами модели для восстановления после сбоя.

## Папка локальных записей

```sh
video-timestamps-batch --help
python -m video_timestamps.batch
```

Пакетная команда распознаёт файлы из `data/` и сохраняет расшифровки в `outputs/`. Главы для нужной расшифровки запускаются отдельной командой выше.
