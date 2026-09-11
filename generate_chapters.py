import argparse
import json
import re
from pathlib import Path


DEFAULT_CHAPTER_MODEL = "Qwen/Qwen2.5-3B-Instruct"
GENERIC_TITLE_KEYWORDS = (
    "продолжение",
    "работы",
    "детали",
    "ресурсы",
    "примеры",
    "использования",
    "обработка",
    "мусора",
)
TECHNICAL_TITLE_KEYWORDS = (
    "z-функц",
    "z функц",
    "lcp",
    "алгоритм",
    "подстрок",
    "префикс",
    "суффикс",
    "строк",
    "блок",
    "метрик",
    "проход",
)


def format_timestamp(seconds: float | None) -> str:
    if seconds is None:
        return "00:00:00"

    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def parse_timestamp(value: str) -> int | None:
    match = re.fullmatch(r"(\d{2}):(\d{2}):(\d{2})", str(value).strip())
    if not match:
        return None

    hours, minutes, seconds = map(int, match.groups())
    return hours * 3600 + minutes * 60 + seconds


def load_whisper_chunks(input_path: Path) -> list[dict]:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    chunks = data.get("chunks")
    if not isinstance(chunks, list):
        raise ValueError(f"Expected Whisper JSON with a 'chunks' list: {input_path}")
    return chunks


def chunk_start_seconds(chunk: dict) -> float:
    timestamp = chunk.get("timestamp") or [0, None]
    start = timestamp[0]
    return float(start or 0)


def build_topic_windows(
    chunks: list[dict],
    window_seconds: int,
    max_window_chars: int,
) -> list[dict]:
    if window_seconds <= 0:
        raise ValueError("--window-seconds must be greater than 0")
    if max_window_chars <= 0:
        raise ValueError("--max-window-chars must be greater than 0")

    windows = []
    current_start = None
    current_text = []

    for chunk in chunks:
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue

        start = chunk_start_seconds(chunk)
        if current_start is None:
            current_start = start

        if start - current_start >= window_seconds and current_text:
            windows.append(
                {
                    "start": current_start,
                    "text": " ".join(current_text)[:max_window_chars],
                }
            )
            current_start = start
            current_text = []

        current_text.append(text)

    if current_start is not None and current_text:
        windows.append(
            {
                "start": current_start,
                "text": " ".join(current_text)[:max_window_chars],
            }
        )

    return windows


def build_prompt(windows: list[dict], min_chapters: int, max_chapters: int) -> str:
    transcript = "\n".join(
        f"[{format_timestamp(window['start'])}] {window['text']}"
        for window in windows
    )
    return f"""Ты помогаешь делать качественные таймкоды для длинных видео и лекций.

Задача:
- Разбей транскрипт на смысловые главы.
- Не делай главу на каждую мелкую фразу.
- Названия должны быть короткими, конкретными и полезными зрителю.
- Если это математическая лекция, сохраняй термины, определения, теоремы, примеры и переходы между темами.
- Избегай общих названий вроде "продолжение работы", "детали алгоритма", "примеры использования", если можно назвать конкретный математический объект или шаг.
- Используй только таймкоды, которые есть в транскрипте.
- Верни не больше {max_chapters} глав.
- Верни только JSON-массив без Markdown.

Формат:
[
  {{
    "start": "00:00:00",
    "title": "Короткое название темы",
    "summary": "Одна фраза о том, что происходит в этом фрагменте."
  }}
]

Количество глав: от {min_chapters} до {max_chapters}.

Транскрипт:
{transcript}
"""


def load_chapter_model(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    return tokenizer, model


def generate_text(model_name: str, prompt: str, max_new_tokens: int) -> str:
    tokenizer, model = load_chapter_model(model_name)
    messages = [
        {
            "role": "system",
            "content": "Ты аккуратный редактор образовательных видео. Отвечай строго валидным JSON.",
        },
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    generated = outputs[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def extract_json_array(text: str) -> list[dict]:
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        raise ValueError("Model output does not contain a JSON array.")
    chapters = json.loads(match.group(0))
    if not isinstance(chapters, list):
        raise ValueError("Model output JSON must be an array.")
    return chapters


def title_score(title: str) -> int:
    normalized = title.casefold()
    score = 0
    for keyword in TECHNICAL_TITLE_KEYWORDS:
        if keyword in normalized:
            score += 3
    for keyword in GENERIC_TITLE_KEYWORDS:
        if keyword in normalized:
            score -= 1
    if 12 <= len(title) <= 60:
        score += 1
    return score


def normalize_chapters(
    chapters: list[dict],
    min_gap_seconds: int,
    max_chapters: int,
) -> list[dict]:
    if max_chapters <= 0:
        raise ValueError("--max-chapters must be greater than 0")
    if min_gap_seconds < 0:
        raise ValueError("--min-gap-seconds must be 0 or greater")

    prepared = []
    for chapter in chapters:
        start = parse_timestamp(chapter.get("start", ""))
        title = str(chapter.get("title", "")).strip()
        summary = str(chapter.get("summary", "")).strip()
        if start is None or not title:
            continue

        prepared.append(
            {
                "start_seconds": start,
                "start": format_timestamp(start),
                "title": title,
                "summary": summary,
                "score": title_score(title),
            }
        )

    prepared.sort(key=lambda chapter: chapter["start_seconds"])
    deduped = []
    for chapter in prepared:
        if not deduped:
            deduped.append(chapter)
            continue

        previous = deduped[-1]
        if chapter["start_seconds"] - previous["start_seconds"] < min_gap_seconds:
            if chapter["score"] > previous["score"]:
                deduped[-1] = chapter
            continue

        deduped.append(chapter)

    while len(deduped) > max_chapters:
        removable = min(
            range(1, len(deduped)),
            key=lambda index: (
                deduped[index]["score"],
                deduped[index]["start_seconds"] - deduped[index - 1]["start_seconds"],
            ),
        )
        deduped.pop(removable)

    return [
        {
            key: chapter[key]
            for key in ("start", "title", "summary")
            if chapter.get(key)
        }
        for chapter in deduped
    ]


def render_youtube_chapters(chapters: list[dict]) -> str:
    lines = []
    for chapter in chapters:
        start = chapter.get("start", "00:00:00")
        title = str(chapter.get("title", "")).strip()
        if title:
            lines.append(f"{start} {title}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate topic chapters from Whisper timestamp JSON."
    )
    parser.add_argument("input_json", type=Path, help="Path to Whisper result JSON.")
    parser.add_argument(
        "--model",
        default=DEFAULT_CHAPTER_MODEL,
        help=f"Text-generation model. Default: {DEFAULT_CHAPTER_MODEL}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("chapters.txt"),
        help="Path for YouTube-style chapters. Default: chapters.txt.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("chapters.json"),
        help="Path for structured chapter JSON. Default: chapters.json.",
    )
    parser.add_argument(
        "--prompt-output",
        type=Path,
        help="Save the prompt sent to the chapter model.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the prompt but do not load the chapter model.",
    )
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=180,
        help="Transcript compression window in seconds. Default: 180.",
    )
    parser.add_argument(
        "--max-window-chars",
        type=int,
        default=900,
        help="Maximum transcript characters per window. Default: 900.",
    )
    parser.add_argument("--min-chapters", type=int, default=4)
    parser.add_argument("--max-chapters", type=int, default=10)
    parser.add_argument(
        "--min-gap-seconds",
        type=int,
        default=180,
        help="Minimum distance between final chapters. Default: 180.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    args = parser.parse_args()

    chunks = load_whisper_chunks(args.input_json)
    windows = build_topic_windows(chunks, args.window_seconds, args.max_window_chars)
    prompt = build_prompt(windows, args.min_chapters, args.max_chapters)

    if args.prompt_output:
        args.prompt_output.write_text(prompt, encoding="utf-8")

    if args.dry_run:
        print(prompt)
        return

    model_output = generate_text(args.model, prompt, args.max_new_tokens)
    chapters = normalize_chapters(
        extract_json_array(model_output),
        args.min_gap_seconds,
        args.max_chapters,
    )

    args.json.write_text(
        json.dumps(chapters, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args.output.write_text(render_youtube_chapters(chapters), encoding="utf-8")


if __name__ == "__main__":
    main()
