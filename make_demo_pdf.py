import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PDF_PATH = OUTPUT_DIR / "auto_video_timestamps_demo.pdf"
CHAPTERS_JSON = OUTPUT_DIR / "2026-02-11_lecture_z_function_chapters.json"
TRANSCRIPT_JSON = OUTPUT_DIR / "2026-02-11_lecture_z_function_result.json"
FONT_REGULAR = Path("C:/Windows/Fonts/DejaVuSans.ttf")
FONT_BOLD = Path("C:/Windows/Fonts/DejaVuSans-Bold.ttf")


def register_fonts() -> tuple[str, str]:
    regular_name = "DejaVuSans"
    bold_name = "DejaVuSansBold"
    pdfmetrics.registerFont(TTFont(regular_name, str(FONT_REGULAR)))
    pdfmetrics.registerFont(TTFont(bold_name, str(FONT_BOLD)))
    return regular_name, bold_name


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def hms(seconds: float) -> str:
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def build_styles(font_name: str, bold_font_name: str):
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=styles["Title"],
            fontName=bold_font_name,
            fontSize=22,
            leading=27,
            spaceAfter=8,
            textColor=colors.HexColor("#1f2937"),
            alignment=TA_LEFT,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=styles["Normal"],
            fontName=font_name,
            fontSize=11,
            leading=16,
            spaceAfter=14,
            textColor=colors.HexColor("#4b5563"),
        ),
        "h2": ParagraphStyle(
            "Heading2",
            parent=styles["Heading2"],
            fontName=bold_font_name,
            fontSize=14,
            leading=18,
            spaceBefore=12,
            spaceAfter=6,
            textColor=colors.HexColor("#111827"),
        ),
        "body": ParagraphStyle(
            "Body",
            parent=styles["BodyText"],
            fontName=font_name,
            fontSize=9.5,
            leading=14,
            spaceAfter=5,
            textColor=colors.HexColor("#111827"),
        ),
        "code": ParagraphStyle(
            "Code",
            parent=styles["Code"],
            fontName=font_name,
            fontSize=8,
            leading=11,
            leftIndent=8,
            rightIndent=8,
            spaceBefore=4,
            spaceAfter=8,
            backColor=colors.HexColor("#f3f4f6"),
        ),
    }


def add_bullets(story: list, items: list[str], style: ParagraphStyle) -> None:
    for item in items:
        story.append(Paragraph(f"- {item}", style))


def build_pdf() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    font_name, bold_font_name = register_fonts()
    styles = build_styles(font_name, bold_font_name)

    chapters = load_json(CHAPTERS_JSON)
    transcript = load_json(TRANSCRIPT_JSON)
    metadata = transcript["metadata"]

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title="auto-video-timestamps demo",
    )
    story = []

    story.append(Paragraph("auto-video-timestamps", styles["title"]))
    story.append(
        Paragraph(
            "Прототип локального пайплайна для автоматической транскрибации лекций "
            "и генерации смысловых таймкодов.",
            styles["subtitle"],
        )
    )

    story.append(Paragraph("Что уже сделано", styles["h2"]))
    add_bullets(
        story,
        [
            "Видео любой длительности режется на аудио-фрагменты с overlap.",
            "Whisper распознает речь и возвращает сегменты с абсолютными таймкодами.",
            "Qwen 2.5 3B Instruct группирует транскрипт в темы и пишет названия глав.",
            "Есть режим быстрого 5-минутного прототипа и batch-обработка папки data/.",
            "Исходные лекции лежат в data/, результаты сохраняются в outputs/.",
        ],
        styles["body"],
    )

    story.append(Paragraph("Схема пайплайна", styles["h2"]))
    story.append(
        Paragraph(
            "data/lecture.mp4 -> ffmpeg audio chunks -> Whisper ASR -> "
            "timestamped transcript JSON -> Qwen chaptering -> YouTube-ready chapters",
            styles["code"],
        )
    )

    story.append(Paragraph("Демо-лекция", styles["h2"]))
    metrics = [
        ["Файл", "2026-02-11 Лекция - Максим Бабенко - Z-функция.mp4"],
        ["Длительность", hms(metadata["duration_seconds"])],
        ["Whisper-сегментов", str(len(transcript["chunks"]))],
        ["Глав после LLM", str(len(chapters))],
        ["Модель ASR", metadata["model"]],
        ["Модель глав", "Qwen/Qwen2.5-3B-Instruct"],
    ]
    table = Table(metrics, colWidths=[45 * mm, 120 * mm])
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("FONTNAME", (0, 0), (0, -1), bold_font_name),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("LEADING", (0, 0), (-1, -1), 12),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f3f4f6")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(table)

    story.append(Paragraph("Пример сгенерированных глав", styles["h2"]))
    chapter_rows = [["Таймкод", "Тема"]]
    for chapter in chapters:
        chapter_rows.append([chapter["start"], chapter["title"]])
    chapter_table = Table(chapter_rows, colWidths=[28 * mm, 137 * mm])
    chapter_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("FONTNAME", (0, 0), (-1, 0), bold_font_name),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("LEADING", (0, 0), (-1, -1), 12),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(chapter_table)

    story.append(Paragraph("Команды для демонстрации", styles["h2"]))
    story.append(
        Paragraph(
            "python process_data.py<br/>"
            "python generate_chapters.py outputs/2026-02-11_lecture_z_function_result.json "
            "--output outputs/2026-02-11_lecture_z_function_chapters.txt "
            "--json outputs/2026-02-11_lecture_z_function_chapters.json",
            styles["code"],
        )
    )

    story.append(Paragraph("Следующий шаг", styles["h2"]))
    add_bullets(
        story,
        [
            "Добавить единый batch-режим: транскрибация и главы одной командой.",
            "Сделать второй LLM-проход для редакторского улучшения названий.",
            "Добавить экспорт SRT/VTT и YouTube description.",
        ],
        styles["body"],
    )

    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            "Статус: рабочий локальный прототип. Все модели запускаются локально, "
            "исходные видео и результаты не попадают в Git.",
            styles["subtitle"],
        )
    )

    doc.build(story)
    print(PDF_PATH)


if __name__ == "__main__":
    build_pdf()
