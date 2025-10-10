"""Gradio interface for interactive MedCAT entity extraction."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import html
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr

try:
    # Package imports (python -m src.gradio_app)
    from .extractor import extract_entities
    from .utils import load_model_pack_auto
except ImportError:  # pragma: no cover - fallback when run as a script
    from extractor import extract_entities
    from utils import load_model_pack_auto

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
CLUSTER_MAP: Dict[str, str] | None = None

PREFERRED_MODEL = "IEE_MedCAT_v1"
SAMPLE_TEXTS: dict[str, str] = {
    "Приклад 1": (
        "The patient presented with long-standing type 2 diabetes mellitus and hypertension. "
        "Laboratory values demonstrated elevated HbA1c, persistent hyperglycemia, and intermittent "
        "hypokalemia requiring supplementation. Despite metformin therapy, fasting glucose remained high, "
        "so endocrinology recommended adding basal insulin and a repeat metabolic panel in two weeks."
    ),
    "Приклад 2": (
        "Patient is an 88 male presenting to the ED with complaints of "
        "decreased p.o. intake and behavioral changes over the past few days. History is "
        "obtained from the son. She states that the patient has a history of "
        "Alzheimer's dementia. He normally walks unassisted but today could not bear "
        "weight in the lower extremities."
    ),
    "Приклад 3": (
        "During the procedure an aerosol therapy was administered intranasally twice per day. Nursing staff "
        "documented mild epistaxis afterwards but no other complications. Follow-up instructions emphasise "
        "proper humidification for any future aerosol therapy intranasally delivered at home."
    ),
}



@dataclass(frozen=True)
class EntityRow:
    """Normalized entity row used for rendering in the table."""

    pretty_name: str
    cluster_title: str
    detected_name: str
    accuracy: float
    start: int
    end: int
    cui: str

    @classmethod
    def from_raw(cls, payload: dict[str, Any], cluster_title: str) -> "EntityRow":
        return cls(
            pretty_name=payload.get("pretty_name", ""),
            cluster_title=cluster_title,
            detected_name=payload.get("detected_name", ""),
            accuracy=float(payload.get("acc", 0.0) or 0.0),
            start=int(payload.get("start", -1) or -1),
            end=int(payload.get("end", -1) or -1),
            cui=str(payload.get("cui", "")),
        )


def _available_models() -> list[str]:
    if not MODELS_DIR.exists():
        return []

    candidates: list[str] = []
    for path in MODELS_DIR.iterdir():
        if path.is_dir() or path.suffix == ".zip":
            candidates.append(path.name)
    return sorted(candidates)


def _resolve_model_path(model_name: str) -> Path:
    target = MODELS_DIR / model_name
    if not target.exists():
        raise FileNotFoundError(
            f"Model '{model_name}' not found in {MODELS_DIR}. "
            "Перевірте, чи файл розташований у директорії `models/`."
        )
    return target


def _is_placeholder_model(model_path: Path) -> bool:
    """Detect placeholder packs that should not be loaded yet."""
    if model_path.is_dir() and (model_path / "PLACEHOLDER.txt").exists():
        return True
    return False


def _cluster_titles() -> Dict[str, str]:
    global CLUSTER_MAP
    if CLUSTER_MAP is not None:
        return CLUSTER_MAP

    mapping: Dict[str, str] = {}
    csv_path = PROJECT_ROOT / "data/internal_short.csv"
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8") as src:
            reader = csv.DictReader(src)
            for row in reader:
                cui = (row.get("uid") or "").strip()
                cluster_title = (row.get("cluster_title") or "").strip()
                cluster_id = (row.get("cluster") or "").strip()
                if cluster_id and cluster_title:
                    mapping[cluster_id] = cluster_title
                    mapping[cluster_id.lower()] = cluster_title
                # fallback by CUI if cluster id missing
                if not cluster_id and cui and cluster_title:
                    mapping[cui] = cluster_title
                    mapping[cui.lower()] = cluster_title

    CLUSTER_MAP = mapping
    return mapping


def _to_json_safe(value: Any) -> Any:
    """Recursively convert dict keys to strings for JSON serialization."""
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_to_json_safe(v) for v in value]
    return value


def _render_highlight(text: str, raw_result: Dict[str, Any]) -> str:
    entities = raw_result.get("entities") or {}
    spans: Dict[Tuple[int, int, str], Dict[str, Any]] = {}

    for entity in entities.values():
        start = entity.get("start")
        end = entity.get("end")
        if isinstance(start, int) and isinstance(end, int) and end > start:
            key = (start, end, "keyword")
            spans.setdefault(
                key,
                {
                    "start": start,
                    "end": end,
                    "type": "keyword",
                    "label": entity.get("pretty_name") or entity.get("detected_name") or entity.get("cui"),
                },
            )

        for hint in entity.get("value_hints") or []:
            h_start = hint.get("start")
            h_end = hint.get("end")
            if isinstance(h_start, int) and isinstance(h_end, int) and h_end > h_start:
                key = (h_start, h_end, "value")
                spans.setdefault(
                    key,
                    {
                        "start": h_start,
                        "end": h_end,
                        "type": "value",
                        "label": hint.get("value") or hint.get("pattern") or hint.get("type"),
                    },
                )

    if not spans:
        return f"<div class='medcat-highlight'>{html.escape(text)}</div>"

    ordered_spans = sorted(spans.values(), key=lambda item: (item["start"], -(item["end"] - item["start"])))
    pieces: List[str] = []
    cursor = 0
    for span in ordered_spans:
        start = span["start"]
        end = span["end"]
        if start > cursor:
            pieces.append(html.escape(text[cursor:start]))
        segment = html.escape(text[start:end])
        css_class = "keyword-span" if span["type"] == "keyword" else "value-span"
        title = html.escape(span.get("label") or "")
        pieces.append(f"<span class='{css_class}' title='{title}'>{segment}</span>")
        cursor = max(cursor, end)

    if cursor < len(text):
        pieces.append(html.escape(text[cursor:]))

    style = """
    <style>
    .medcat-highlight { font-family: var(--body-font); line-height: 1.5; white-space: pre-wrap; }
    .medcat-highlight .keyword-span {
        border: 1px solid #c0392b;
        background: rgba(192, 57, 43, 0.15);
        padding: 0 2px;
        border-radius: 3px;
    }
    .medcat-highlight .value-span {
        border: 1px solid #27ae60;
        background: rgba(39, 174, 96, 0.15);
        padding: 0 2px;
        border-radius: 3px;
    }
    </style>
    """
    return f"{style}<div class='medcat-highlight'>{''.join(pieces)}</div>"


def _run_extraction(text: str, model_name: str, min_accuracy: float) -> tuple[list[list[Any]], dict[str, Any], str, str]:
    if not text.strip():
        return [], {}, "Введіть текст для аналізу.", "<div class='medcat-highlight'></div>"

    model_path = _resolve_model_path(model_name)
    if _is_placeholder_model(model_path):
        return (
            [],
            {},
            "Обраний пак є плейсхолдером. Запустіть пайплайн створення кастомної моделі "
            "та замініть вміст `models/IEE_MedCAT_v1/` на реальний MedCAT пак.",
            "<div class='medcat-highlight'></div>",
        )
    cat = load_model_pack_auto(model_path)
    raw_result = extract_entities(cat, text)
    entities = raw_result.get("entities", {})

    rows = []
    cluster_map = _cluster_titles()
    for entity in entities.values():
        cluster_title = "—"
        type_ids = entity.get("type_ids") or []
        for type_id in type_ids:
            title = cluster_map.get(type_id) or cluster_map.get(str(type_id).lower())
            if title:
                cluster_title = title
                break
        if cluster_title == "—":
            cui = str(entity.get("cui", "") or "")
            cluster_title = cluster_map.get(cui) or cluster_map.get(cui.lower(), "—")
        row = EntityRow.from_raw(entity, cluster_title=cluster_title)
        if row.accuracy >= min_accuracy:
            rows.append([
                row.pretty_name,
                row.cluster_title,
                row.detected_name,
                f"{row.accuracy:.3f}",
                row.start,
                row.end,
            ])

    if not rows:
        message = "Сутностей не знайдено за заданими критеріями."
    else:
        message = f"Знайдено {len(rows)} сутностей."

    json_safe = _to_json_safe(raw_result)
    highlight_html = _render_highlight(text, raw_result)

    return rows, json_safe, highlight_html, message


def build_demo() -> gr.Blocks:
    model_choices = _available_models()
    preferred_default = PREFERRED_MODEL
    if preferred_default in model_choices:
        default_model = preferred_default
    else:
        fallback_default = "v2_Snomed2025_MIMIC_IV_bbe806e192df009f.zip"
        if fallback_default in model_choices:
            default_model = fallback_default
        else:
            default_model = model_choices[0] if model_choices else ""

    with gr.Blocks(title="MedCAT Entity Extraction Demo") as demo:
        gr.Markdown(
            """
            # MedCAT Demo
            Інтерактивний інтерфейс для випробування витягання медичних сутностей за допомогою MedCAT.
            Виберіть модель, введіть текст і натисніть **Запустити**.
            """
        )

        with gr.Row():
            model_dropdown = gr.Dropdown(
                label="Модель",
                choices=model_choices,
                value=default_model,
                interactive=True,
                info="Файли та директорії у `models/`",
            )
            min_acc_slider = gr.Slider(
                label="Мінімальна впевненість",
                minimum=0.0,
                maximum=1.0,
                value=0.0,
                step=0.05,
            )


        with gr.Row():
            example_choices = gr.Radio(
                choices=list(SAMPLE_TEXTS.keys()),
                label="Приклади текстів",
                value=list(SAMPLE_TEXTS.keys())[0],
            )
            load_example_btn = gr.Button("Завантажити приклад")

        text_input = gr.Textbox(
            label="Вхідний текст",
            placeholder="Введіть клінічний запис...",
            lines=8,
        )
        run_button = gr.Button("Запустити")

        def _load_example(example_key: str) -> str:
            return SAMPLE_TEXTS.get(example_key, "")

        load_example_btn.click(
            fn=_load_example,
            inputs=[example_choices],
            outputs=[text_input],
        )

        with gr.Row():
            entities_table = gr.Dataframe(
                headers=["Назва", "Кластер", "Виявлене значення", "Впевненість", "Початок", "Кінець"],
                datatype=["str", "str", "str", "str", "number", "number"],
                row_count=(0, "dynamic"),
                col_count=6,
                label="Розпізнані сутності",
                value=[],
            )
            raw_json = gr.JSON(label="Сирий результат MedCAT")
        text_highlight = gr.HTML(label="Підсвічений текст")

        status = gr.Markdown()

        run_button.click(
            fn=_run_extraction,
            inputs=[text_input, model_dropdown, min_acc_slider],
            outputs=[entities_table, raw_json, text_highlight, status],
        )

    return demo


def launch(*, share: bool = False, server_port: int | None = None) -> None:
    """Launch the Gradio demo server."""
    demo = build_demo()
    demo.launch(share=share, server_port=server_port)


if __name__ == "__main__":
    launch()
