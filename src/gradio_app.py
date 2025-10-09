"""Gradio interface for interactive MedCAT entity extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


@dataclass(frozen=True)
class EntityRow:
    """Normalized entity row used for rendering in the table."""

    pretty_name: str
    cui: str
    detected_name: str
    accuracy: float
    start: int
    end: int

    @classmethod
    def from_raw(cls, payload: dict[str, Any]) -> "EntityRow":
        return cls(
            pretty_name=payload.get("pretty_name", ""),
            cui=str(payload.get("cui", "")),
            detected_name=payload.get("detected_name", ""),
            accuracy=float(payload.get("acc", 0.0) or 0.0),
            start=int(payload.get("start", -1) or -1),
            end=int(payload.get("end", -1) or -1),
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


def _to_json_safe(value: Any) -> Any:
    """Recursively convert dict keys to strings for JSON serialization."""
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_to_json_safe(v) for v in value]
    return value


def _run_extraction(text: str, model_name: str, min_accuracy: float) -> tuple[list[list[Any]], dict[str, Any], str]:
    if not text.strip():
        return [], {}, "Введіть текст для аналізу."

    model_path = _resolve_model_path(model_name)
    if _is_placeholder_model(model_path):
        return (
            [],
            {},
            "Обраний пак є плейсхолдером. Запустіть пайплайн створення кастомної моделі "
            "та замініть вміст `models/custom_internal_demo_pack/` на реальний MedCAT пак.",
        )
    cat = load_model_pack_auto(model_path)
    raw_result = extract_entities(cat, text)
    entities = raw_result.get("entities", {})

    rows = []
    for entity in entities.values():
        row = EntityRow.from_raw(entity)
        if row.accuracy >= min_accuracy:
            rows.append([
                row.pretty_name,
                row.cui,
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

    return rows, json_safe, message


def build_demo() -> gr.Blocks:
    model_choices = _available_models()
    preferred_default = "custom_internal_demo_pack"
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

        text_input = gr.Textbox(
            label="Вхідний текст",
            placeholder="Введіть клінічний запис...",
            lines=8,
        )
        run_button = gr.Button("Запустити")

        with gr.Row():
            entities_table = gr.Dataframe(
                headers=["Назва", "CUI", "Виявлене значення", "Впевненість", "Початок", "Кінець"],
                datatype=["str", "str", "str", "str", "number", "number"],
                row_count=(0, "dynamic"),
                col_count=6,
                label="Розпізнані сутності",
                value=[],
            )
            raw_json = gr.JSON(label="Сирий результат MedCAT")

        status = gr.Markdown()

        run_button.click(
            fn=_run_extraction,
            inputs=[text_input, model_dropdown, min_acc_slider],
            outputs=[entities_table, raw_json, status],
        )

    return demo


def launch(*, share: bool = False, server_port: int | None = None) -> None:
    """Launch the Gradio demo server."""
    demo = build_demo()
    demo.launch(share=share, server_port=server_port)


if __name__ == "__main__":
    launch()
