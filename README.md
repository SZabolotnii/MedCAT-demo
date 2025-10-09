# MedCAT Demo Project

Цей репозиторій містить демонстраційний проект для роботи з [MedCAT](https://github.com/CogStack/MedCAT) - інструментом для медичного NER (Named Entity Recognition) та концептуального зв'язування.

## Структура проекту

```
.
├── models/
│   ├── umls_sm_pt2ch_533bab5115c6c2d6.zip           # UMLS модель
│   ├── v2_Snomed2025_MIMIC_IV_bbe806e192df009f/     # SNOMED модель
│   └── v2_Snomed2025_MIMIC_IV_bbe806e192df009f.zip  # Архів SNOMED моделі
├── src/
│   ├── __init__.py
│   ├── extractor.py               # Основний модуль для екстракції сутностей
│   └── utils.py                   # Утиліти для роботи з моделями
├── requirements.txt               # Залежності Python
├── .gitignore                     # Git ignore файл
└── README.md                      # Документація
```

## Встановлення та налаштування

1. Створіть та активуйте віртуальне середовище:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # На Linux/macOS
   # або
   .venv\Scripts\activate     # На Windows
   ```

2. Встановіть залежності:
   ```bash
   pip install -r requirements.txt
   ```

3. Моделі MedCAT вже включені в проект:
   - **UMLS модель**: для загальної медичної термінології
   - **SNOMED модель**: для роботи з SNOMED CT та MIMIC-IV даними

## Використання

```python
from src.utils import load_model_pack
from src.extractor import extract_entities

# Завантаження моделі (підтримує .zip або розпаковану директорію)
cat = load_model_pack("models/v2_Snomed2025_MIMIC_IV_bbe806e192df009f.zip")

# Обробка тексту
text = "Пацієнт скаржиться на головний біль та підвищену температуру."
results = extract_entities(cat, text)
print(results)
```

Модуль `src.utils` містить один вхідний пункт для завантаження моделей (`load_model_pack`), а `src.extractor` фокусується на викликах для вилучення сутностей (`extract_entities`). Це дозволяє уникнути дубльованої логіки роботи з MedCAT.

`load_model_pack` кешує завантажений екземпляр моделі в межах процесу — це значно скорочує час відгуку сервісів, які багаторазово звертаються до одного й того ж паку. Для вимушеного оновлення можна передати `force_reload=True`, а для вимкнення кешу — `use_cache=False`.

## Gradio інтерфейс

У проєкті також є інтерактивний інтерфейс на основі Gradio. Запустити його можна так:

```bash
python -m src.gradio_app
```

Після запуску у консолі з'явиться локальна адреса (зазвичай `http://127.0.0.1:7860/`). Інтерфейс дає змогу:
- вибрати модель з директорії `models/`;
- ввести текст для аналізу;
- фільтрувати сутності за мінімальним рівнем впевненості;
- переглянути таблицю розпізнаних сутностей та сирий JSON-вивід MedCAT.

## Конвертація внутрішніх даних

Щоб перетворити `data/internal.json` у табличний CSV, скористайтесь утилітою:

```bash
python -m scripts.convert_internal_json_to_csv
```

За замовчуванням результат буде записаний у `data/internal.csv`. Шлях до вхідного та вихідного файлів можна змінити за допомогою аргументів `--input` та `--output`.

## Залежності

Основні залежності включають:
- `medcat[spacy,meta-cat,deid,rel-cat,dict-ner]` - основний пакет MedCAT
- `torch` - PyTorch для машинного навчання
- `transformers` - для роботи з трансформерами
- `scikit-learn` - для ML утиліт

## Ліцензія

Цей проект використовується для демонстраційних цілей.

## Кастомна внутрішня онтологія (підготовка)

1. Згенерувати CSV у форматі MedCAT та метадані комбінованих hints:
   ```bash
   python -m scripts.transform_to_medcat_format \
       --input data/internal_short.csv \
       --output data/internal_medcat_v2.csv
   ```
2. Побудувати CDB зі словником:
   ```bash
   python -m scripts.create_cdb_v2 \
       --csv data/internal_medcat_v2.csv \
       --output-dir models/IEE_MedCAT_v1
   ```
3. Запустити швидку валідацію на синтетичних нотатках:
   ```bash
   python -m scripts.validate_phase1a \
       --model models/IEE_MedCAT_v1 \
       --combined-hints data/internal_combined_hints.json \
       --test-set data/test_clinical_notes.json
   ```
   Результат буде записаний у `reports/phase1a_validation.md`.
4. (Опційно) Запакувати результат для поширення:
   ```bash
   python -m scripts.create_model_pack \
       --source-dir models/IEE_MedCAT_v1 \
       --output models/IEE_MedCAT_v1.zip
   ```
5. Використати отриманий пак у Gradio інтерфейсі (він уже з'явиться у списку
   моделей). Якщо пак ще не створений, інтерфейс підкаже замінити плейсхолдер.

Для швидких перевірок доступні синтетичні документи в `data/test_docs/`.
