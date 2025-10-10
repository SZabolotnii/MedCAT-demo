# MedCAT Demo Project

Цей репозиторій містить демонстраційний проект для роботи з [MedCAT](https://github.com/CogStack/MedCAT) - інструментом для медичного NER (Named Entity Recognition) та концептуального зв'язування.

## Структура проекту

```
.
├── data/                          # Тестові набори й службові CSV/JSON
│   ├── phase1a_annotated_entities.json
│   ├── test_clinical_notes.json
│   ├── test_docs/
│   └── valid_clusters.json
├── info/                          # Довідкові матеріали та методології
│   ├── MedCAT Testing Methodology & Framework.md
│   └── ...
├── models/                        # Пакети MedCAT
│   ├── IEE_MedCAT_v1/             # Розпакований кастомний пак
│   └── IEE_MedCAT_v1.zip          # Архів для розповсюдження
├── reports/                       # Автоматично згенеровані звіти
│   ├── phase1a_validation.md
│   └── validation_suite.json
├── scripts/                       # Утиліти для перетворення й тестування
│   ├── run_validation_suite.py
│   ├── validate_phase1a.py
│   └── performance_benchmark.py
├── src/                           # Джерельний код
│   ├── custom_cat_v2.py
│   ├── extractor.py
│   ├── gradio_app.py
│   ├── testing_framework/
│   └── utils.py
├── tests/                         # Pytest-специфікації
│   ├── test_dictionary_coverage.py
│   └── ...
├── TESTING_INSTRUCTION.md         # Покрокова інструкція
├── TESTING_METHODOLOGY.md         # Стратегія тестування
├── requirements.txt
└── README.md
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
cat = load_model_pack("models/IEE_MedCAT_v1.zip")

# Обробка тексту
text = "The patient complains of headache and fever."
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

## Нотатки по інтерфейсу

- Gradio (`python -m src.gradio_app`) автоматично підхоплює внутрішній пак `models/IEE_MedCAT_v1/` і показує назви кластерів з `data/internal_short.csv`. Якщо запускаєте файл напряму (`.venv/bin/python src/gradio_app.py`), імпорт пакетів також працює — додано fallback для абсолютних імпортів.
- Стовпчик «Кластер» у таблиці може бути `—` для моделей, у яких `type_ids` відсутні у нашій онтології або пакет зовнішній (без `data/internal_short.csv`).

Для швидких перевірок доступні синтетичні документи в `data/test_docs/`.

## Тестування та валідація

- Запуск всіх автоматичних тестів:
  ```bash
  pytest
  ```
  Серед них є:
  - покриття словника (`tests/test_dictionary_coverage.py`) з перевіркою відповідності `cdb_stats.json`;
  - оцінка точності NER на `data/phase1a_annotated_entities.json`;
  - тести на комбіновані підказки та їхню допускову «щілину»;
  - бенчмарк продуктивності (`tests/test_performance_benchmark.py`).

- Повний валідатор фази 1A з репортуванням:
  ```bash
  python -m scripts.run_validation_suite \
      --model models/IEE_MedCAT_v1 \
      --combined-hints models/IEE_MedCAT_v1/internal_combined_hints.json \
      --performance-batch-sizes 1 10 50
  ```
  Скрипт генерує агрегований JSON у `reports/validation_suite.json` та сигналізує, чи виконані критерії (`F1 ≥ 0.75`, `Precision/Recall` тощо).

- Окремий бенчмарк продуктивності з можливістю порівняння з базовим заміром:
  ```bash
  python -m scripts.performance_benchmark \
      --documents data/test_docs \
      --output reports/performance_benchmark.json
  ```
  Підтримується `--baseline` для порівняння з попередніми запускaми.

- Контрольні набори:
  - `data/test_clinical_notes.json` — 3 короткі нотатки для швидкої перевірки; усі очікувані CUI зберігаються у верхньому регістрі й синхронізовані з поточною моделлю (100 % покриття).
  - `data/phase1a_annotated_entities.json` — розмічений датасет для обчислення F1/Precision/Recall.
  - `data/test_docs/` — синтетичні тексти для бенчмарку продуктивності.

- Звіти та артефакти:
  - `reports/phase1a_validation.md` — узагальнена вивантаження Phase 1A (`scripts.validate_phase1a`).
  - `reports/validation_suite.json` — агрегований звіт інтегрованого валідатора з ключовими метриками.
  - `reports/performance_benchmark.json` — останній замір швидкодії; за потреби зберігайте попередній у `reports/baseline_performance.json`.

Повний опис методології та чеклісти для ручних перевірок:
- `TESTING_METHODOLOGY.md` — стратегія та критерії GO/NO-GO.
- `TESTING_INSTRUCTION.md` — покрокова інструкція запуску тестів і збора артефактів.
