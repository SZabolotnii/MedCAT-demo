# Покрокова інструкція з тестування MedCAT (IEE_MedCAT_v1)

Ця інструкція описує практичні кроки для повної перевірки моделі `models/IEE_MedCAT_v1` відповідно до методики з `info/MedCAT Testing Methodology & Framework.md`.

## 1. Попередня підготовка
- **Середовище**: активуйте віртуальне середовище та встановіть залежності.
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
- **Перевірка ресурсів**: переконайтесь, що в проекті наявні:
  - `models/IEE_MedCAT_v1/` (розпакований пак з custom CDB);
  - `data/valid_clusters.json` (мапінг кластерів);
  - `data/internal_combined_hints.json` або `models/IEE_MedCAT_v1/internal_combined_hints.json`;
  - `data/phase1a_annotated_entities.json` (розмітка для тестів).

## 2. Базове sanity-перевіряння
1. Переконайтесь у цілісності файлів CDB:
   ```bash
   ls models/IEE_MedCAT_v1/custom_cdb_v2
   cat models/IEE_MedCAT_v1/cdb_stats.json
   ```
2. Опційно: швидкий запуск `scripts/validate_phase1a.py`, щоб переконатися у працездатності кастомного `CustomCAT`.

## 3. Автоматизовані тести (щоденно/при змінах)
1. Запустіть усі pytest-тести:
   ```bash
   pytest
   ```
   Тести покривають:
   - перевірку відповідності словника (`tests/test_dictionary_coverage.py`);
   - NER-метрики на розмічених документах (`tests/test_entity_detection.py`);
   - gap-tolerant комбіновані підказки (`tests/test_combined_hints.py`);
   - базовий бенчмарк продуктивності (`tests/test_performance_benchmark.py`).
2. Випишіть попередження чи падіння тестів у `reports/phase1a_validation.md` або інший лог.

## 4. Запуск інтегрованого валідатора (щоденно)
1. Виконайте скрипт:
   ```bash
   python -m scripts.run_validation_suite \
       --model models/IEE_MedCAT_v1 \
       --combined-hints models/IEE_MedCAT_v1/internal_combined_hints.json \
       --performance-batch-sizes 1 10 50
   ```
2. Перевірте файл `reports/validation_suite.json`:
   - `dictionary.success` має бути `true`;
   - `entity_detection.metrics.exact_match.f1 ≥ 0.75`, `partial_match.f1 ≥ 0.80`, `type_accuracy.accuracy ≥ 0.85`;
   - `combined_hints.success == true`;
   - `performance.success == true` (≥10 docs/s, ≤2GB memory delta).
3. За потреби зафіксуйте результати у `reports/phase1a_validation.md` (додайте коротке резюме).

## 5. Окремий бенчмарк продуктивності (щотижня або після оптимізацій)
1. Запуск без базового файлу:
   ```bash
   python -m scripts.performance_benchmark \
       --documents data/test_docs \
       --output reports/performance_benchmark.json
   ```
2. Для порівняння з попереднім заміром додайте `--baseline reports/baseline_performance.json`. Зміни у швидкодії чи пам'яті >10% задокументуйте у звіті.

## 6. Розширення тестового набору
- Оновлюйте `data/phase1a_annotated_entities.json`, коли з’являються нові сутності або складні випадки вхідних текстів.
- Для великого корпусу використовуйте формат JSONL (кожен рядок — документ з ключем `text`) та передавайте шлях у `scripts/performance_benchmark.py`/`scripts.run_validation_suite`.
- Після оновлення обов’язково повторно запустіть `pytest` і `scripts.run_validation_suite`.

## 7. Робота з комбінованими підказками
1. Додавання нових патернів:
   - дописуйте `data/internal_combined_hints.json` або `models/IEE_MedCAT_v1/internal_combined_hints.json`;
   - дотримуйтесь структури: `{"cui": "...", "name": "...", "components": [...], "max_gap": N}`.
2. Перевалідація:
   - запустіть `pytest tests/test_combined_hints.py`;
   - переконайтесь, що нові патерни покриваються сценаріями (за потреби додайте case у тест).

## 8. Щотижнева експертна перевірка
- **Медичний експерт**:
  - ознайомлюється з `combined_hint_matches` та false positives із `reports/validation_suite.json`;
  - приймає рішення щодо корекції онтології чи патернів.
- **Технічний огляд**:
  - аналіз продуктивності;
  - огляд змін у коді, тестах, комбінованих підказках.

## 9. GO/NO-GO для Phase 1B
- Критерії:
  - F1 ≥ 0.78 (точні збіги) та Precision ≥ 0.80;
  - ΔPerformance в межах допусків;
  - відсутність критичних дефектів у комбінованих hints.
- Документуйте рішення в окремому розділі `reports/phase1a_validation.md`, додаючи:
  - основні метрики (F1, Precision, Recall);
  - підсумок експертного рев’ю;
  - рекомендації для наступної фази.

## 10. Автоматизація (CI/CD)
- Додайте `python -m scripts.run_validation_suite` у pre-merge pipeline.
- Опціонально: розширте GitHub Actions чи інший CI задачами для
  - `pytest`;
  - `scripts.performance_benchmark` (з невеликим корпусом);
  - публікації `reports/validation_suite.json` як артефакта.

---

> **Пам’ятайте:** після кожної зміни у моделях, словнику або коді проходьте повний цикл: `pytest` → `run_validation_suite` → оновлення звітів. Це гарантує контроль якості та трасованість рішень щодо переходу до Phase 1B.
