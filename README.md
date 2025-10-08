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
from src.extractor import MedCATExtractor

# Ініціалізація екстрактора
extractor = MedCATExtractor()

# Обробка тексту
text = "Пацієнт скаржиться на головний біль та підвищену температуру."
results = extractor.extract_entities(text)
print(results)
```

## Залежності

Основні залежності включають:
- `medcat[spacy,meta-cat,deid,rel-cat,dict-ner]` - основний пакет MedCAT
- `torch` - PyTorch для машинного навчання
- `transformers` - для роботи з трансформерами
- `scikit-learn` - для ML утиліт

## Ліцензія

Цей проект використовується для демонстраційних цілей.
