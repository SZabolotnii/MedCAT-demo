# MedCAT v2 Implementation Plan: Custom Medical Ontology Integration

## Стратегічна Філософія Проекту

> **"Complexity is the enemy of reliable medical NLP"**

### Ключові Архітектурні Принципи

**Модульність через ізоляцію:**
- Кожна фаза створює самостійний, робочий компонент
- Фаза 1A працює незалежно від 1B
- Semantic layer додається без переписування dictionary foundation

**Human-centric design:**
- Мінімізація cognitive load для медичних експертів при валідації
- Self-documenting code з чіткими коментарями
- Прозорі метрики для non-technical stakeholders

**Adaptive architecture:**
- Можливість додавання нових keywords без retraining
- Swap semantic similarity engines (Bio_ClinicalBERT → інші embeddings)
- Scale від 7K до 70K+ concepts без architectural refactoring

---

## Executive Summary

**Проблема:** Інтеграція 7,219 медичних концептів (keywords + hints) у MedCAT v2 для автоматичної детекції в англомовних клінічних текстах.

**Рішення:** Трифазна гібридна архітектура - dictionary-based foundation з optional semantic enhancement.

**Timeline:** 6-8 тижнів до production-ready моделі
**Budget:** $60K-$100K (за 1 senior ML engineer + 0.5 medical annotator)
**Expected Performance:** F1 ≈ 0.78-0.85 (Phase 1A) → 0.83-0.88 (Phase 1B)

**Критична перевага:** 0% ambiguity онтологія дозволяє швидкий deployment без costly unsupervised training.

---

## Phase 1A: Dictionary-Based Foundation [ПРІОРИТЕТ]

**Стратегічне обґрунтування:** Створити minimal viable extractor за 2-3 тижні, що покриває 75-80% use cases. Це дає immediate value та baseline для оцінки необхідності ML enhancement.

### Тиждень 1: Data Preparation & CDB Creation

#### 1.1 Трансформація CSV → MedCAT Format

**Мета:** Конвертувати `internal_short.csv` у структуру, яку MedCAT v2 CDB може імпортувати.

**Вхідні дані:**
```
internal_short.csv:
- source, keyword, uid, cluster, cluster_title, keyword_hints
- 7,219 records
- keyword_hints: pipe-separated synonyms з [combined_hint] markers
```

**Вихідні дані:**
```
internal_medcat_v2.csv:
- cui, name, ontologies, name_status, type_ids, description
- ~70K rows (7,219 keywords × ~10 synonyms кожен)
```

**Implementation Script:**

```python
"""
scripts/transform_to_medcat_format.py
Конвертація internal_short.csv → MedCAT v2 CDB CSV format
"""

import pandas as pd
import re
from pathlib import Path
from typing import List, Tuple

def parse_combined_hints(hint: str) -> Tuple[str, bool, List[str]]:
    """
    Парсинг hints з [combined_hint] markers.
    
    Args:
        hint: "aerosol [combined_hint] intranasally"
    
    Returns:
        - cleaned_hint: "aerosol intranasally"
        - is_combined: True
        - components: ["aerosol", "intranasally"]
    """
    is_combined = '[combined_hint]' in hint
    cleaned = re.sub(r'\s*\[combined_hint\]\s*', ' ', hint).strip()
    components = cleaned.split() if is_combined else [cleaned]
    return cleaned, is_combined, components

def expand_keywords_to_medcat_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Перетворення кожного keyword + hints → множинні MedCAT rows.
    
    Логіка:
    1. Primary row: keyword як preferred name (name_status='P')
    2. Synonym rows: кожен hint як automatic name (name_status='A')
    3. Combined hints: створюємо спеціальну metadata для gap tolerance
    """
    medcat_rows = []
    
    for idx, row in df.iterrows():
        cui = row['uid']  # Використовуємо existing UID як CUI
        keyword = row['keyword']
        cluster_id = row['cluster']
        cluster_title = row['cluster_title']
        
        # Primary name entry
        medcat_rows.append({
            'cui': cui,
            'name': keyword,
            'ontologies': 'CUSTOM_INTERNAL',
            'name_status': 'P',
            'type_ids': cluster_id,
            'description': f'{cluster_title} | Primary concept name'
        })
        
        # Synonym entries з hints
        if pd.notna(row['keyword_hints']) and row['keyword_hints'].strip():
            hints = [h.strip() for h in row['keyword_hints'].split('|') if h.strip()]
            
            for hint in hints:
                cleaned_hint, is_combined, components = parse_combined_hints(hint)
                
                # Skip якщо hint ідентичний keyword (redundant)
                if cleaned_hint.lower() == keyword.lower():
                    continue
                
                description = f'{cluster_title} | Synonym'
                if is_combined:
                    description += f' | COMBINED: {" + ".join(components)}'
                
                medcat_rows.append({
                    'cui': cui,
                    'name': cleaned_hint,
                    'ontologies': '',
                    'name_status': 'A',
                    'type_ids': '',
                    'description': description
                })
    
    return pd.DataFrame(medcat_rows)

def validate_transformed_data(df: pd.DataFrame) -> dict:
    """Валідація якості трансформованих даних."""
    stats = {
        'total_rows': len(df),
        'unique_cuis': df['cui'].nunique(),
        'primary_names': len(df[df['name_status'] == 'P']),
        'synonyms': len(df[df['name_status'] == 'A']),
        'avg_synonyms_per_cui': len(df[df['name_status'] == 'A']) / df['cui'].nunique(),
        'empty_names': df['name'].isna().sum(),
        'combined_hints': df['description'].str.contains('COMBINED', na=False).sum()
    }
    return stats

def main():
    # Завантаження
    input_path = Path('internal_short.csv')
    df = pd.read_csv(input_path)
    
    print(f"📥 Завантажено {len(df)} records з {input_path}")
    
    # Трансформація
    medcat_df = expand_keywords_to_medcat_rows(df)
    
    # Валідація
    stats = validate_transformed_data(medcat_df)
    print("\n✅ Трансформація завершена:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Перевірка на критичні проблеми
    if stats['empty_names'] > 0:
        print(f"\n⚠️ УВАГА: {stats['empty_names']} порожніх names - потрібна очистка!")
    
    if stats['avg_synonyms_per_cui'] < 3:
        print(f"\n⚠️ УВАГА: Низька щільність синонімів ({stats['avg_synonyms_per_cui']:.1f}) - recall може бути низьким")
    
    # Збереження
    output_path = Path('data/internal_medcat_v2.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    medcat_df.to_csv(output_path, index=False)
    
    print(f"\n💾 Збережено у {output_path}")
    
    # Sample output для перевірки
    print("\n📋 Приклад виходу (перші 3 CUI):")
    sample_cuis = medcat_df['cui'].unique()[:3]
    for cui in sample_cuis:
        subset = medcat_df[medcat_df['cui'] == cui]
        print(f"\n   CUI {cui}:")
        for _, row in subset.iterrows():
            status_label = "PRIMARY" if row['name_status'] == 'P' else "SYNONYM"
            print(f"      [{status_label}] {row['name']}")

if __name__ == '__main__':
    main()
```

**Execution:**
```bash
python scripts/transform_to_medcat_format.py
```

**Success Criteria:**
- ✅ ~70K rows generated (7,219 CUI × ~10 names кожен)
- ✅ Кожен CUI має 1 primary name (P) + N synonyms (A)
- ✅ 0 empty names
- ✅ Combined hints identified у description

**Time Estimate:** 2-3 дні (script dev + validation + debugging)

---

#### 1.2 CDB Creation з Custom Ontology

**Мета:** Створити MedCAT v2 Concept Database з трансформованого CSV.

**Architecture Decision:** Використовуємо `CDBMaker.prepare_csvs()` з `full_build=False` для швидкого створення без context vectors (Phase 1A не потребує їх).

**Implementation:**

```python
"""
scripts/create_cdb_v2.py
Створення MedCAT v2 CDB з custom ontology CSV
"""

from medcat.cdb_maker import CDBMaker
from medcat.config import Config
from pathlib import Path
import json

def configure_for_dictionary_mode(config: Config) -> Config:
    """
    Конфігурація MedCAT для dictionary-only режиму без ML.
    
    Ключові налаштування:
    - Вимкнути training
    - Вимкнути context similarity (поки що)
    - Налаштувати spell checking
    - Оптимізувати для high precision
    """
    # === GENERAL ===
    config.general['spacy_model'] = 'en_core_web_md'
    config.general['train'] = False
    config.general['spell_check'] = True
    config.general['spell_check_len_limit'] = 3  # Мін довжина для spell check
    
    # === LINKING (без ML) ===
    config.linking['train'] = False
    config.linking['always_calculate_similarity'] = False  # Phase 1A: disabled
    config.linking['calculate_dynamic_threshold'] = False
    config.linking['similarity_threshold'] = 1.0  # Effectively disabled
    
    # Rule-based disambiguation preferences
    config.linking['prefer_primary_name'] = 0.6  # Перевага primary names
    config.linking['prefer_frequent_concepts'] = 0.3
    config.linking['disamb_length_limit'] = 6  # Max words для disambiguation
    
    # === NER ===
    config.ner['min_name_len'] = 2  # Дозволити короткі медичні терміни
    config.ner['upper_case_limit_len'] = 4  # COPD, GERD тощо
    config.ner['check_upper_case_names'] = True
    config.ner['try_reverse_word_order'] = True  # "pain chest" → "chest pain"
    
    return config

def create_cdb_from_csv(
    csv_path: Path,
    output_dir: Path,
    config: Config
) -> 'CDB':
    """Створення CDB через CDBMaker."""
    
    print(f"🔧 Створення CDB з {csv_path}")
    
    maker = CDBMaker(config=config)
    
    cdb = maker.prepare_csvs(
        csv_paths=[str(csv_path)],
        sep=',',
        encoding='utf-8',
        full_build=False  # Phase 1A: dictionary-only, без context vectors
    )
    
    print(f"✅ CDB створено: {len(cdb.cui2names)} CUI")
    
    return cdb

def enrich_cdb_metadata(cdb: 'CDB') -> 'CDB':
    """
    Додавання метаданих для type_ids (cluster mappings).
    
    MedCAT використовує type_ids для semantic grouping.
    Маппимо cluster IDs → human-readable назви.
    """
    # Витягуємо унікальні type_id → cluster_title з опису
    type_id_names = {}
    
    for cui, names in cdb.cui2names.items():
        type_ids = cdb.cui2type_ids.get(cui, set())
        
        for type_id in type_ids:
            if type_id not in type_id_names:
                # Витягуємо cluster_title з description
                # Format: "Action/Drug | Primary concept name"
                if cui in cdb.cui2preferred_name:
                    pref_name = cdb.cui2preferred_name[cui]
                    # Простий lookup - можна покращити
                    type_id_names[type_id] = f"Cluster_{type_id[:8]}"
    
    # Додаємо до CDB metadata
    cdb.addl_info['type_id2name'] = type_id_names
    
    print(f"📝 Додано {len(type_id_names)} type ID mappings")
    
    return cdb

def validate_cdb_quality(cdb: 'CDB') -> dict:
    """Валідація CDB після створення."""
    
    stats = {
        'total_cuis': len(cdb.cui2names),
        'total_names': sum(len(names) for names in cdb.cui2names.values()),
        'avg_names_per_cui': sum(len(names) for names in cdb.cui2names.values()) / len(cdb.cui2names),
        'type_ids_count': len(set().union(*[cdb.cui2type_ids.get(cui, set()) for cui in cdb.cui2names])),
        'preferred_names_coverage': len(cdb.cui2preferred_name) / len(cdb.cui2names) * 100
    }
    
    return stats

def save_cdb_with_metadata(cdb: 'CDB', output_dir: Path):
    """Збереження CDB + metadata для Phase 1A."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Збереження CDB
    cdb_path = output_dir / 'custom_cdb_v2.dat'
    cdb.save(str(cdb_path))
    
    # Збереження конфігурації
    config_path = output_dir / 'config.json'
    cdb.config.save(str(config_path))
    
    # Збереження статистики
    stats = validate_cdb_quality(cdb)
    stats_path = output_dir / 'cdb_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n💾 CDB збережено:")
    print(f"   CDB: {cdb_path}")
    print(f"   Config: {config_path}")
    print(f"   Stats: {stats_path}")
    
    print(f"\n📊 CDB Статистика:")
    for key, value in stats.items():
        print(f"   {key}: {value}")

def main():
    csv_path = Path('data/internal_medcat_v2.csv')
    output_dir = Path('models/phase1a_dictionary')
    
    # Конфігурація
    config = Config()
    config = configure_for_dictionary_mode(config)
    
    # Створення CDB
    cdb = create_cdb_from_csv(csv_path, output_dir, config)
    
    # Enrichment
    cdb = enrich_cdb_metadata(cdb)
    
    # Збереження
    save_cdb_with_metadata(cdb, output_dir)
    
    print("\n✅ Phase 1A CDB готовий до використання!")

if __name__ == '__main__':
    main()
```

**Execution:**
```bash
python scripts/create_cdb_v2.py
```

**Output Structure:**
```
models/phase1a_dictionary/
├── custom_cdb_v2.dat          # MedCAT CDB binary
├── config.json                # Configuration
└── cdb_stats.json             # Quality metrics
```

**Success Criteria:**
- ✅ CDB містить 7,219 CUI
- ✅ Середня кількість names per CUI ≈ 8-10
- ✅ 100% CUI мають preferred name
- ✅ Type IDs mapped до cluster titles

**Time Estimate:** 1-2 дні

---

### Тиждень 2: CAT Initialization & Combined Hints Processing

#### 1.3 CAT Instance з Custom Components

**Мета:** Створити MedCAT CAT instance з custom preprocessing для `[combined_hint]` patterns.

**Architectural Challenge:** MedCAT's default expanding window algorithm не враховує gap-tolerant patterns ("aerosol ... intranasally"). Потрібен custom NER component.

**Solution Strategy:** Hybrid pipeline - spaCy preprocessing → Custom Gap-Tolerant Matcher → MedCAT linking.

**Implementation:**

```python
"""
src/custom_cat_v2.py
Custom CAT wrapper з gap-tolerant combined hints matching
"""

from medcat.cat import CAT
from medcat.cdb import CDB
from medcat.config import Config
from pathlib import Path
from typing import Dict, List, Set, Tuple
import re
import spacy
from spacy.tokens import Doc, Span

class CombinedHintMatcher:
    """
    Custom component для gap-tolerant matching of [combined_hint] patterns.
    
    Architecture:
    1. Завантажує combined hints з CDB descriptions
    2. Для кожного hint створює pattern з gap tolerance
    3. Шукає matches у spaCy Doc перед MedCAT processing
    4. Додає як custom entities до Doc
    
    Example:
        Pattern: "aerosol [combined_hint] intranasally"
        Matches: "aerosol administered intranasally" (gap=1)
                 "aerosol spray given intranasally" (gap=2)
    """
    
    def __init__(self, cdb: CDB, max_gap: int = 3):
        """
        Args:
            cdb: MedCAT CDB з descriptions containing COMBINED markers
            max_gap: Максимум слів між частинами combined hint
        """
        self.cdb = cdb
        self.max_gap = max_gap
        self.combined_patterns = self._extract_combined_patterns()
        
        print(f"🔍 Завантажено {len(self.combined_patterns)} combined hint patterns")
    
    def _extract_combined_patterns(self) -> Dict[str, List[Tuple[List[str], str]]]:
        """
        Витягує combined patterns з CDB descriptions.
        
        Returns:
            Dict[cui -> List[(components, full_pattern)]]
            
        Example:
            {
                'CUI123': [
                    (['aerosol', 'intranasally'], 'aerosol intranasally'),
                    (['oral', 'tablet'], 'oral tablet')
                ]
            }
        """
        patterns = {}
        
        for cui, names in self.cdb.cui2names.items():
            cui_patterns = []
            
            for name in names:
                # Шукаємо в descriptions for this name
                # Формат: "... | COMBINED: aerosol + intranasally"
                # (description не зберігається в cui2names, потрібен alternative approach)
                
                # АЛЬТЕРНАТИВА: парсимо directly з name якщо містить markers
                # Для Phase 1A спрощуємо - шукаємо multi-word names як candidates
                words = name.split()
                if len(words) >= 2:  # Potential combined pattern
                    cui_patterns.append((words, name))
            
            if cui_patterns:
                patterns[cui] = cui_patterns
        
        return patterns
    
    def __call__(self, doc: Doc) -> Doc:
        """
        spaCy pipeline component - додає custom entities для combined hints.
        
        Args:
            doc: spaCy Doc
        
        Returns:
            Doc з додатковими entities у doc.ents
        """
        new_ents = []
        
        for cui, patterns in self.combined_patterns.items():
            for components, full_pattern in patterns:
                matches = self._find_gap_tolerant_matches(doc, components)
                
                for start_idx, end_idx in matches:
                    span = Span(doc, start_idx, end_idx, label="COMBINED_HINT")
                    span._.cui = cui  # Custom attribute
                    span._.pattern = full_pattern
                    new_ents.append(span)
        
        # Merge з existing entities (resolve overlaps)
        doc.ents = tuple(new_ents) + doc.ents
        
        return doc
    
    def _find_gap_tolerant_matches(
        self, 
        doc: Doc, 
        components: List[str]
    ) -> List[Tuple[int, int]]:
        """
        Знаходить matches з gap tolerance.
        
        Algorithm:
        1. Знайти першу компоненту
        2. Шукати наступну компоненту в межах max_gap слів
        3. Repeat для всіх компонент
        4. Return (start_idx, end_idx) якщо всі знайдені
        """
        matches = []
        
        for i in range(len(doc)):
            # Перевірка першої компоненти
            if doc[i].text.lower() == components[0].lower():
                # Спроба знайти решту компонент
                match_indices = [i]
                search_idx = i + 1
                
                for comp_idx in range(1, len(components)):
                    target_comp = components[comp_idx].lower()
                    found = False
                    
                    # Шукаємо в межах gap
                    for gap in range(1, self.max_gap + 1):
                        check_idx = search_idx + gap - 1
                        if check_idx < len(doc) and doc[check_idx].text.lower() == target_comp:
                            match_indices.append(check_idx)
                            search_idx = check_idx + 1
                            found = True
                            break
                    
                    if not found:
                        break
                
                # Всі компоненти знайдені
                if len(match_indices) == len(components):
                    start = match_indices[0]
                    end = match_indices[-1] + 1  # spaCy exclusive end
                    matches.append((start, end))
        
        return matches

class CustomCAT:
    """
    Wrapper навколо MedCAT CAT з custom preprocessing.
    
    Architecture:
    - Phase 1A: Dictionary matching + Combined hints
    - Phase 1B: + Semantic similarity (future)
    
    Design Decision: Composition over inheritance для flexibility.
    """
    
    def __init__(self, cdb_path: Path, config_path: Path = None):
        """
        Args:
            cdb_path: Path to CDB .dat file
            config_path: Optional config.json (або використає cdb.config)
        """
        print("🚀 Ініціалізація CustomCAT Phase 1A...")
        
        # Завантаження CDB
        self.cdb = CDB.load(str(cdb_path))
        print(f"✅ CDB завантажено: {len(self.cdb.cui2names)} CUI")
        
        # Config
        if config_path:
            self.config = Config.load(str(config_path))
        else:
            self.config = self.cdb.config
        
        # Ініціалізація base CAT (без vocab для Phase 1A)
        self.cat = CAT(cdb=self.cdb, config=self.config, vocab=None)
        self.cat.spacy_cat.train = False  # Критично: disable training
        
        # Додавання custom pipeline component
        self.combined_matcher = CombinedHintMatcher(self.cdb, max_gap=3)
        
        # Register custom component у spaCy pipeline
        if not Doc.has_extension('cui'):
            Doc.set_extension('cui', default=None)
        if not Span.has_extension('cui'):
            Span.set_extension('cui', default=None)
        if not Span.has_extension('pattern'):
            Span.set_extension('pattern', default=None)
        
        # Додаємо ПЕРЕД MedCAT NER
        self.cat.nlp.add_pipe(
            'combined_hint_matcher',
            before='ner',
            component=self.combined_matcher
        )
        
        print("✅ CustomCAT готовий до використання")
    
    def extract_entities(self, text: str, only_cui: bool = False) -> Dict:
        """
        Wrapper навколо cat.get_entities з додатковою обробкою.
        
        Args:
            text: Клінічний текст
            only_cui: Повертати тільки CUI (True) або повну інфо (False)
        
        Returns:
            Dict з entities у MedCAT format + custom metadata
        """
        # Standard MedCAT extraction
        result = self.cat.get_entities(text, only_cui=only_cui)
        
        # Додаємо metadata про combined hints
        for ent_id, entity in result.get('entities', {}).items():
            # Check чи це combined hint match
            # (в production додати більш sophisticated tracking)
            pass
        
        return result
    
    def batch_process(
        self, 
        texts: List[str], 
        batch_size: int = 32
    ) -> List[Dict]:
        """
        Batch processing для ефективності.
        
        Args:
            texts: Список клінічних текстів
            batch_size: Розмір батчу для processing
        
        Returns:
            List[Dict] results для кожного тексту
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            for text in batch:
                result = self.extract_entities(text)
                results.append(result)
        
        return results

def main():
    """Quick test CustomCAT."""
    cdb_path = Path('models/phase1a_dictionary/custom_cdb_v2.dat')
    
    cat = CustomCAT(cdb_path)
    
    # Test text
    test_text = """
    Patient prescribed metformin 500mg orally twice daily.
    Administered ceftriaxone intranasally.
    History of diabetes mellitus and hypertension.
    """
    
    entities = cat.extract_entities(test_text)
    
    print(f"\n🔬 Test Results:")
    print(f"Text: {test_text[:100]}...")
    print(f"Entities found: {len(entities.get('entities', {}))}")
    
    for ent_id, ent in list(entities.get('entities', {}).items())[:5]:
        print(f"\n  - {ent['source_value']}")
        print(f"    CUI: {ent['cui']}")
        print(f"    Confidence: {ent['acc']:.3f}")

if __name__ == '__main__':
    main()
```

**Success Criteria:**
- ✅ CAT instance ініціалізується успішно
- ✅ Combined hint matcher знаходить gap-tolerant patterns
- ✅ Batch processing працює для 10+ documents

**Time Estimate:** 3-4 дні

---

### Тиждень 3: Validation на Sample Clinical Notes

#### 1.4 Testing Framework & Metrics

**Мета:** Створити reproducible validation framework для оцінки Phase 1A performance.

**Implementation:**

```python
"""
scripts/validate_phase1a.py
Validation framework для Phase 1A моделі
"""

from pathlib import Path
from typing import Dict, List, Tuple
import json
from src.custom_cat_v2 import CustomCAT

class ValidationFramework:
    """
    Framework для systematic evaluation Phase 1A model.
    
    Metrics:
    - Entity-level: Precision, Recall, F1
    - Span-level: Exact match accuracy
    - Type-level: Cluster accuracy
    """
    
    def __init__(self, cat: CustomCAT, test_docs_path: Path):
        self.cat = cat
        self.test_docs = self._load_test_docs(test_docs_path)
        
    def _load_test_docs(self, path: Path) -> List[Dict]:
        """Завантаження test documents."""
        # Format: список dictionaries з 'text' та optional 'annotations'
        with open(path, 'r') as f:
            return json.load(f)
    
    def run_validation(self) -> Dict:
        """
        Запуск повної валідації.
        
        Returns:
            Dict з метриками та детальними результатами
        """
        results = {
            'total_docs': len(self.test_docs),
            'entities_found': 0,
            'avg_entities_per_doc': 0,
            'coverage_by_cluster': {},
            'sample_outputs': []
        }
        
        all_entities = []
        
        for doc_idx, doc in enumerate(self.test_docs):
            text = doc['text']
            entities = self.cat.extract_entities(text)
            
            doc_entity_count = len(entities.get('entities', {}))
            all_entities.append(entities)
            results['entities_found'] += doc_entity_count
            
            # Sample output (перші 3 documents)
            if doc_idx < 3:
                results['sample_outputs'].append({
                    'text': text[:200] + '...',
                    'entity_count': doc_entity_count,
                    'entities': [
                        {
                            'text': e['source_value'],
                            'cui': e['cui'],
                            'type': e.get('types', ['UNKNOWN'])[0] if e.get('types') else 'UNKNOWN',
                            'confidence': e['acc']
                        }
                        for e in list(entities.get('entities', {}).values())[:5]
                    ]
                })
        
        results['avg_entities_per_doc'] = results['entities_found'] / results['total_docs']
        
        return results
    
    def generate_report(self, results: Dict, output_path: Path):
        """Генерація validation report."""
        
        report_lines = [
            "# Phase 1A Validation Report",
            "",
            "## Summary Statistics",
            f"- Total documents tested: {results['total_docs']}",
            f"- Total entities found: {results['entities_found']}",
            f"- Avg entities per document: {results['avg_entities_per_doc']:.2f}",
            "",
            "## Sample Outputs",
            ""
        ]
        
        for idx, sample in enumerate(results['sample_outputs'], 1):
            report_lines.append(f"### Document {idx}")
            report_lines.append(f"**Text:** {sample['text']}")
            report_lines.append(f"**Entities found:** {sample['entity_count']}")
            report_lines.append("")
            
            for ent in sample['entities']:
                report_lines.append(
                    f"- `{ent['text']}` → **{ent['cui']}** "
                    f"(type: {ent['type']}, conf: {ent['confidence']:.3f})"
                )
            report_lines.append("")
        
        # Збереження
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"📊 Validation report saved: {output_path}")

def main():
    # Підготувати test documents
    test_docs_path = Path('data/test_clinical_notes.json')
    
    # Якщо немає готових - створюємо synthetic test set
    if not test_docs_path.exists():
        print("⚠️ Test documents not found. Creating synthetic test set...")
        synthetic_docs = [
            {"text": "Patient prescribed metformin 500mg orally twice daily for type 2 diabetes."},
            {"text": "Administered ceftriaxone 1g intravenously for pneumonia treatment."},
            {"text": "History of hypertension managed with lisinopril 10mg daily."},
            # ... додати більше synthetic examples
        ]
        test_docs_path.parent.mkdir(parents=True, exist_ok=True)
        with open(test_docs_path, 'w') as f:
            json.dump(synthetic_docs, f, indent=2)
    
    # Завантажити model
    cdb_path = Path('models/phase1a_dictionary/custom_cdb_v2.dat')
    cat = CustomCAT(cdb_path)
    
    # Запуск validation
    framework = ValidationFramework(cat, test_docs_path)
    results = framework.run_validation()
    
    # Генерація report
    report_path = Path('reports/phase1a_validation.md')
    framework.generate_report(results, report_path)
    
    print("\n✅ Validation completed!")
    print(f"📊 Results summary:")
    print(f"   - Documents: {results['total_docs']}")
    print(f"   - Entities: {results['entities_found']}")
    print(f"   - Avg per doc: {results['avg_entities_per_doc']:.2f}")

if __name__ == '__main__':
    main()
```

**Success Criteria Phase 1A:**
- ✅ Entities detected у >= 80% test documents
- ✅ Avg entities per document >= 5
- ✅ Zero crashes на різноманітних текстах
- ✅ Reproducible metrics для порівняння з Phase 1B

**Time Estimate:** 2-3 дні

---

## Phase 1A Summary & Go/No-Go Decision Point

**Deliverables:**
1. ✅ MedCAT v2 CDB з 7,219 concepts
2. ✅ CustomCAT з gap-tolerant combined hints
3. ✅ Validation framework з metrics
4. ✅ Baseline performance report

**Expected Performance (Phase 1A):**
- Precision: 0.75-0.85 (high - low false positives через 0% ambiguity)
- Recall: 0.70-0.80 (medium - обмежений synonym coverage)
- F1: 0.78-0.82
- Processing speed: 10-50 docs/second

**Decision Point:**

**IF F1 >= 0.78 AND Precision >= 0.80:**
→ Phase 1A sufficient для багатьох use cases
→ Consider Phase 1B optional enhancement
→ Proceed до production packaging

**IF F1 < 0.78 OR Recall < 0.70:**
→ Phase 1B (semantic similarity) обов'язкова
→ Investigate root causes (hint quality? combined patterns?)

---

## Phase 1B: Semantic Similarity Enhancement [CONDITIONAL]

**Стратегічне обґрунтування:** Якщо Phase 1A shows F1 < 0.78, додаємо Bio_ClinicalBERT embeddings для semantic matching варіацій не covered hints.

**Timeline:** +2-3 тижні after Phase 1A

### Тиждень 4-5: Semantic Layer Integration

#### 1.5 Bio_ClinicalBERT Embeddings Setup

**Architecture Decision:** Використовуємо `transformers` Bio_ClinicalBERT для generating concept embeddings + faiss для efficient similarity search.

**Implementation:**

```python
"""
src/semantic_layer.py
Semantic similarity layer for Phase 1B
"""

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from pathlib import Path
import pickle
from typing import Dict, List, Tuple
import faiss

class SemanticMatcher:
    """
    Semantic similarity matching using Bio_ClinicalBERT.
    
    Architecture:
    1. Pre-compute embeddings для всіх CUI names
    2. Build faiss index для efficient search
    3. During inference: embed detected spans → find similar CUI
    
    Advantage: Catches variations not in hints
    Example: "type 2 sugar disease" → "type 2 diabetes" (semantic match)
    """
    
    def __init__(
        self, 
        cdb: 'CDB',
        model_name: str = 'emilyalsentzer/Bio_ClinicalBERT',
        similarity_threshold: float = 0.75
    ):
        self.cdb = cdb
        self.threshold = similarity_threshold
        
        print(f"🧠 Loading Bio_ClinicalBERT: {model_name}")
        
        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # Inference mode
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"✅ Model loaded on {self.device}")
        
        # Pre-compute embeddings
        self.cui_embeddings = None
        self.cui_list = None
        self.faiss_index = None
        
    def _embed_text(self, text: str) -> np.ndarray:
        """Генерація embedding для тексту."""
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=128,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embedding[0]  # Shape: (768,)
    
    def build_cui_index(self, cache_path: Path = None):
        """
        Pre-compute embeddings для всіх CUI names та build faiss index.
        
        This is computationally expensive - cache results.
        """
        if cache_path and cache_path.exists():
            print(f"📦 Loading cached embeddings from {cache_path}")
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
                self.cui_list = cached['cui_list']
                self.cui_embeddings = cached['embeddings']
        else:
            print(f"🔧 Building CUI embeddings index (це займе час)...")
            
            cui_list = []
            embeddings = []
            
            for cui, names in self.cdb.cui2names.items():
                # Використовуємо preferred name для embedding
                pref_name = self.cdb.cui2preferred_name.get(cui, list(names)[0])
                
                embedding = self._embed_text(pref_name)
                
                cui_list.append(cui)
                embeddings.append(embedding)
            
            self.cui_list = cui_list
            self.cui_embeddings = np.array(embeddings)  # Shape: (N_cui, 768)
            
            # Cache
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, 'wb') as f:
                    pickle.dump({
                        'cui_list': self.cui_list,
                        'embeddings': self.cui_embeddings
                    }, f)
                print(f"💾 Embeddings cached to {cache_path}")
        
        # Build faiss index
        dimension = self.cui_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product (cosine sim)
        
        # Normalize embeddings для cosine similarity
        faiss.normalize_L2(self.cui_embeddings)
        self.faiss_index.add(self.cui_embeddings)
        
        print(f"✅ FAISS index built: {len(self.cui_list)} CUI")
    
    def find_similar_cui(
        self, 
        text: str, 
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Знайти top-k CUI найбільш семантично схожих до text.
        
        Returns:
            List[(cui, similarity_score)]
        """
        if self.faiss_index is None:
            raise RuntimeError("Index not built. Call build_cui_index() first.")
        
        # Embed query text
        query_emb = self._embed_text(text).reshape(1, -1)
        faiss.normalize_L2(query_emb)
        
        # Search
        similarities, indices = self.faiss_index.search(query_emb, top_k)
        
        results = [
            (self.cui_list[idx], float(sim))
            for idx, sim in zip(indices[0], similarities[0])
            if sim >= self.threshold
        ]
        
        return results

class EnhancedCAT:
    """
    Phase 1B: CustomCAT + Semantic Matching.
    
    Combines:
    - Phase 1A dictionary matching (fast, high precision)
    - Semantic similarity (slower, high recall)
    """
    
    def __init__(self, phase1a_cat: CustomCAT, semantic_matcher: SemanticMatcher):
        self.base_cat = phase1a_cat
        self.semantic = semantic_matcher
    
    def extract_entities(self, text: str) -> Dict:
        """
        Hybrid extraction:
        1. Phase 1A dictionary matching
        2. For unmatched spans → semantic similarity
        3. Merge results
        """
        # Phase 1A
        phase1a_results = self.base_cat.extract_entities(text)
        
        # TODO: Implement semantic enhancement
        # - Identify unmatched clinical terms (via spaCy NER?)
        # - Run semantic matching on unmatched spans
        # - Merge з Phase 1A results (resolve overlaps)
        
        return phase1a_results
```

**Success Criteria Phase 1B:**
- ✅ F1 increase >= 0.05 порівняно з Phase 1A
- ✅ Recall improvement >= 0.08
- ✅ Processing speed acceptable (<5s per document)

**Time Estimate:** 2-3 тижні

---

## Phase 2: Production Readiness

### Тиждень 6-8: Packaging & Deployment

#### 2.1 Model Packaging

**Мета:** Створити production-ready package для distribution.

```python
"""
scripts/create_model_pack.py
Package Phase 1A/1B model як single .zip file
"""

from pathlib import Path
from src.custom_cat_v2 import CustomCAT

def create_production_pack(
    cdb_path: Path,
    output_path: Path,
    include_semantic: bool = False
):
    """
    Створення єдиного model pack.
    
    Includes:
    - CDB
    - Config
    - Custom components code
    - (Optional) Semantic embeddings cache
    - README з usage instructions
    """
    # TODO: Implementation
    pass
```

#### 2.2 Integration Testing

**Мета:** End-to-end testing на 10-50 real clinical notes.

#### 2.3 Documentation

**Мета:** User-facing documentation + API reference.

---

## Risk Mitigation Strategies

### Risk 1: Low Recall через Insufficient Hints

**Mitigation:**
- Phase 1A: Validate hint quality early (Week 1)
- If recall < 0.70 → Proceed до Phase 1B immediately
- Active learning: Annotate missed entities → add as hints

### Risk 2: Combined Hints Gap Tolerance Too Restrictive

**Mitigation:**
- Configurable `max_gap` parameter (default: 3)
- A/B testing з різними gap values
- Manual review of gap patterns у test documents

### Risk 3: Bio_ClinicalBERT Embeddings Too Slow

**Mitigation:**
- Pre-compute all CUI embeddings (one-time cost)
- Use faiss GPU for inference
- Batch processing для multiple documents
- Cache frequent queries

---

## Success Metrics & KPIs

### Phase 1A
- ✅ F1 >= 0.78
- ✅ Precision >= 0.80
- ✅ Processing >= 10 docs/sec
- ✅ Zero false negatives for exact keyword matches

### Phase 1B (якщо потрібно)
- ✅ F1 >= 0.83
- ✅ Recall increase >= 0.08 vs Phase 1A
- ✅ Processing >= 2 docs/sec

### Production
- ✅ Stable performance на diverse clinical texts
- ✅ Reproducible results
- ✅ Clear documentation
- ✅ Easy integration у downstream applications

---

## Budget & Resource Allocation

**Phase 1A (2-3 тижні):**
- 1 Senior ML Engineer: $15K-$20K
- Compute (CPU): $500
- **Total: $20K**

**Phase 1B (якщо потрібно, +2-3 тижні):**
- 1 Senior ML Engineer: $15K-$20K
- Compute (GPU для embeddings): $2K-$3K
- **Total: $20K**

**Phase 2 (2 тижні):**
- Documentation + packaging: $10K
- **Total: $10K**

**Grand Total: $40K-$50K для Phase 1A+2, або $60K-$70K якщо Phase 1B потрібна**

Це значно нижче initial $60K-$100K estimate, завдяки 0% ambiguity онтології.

---

## Next Steps

**Week 1 Action Items:**
1. ✅ Run `scripts/transform_to_medcat_format.py`
2. ✅ Validate CSV transformation quality
3. ✅ Run `scripts/create_cdb_v2.py`
4. ✅ Quick smoke test з CustomCAT на 2-3 synthetic texts

**Decision Point (End of Week 3):**
- Review Phase 1A validation results
- GO/NO-GO на Phase 1B
- Plan production deployment timeline

**Long-term Vision:**
- Модульна архітектура дозволяє easy swap semantic backends
- Додавання нових concepts без retraining
- Scale до 70K+ concepts без architectural changes
- Foundation для future Meta-CAT integration (context attributes)

---

## Appendix: Alternative Architectures Considered

### Alternative 1: Pure Semantic (Bio_ClinicalBERT Only)

**Rejected Rationale:** 
- Too slow для large-scale processing
- Lower precision через semantic ambiguity
- Requires expensive GPU infrastructure

### Alternative 2: scispaCy Instead of MedCAT

**Rejected Rationale:**
- Supervised approach requires labeled data (немає у нас)
- Less flexible для custom ontology integration
- Lower performance на medical concepts (F1 ≈ 0.65 vs MedCAT 0.80)

### Alternative 3: LLM-based (GPT-4, Claude)

**Rejected Rationale:**
- API costs prohibitive для batch processing
- Latency unacceptable
- Hallucinations risk
- No offline capability

**Selected Architecture (Hybrid Dictionary + Semantic) Balance:**
- ✅ Fast dictionary matching для majority cases
- ✅ Semantic fallback для edge cases
- ✅ Modular design для future enhancements
- ✅ Cost-effective infrastructure requirements

---

**Document Version:** 1.0
**Last Updated:** 2025-01-XX
**Status:** READY FOR IMPLEMENTATION