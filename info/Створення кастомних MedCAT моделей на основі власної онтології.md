# Створення кастомних MedCAT моделей на основі власної онтології

## Вступ

**MedCAT (Medical Concept Annotation Toolkit)** — це відкрита система для розпізнавання та зв'язування медичних сутностей (Named Entity Recognition + Linking) у клінічних текстах. Цей гайд показує, як створити повністю кастомну модель на основі вашої ієрархічної медичної онтології з кластерами, keywords і текстовими патернами.

---

## 1. Архітектура MedCAT та підготовка онтології

### 1.1 Основні компоненти MedCAT

MedCAT складається з трьох основних компонентів:

**CDB (Concept Database)** — база даних концептів, що містить:
- CUI (Concept Unique Identifiers) для кожного медичного терміну
- Назви концептів та їх синоніми
- Type IDs (семантичні типи, аналогічні UMLS TUI)
- Контекстні векторні представлення
- Метадані та ієрархічні зв'язки

**Vocab (Vocabulary)** — словник слів з embeddings для контекстного аналізу

**Meta-CAT** — модулі для екстракції контекстних атрибутів (status, severity, temporality тощо)

### 1.2 Структура даних CDB

CDB зберігає концепти у вигляді словників Python:

```python
# Основні словники CDB
cdb.name2cuis: Dict[str, List[str]]  # Назва → список CUI
cdb.cui2names: Dict[str, Set[str]]  # CUI → усі назви
cdb.cui2preferred_name: Dict[str, str]  # CUI → preferred назва
cdb.cui2type_ids: Dict[str, Set[str]]  # CUI → Type IDs
cdb.cui2tags: Dict[str, List[str]]  # CUI → теги для ієрархії
cdb.cui2count_train: Dict[str, int]  # Кількість навчальних прикладів
cdb.addl_info['type_id2name']: Dict[str, str]  # Type ID → назва типу
```

### 1.3 Адаптація вашої онтології

Ваша онтологія має структуру: **Кластери → Keywords → Хінти/Патерни**

**Приклад mapping для вашої структури:**

```python
# Core Actions and Processes
cluster_mapping = {
    'Action': {
        'type_id': 'T_ACTION',
        'keywords': {
            'done': {'cui': 'ACTION_DONE', 'patterns': ['completed', 'finished', 'done']},
            'started': {'cui': 'ACTION_STARTED', 'patterns': ['initiated', 'began', 'started']},
            'canceled': {'cui': 'ACTION_CANCELED', 'patterns': ['cancelled', 'stopped', 'discontinued']}
        }
    },
    'Drug': {
        'type_id': 'T_DRUG',
        'keywords': {
            'decided': {'cui': 'DRUG_DECIDED', 'patterns': ['prescribed', 'decided to use', 'planned']},
            'taken': {'cui': 'DRUG_TAKEN', 'patterns': ['administered', 'given', 'took', 'taking']},
            'not_taken': {'cui': 'DRUG_NOT_TAKEN', 'patterns': ['refused', 'did not take', 'declined']}
        }
    },
    'Symptoms': {
        'type_id': 'T_SYMPTOM',
        'keywords': {
            'yes': {'cui': 'SYMPTOM_PRESENT', 'patterns': ['experiencing', 'has', 'presents with']},
            'no': {'cui': 'SYMPTOM_ABSENT', 'patterns': ['no', 'denies', 'without']},
            'unknown': {'cui': 'SYMPTOM_UNKNOWN', 'patterns': ['unclear', 'uncertain', 'unknown']}
        }
    }
}

# Symptoms localization
symptom_locations = {
    'jaw': 'SYMP_LOC_JAW',
    'chest': 'SYMP_LOC_CHEST',
    'back': 'SYMP_LOC_BACK',
    # ... всі інші локалізації
}

# Symptoms severity
symptom_severity = {
    'mild': 'SEVERITY_MILD',
    'moderate': 'SEVERITY_MODERATE',
    'severe': 'SEVERITY_SEVERE'
}
```

---

## 2. Створення CDB програмно

### 2.1 Базовий метод: CSV імпорт

**Формат CSV файлу:**

```csv
cui,name,ontologies,name_status,type_ids,description
ACTION_DONE,completed,CUSTOM_ONTOLOGY,P,T_ACTION,Action was completed
ACTION_DONE,done,,A,,
ACTION_DONE,finished,,A,,
DRUG_TAKEN,administered,CUSTOM_ONTOLOGY,P,T_DRUG,Drug was administered to patient
DRUG_TAKEN,given|took|taking,,A,,
SYMPTOM_PRESENT,experiencing,CUSTOM_ONTOLOGY,P,T_SYMPTOM,Patient experiencing symptom
SYMP_LOC_CHEST,chest pain,CUSTOM_ONTOLOGY,P,T_SYMPTOM_LOC,Pain localized to chest
SYMP_LOC_CHEST,thoracic pain|chest discomfort,,A,,
SEVERITY_SEVERE,severe,CUSTOM_ONTOLOGY,P,T_SEVERITY,High severity level
```

**Код для створення CDB з CSV:**

```python
from medcat.cdb_maker import CDBMaker
from medcat.config import Config
import pandas as pd

# Крок 1: Підготовка даних вашої онтології
def prepare_ontology_csv(cluster_mapping, output_path='ontology.csv'):
    """Конвертує вашу онтологію у CSV формат для MedCAT"""
    rows = []
    
    for cluster_name, cluster_data in cluster_mapping.items():
        type_id = cluster_data['type_id']
        
        for keyword, keyword_data in cluster_data['keywords'].items():
            cui = keyword_data['cui']
            patterns = keyword_data['patterns']
            
            # Основна назва (Primary)
            rows.append({
                'cui': cui,
                'name': keyword,
                'ontologies': 'CUSTOM_ONTOLOGY',
                'name_status': 'P',
                'type_ids': type_id,
                'description': f'{cluster_name} - {keyword}'
            })
            
            # Патерни як синоніми
            for pattern in patterns:
                rows.append({
                    'cui': cui,
                    'name': pattern,
                    'ontologies': '',
                    'name_status': 'A',
                    'type_ids': '',
                    'description': ''
                })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return output_path

# Крок 2: Створення CDB
config = Config()
config.cdb_maker.name_max_words = 20
config.cdb_maker.min_letters_required = 2

cdb_maker = CDBMaker(config=config)

# Підготовка CSV
csv_path = prepare_ontology_csv(cluster_mapping)

# Створення CDB
cdb = cdb_maker.prepare_csvs(
    csv_paths=[csv_path],
    sep=',',
    encoding='utf-8',
    full_build=True
)

# Крок 3: Додавання Type ID описів
cdb.addl_info['type_id2name']['T_ACTION'] = 'Clinical Action'
cdb.addl_info['type_id2name']['T_DRUG'] = 'Drug Administration'
cdb.addl_info['type_id2name']['T_SYMPTOM'] = 'Clinical Symptom'
cdb.addl_info['type_id2name']['T_SYMPTOM_LOC'] = 'Symptom Localization'
cdb.addl_info['type_id2name']['T_SEVERITY'] = 'Severity Level'

# Збереження CDB
cdb.save('custom_medical_cdb.dat')
print(f"CDB створено: {len(cdb.cui2names)} концептів")
```

### 2.2 Програмне додавання концептів

Для більш складних сценаріїв:

```python
from medcat.cdb import CDB
from medcat.preprocessing.cleaners import prepare_name
from medcat.pipe import Pipe
from medcat.preprocessing.tokenizers import spacy_split_all

# Ініціалізація
config = Config()
cdb = CDB(config=config)
pipe = Pipe(tokenizer=spacy_split_all, config=config)

def add_concept_with_patterns(cdb, pipe, cui, primary_name, patterns, 
                              type_id, cluster_name, description=''):
    """Додає концепт з патернами до CDB"""
    # Всі назви (primary + patterns)
    all_names = [primary_name] + patterns
    
    # Підготовка назв
    names_dict = {}
    for name in all_names:
        prepared = prepare_name(
            name=name,
            pipe=pipe,
            config=config
        )
        names_dict.update(prepared)
    
    # Додавання до CDB
    cdb._add_concept(
        cui=cui,
        names=names_dict,
        ontologies={'CUSTOM_ONTOLOGY'},
        name_status='P' if len(names_dict) == 1 else 'A',
        type_ids={type_id},
        description=description,
        full_build=True
    )
    
    # Додавання тегів для ієрархії
    if cui not in cdb.cui2tags:
        cdb.cui2tags[cui] = []
    cdb.cui2tags[cui].append(f'cluster:{cluster_name}')
    cdb.cui2tags[cui].append(f'type:{type_id}')

# Приклад використання
for cluster_name, cluster_data in cluster_mapping.items():
    for keyword, keyword_data in cluster_data['keywords'].items():
        add_concept_with_patterns(
            cdb=cdb,
            pipe=pipe,
            cui=keyword_data['cui'],
            primary_name=keyword,
            patterns=keyword_data['patterns'],
            type_id=cluster_data['type_id'],
            cluster_name=cluster_name,
            description=f'{cluster_name}: {keyword}'
        )

cdb.save('programmatic_cdb.dat')
```

### 2.3 Комплексна підготовка для всієї онтології

**Повний робочий приклад:**

```python
import json
from medcat.cdb import CDB
from medcat.cdb_maker import CDBMaker
from medcat.config import Config
import pandas as pd

# Ваша повна онтологія
full_ontology = {
    'Core Actions and Processes': {
        'Action': {
            'type_id': 'T001',
            'keywords': {
                'done': ['completed', 'finished', 'done'],
                'started': ['initiated', 'began', 'started'],
                'decided': ['planned', 'scheduled', 'decided'],
                'canceled': ['cancelled', 'stopped', 'discontinued'],
                'refusal': ['refused', 'declined', 'rejected']
            }
        },
        'Drug': {
            'type_id': 'T002',
            'keywords': {
                'decided': ['prescribed', 'decided to prescribe'],
                'taken': ['administered', 'given', 'took'],
                'not_taken': ['refused', 'did not take', 'declined'],
                'increased': ['dose increased', 'increased dosage'],
                'decreased': ['dose decreased', 'reduced dosage']
            }
        }
    },
    'Diagnostics and Examinations': {
        'Finding': {
            'type_id': 'T003',
            'keywords': {
                'yes': ['positive', 'found', 'detected'],
                'no': ['negative', 'not found', 'absent']
            }
        },
        'Problem': {
            'type_id': 'T004',
            'keywords': {
                'exist': ['diagnosed with', 'has', 'suffers from'],
                'suspected': ['possible', 'suspected', 'rule out'],
                'not_exist': ['no evidence of', 'ruled out']
            }
        }
    },
    'Symptoms': {
        'Symptoms': {
            'type_id': 'T005',
            'keywords': {
                'yes': ['experiencing', 'reports', 'complains of'],
                'no': ['denies', 'no', 'without'],
                'unknown': ['unclear', 'uncertain', 'not sure']
            }
        },
        'Symptoms_Severity': {
            'type_id': 'T006',
            'keywords': {
                'mild': ['mild', 'slight', 'minor'],
                'moderate': ['moderate', 'medium'],
                'severe': ['severe', 'intense', 'extreme']
            }
        },
        'Symptoms_Localization': {
            'type_id': 'T007',
            'keywords': {
                'chest': ['chest', 'thorax', 'thoracic'],
                'back': ['back', 'dorsal'],
                'head': ['head', 'cranial', 'cephalic'],
                'abdomen': ['abdomen', 'abdominal', 'belly']
                # ... додайте всі локалізації
            }
        }
    }
}

def create_complete_cdb(ontology, output_dir='./models'):
    """Створює повний CDB з вашої онтології"""
    
    # Крок 1: Генерація CUI та підготовка даних
    csv_data = []
    cui_counter = 1
    
    for category, subcategories in ontology.items():
        for subcategory, subcat_data in subcategories.items():
            type_id = subcat_data['type_id']
            
            for keyword, patterns in subcat_data['keywords'].items():
                cui = f'CUSTOM_{cui_counter:06d}'
                cui_counter += 1
                
                # Primary name
                csv_data.append({
                    'cui': cui,
                    'name': keyword,
                    'ontologies': 'CUSTOM',
                    'name_status': 'P',
                    'type_ids': type_id,
                    'description': f'{category} / {subcategory} / {keyword}'
                })
                
                # Patterns as synonyms
                for pattern in patterns:
                    csv_data.append({
                        'cui': cui,
                        'name': pattern,
                        'ontologies': '',
                        'name_status': 'A',
                        'type_ids': '',
                        'description': ''
                    })
    
    # Збереження CSV
    df = pd.DataFrame(csv_data)
    csv_path = f'{output_dir}/ontology.csv'
    df.to_csv(csv_path, index=False)
    
    # Крок 2: Створення CDB
    config = Config()
    config.general['spacy_model'] = 'en_core_web_md'  # або 'uk_core_news_sm'
    
    cdb_maker = CDBMaker(config=config)
    cdb = cdb_maker.prepare_csvs([csv_path], sep=',', full_build=True)
    
    # Крок 3: Додавання метаданих Type IDs
    type_id_names = {
        'T001': 'Clinical_Action',
        'T002': 'Drug_Administration',
        'T003': 'Clinical_Finding',
        'T004': 'Medical_Problem',
        'T005': 'Clinical_Symptom',
        'T006': 'Symptom_Severity',
        'T007': 'Symptom_Localization'
    }
    
    for tid, tname in type_id_names.items():
        cdb.addl_info['type_id2name'][tid] = tname
    
    # Збереження
    cdb.save(f'{output_dir}/custom_cdb.dat')
    
    print(f"✓ CDB створено: {len(cdb.cui2names)} концептів")
    print(f"✓ Type IDs: {len(type_id_names)}")
    print(f"✓ Збережено: {output_dir}/custom_cdb.dat")
    
    return cdb

# Використання
cdb = create_complete_cdb(full_ontology)
```

---

## 3. Навчання моделі: Unsupervised Training

### 3.1 Підготовка даних

Unsupervised training використовує великі обсяги неанотованих клінічних текстів для навчання контекстних векторів.

```python
from medcat.vocab import Vocab
from medcat.cat import CAT

# Крок 1: Підготовка клінічних текстів
clinical_texts = [
    "Patient presents with severe chest pain radiating to left arm",
    "Started metformin 500mg BID for type 2 diabetes",
    "No history of hypertension or cardiac disease",
    "CT scan shows pneumonia in right lower lobe",
    # ... тисячі текстів з вашої бази
]

# Крок 2: Створення Vocabulary
vocab = Vocab()

# Додавання слів з корпусу
for text in clinical_texts:
    words = text.lower().split()
    for word in words:
        vocab.add_word(word)

# Або завантаження готового vocab
# vocab = Vocab.load('path/to/vocab.dat')

# Крок 3: Завантаження CDB
from medcat.cdb import CDB
cdb = CDB.load('./models/custom_cdb.dat')

# Крок 4: Конфігурація
config = Config()
config.general['spacy_model'] = 'en_core_web_md'
config.linking['train'] = True
config.linking['similarity_threshold'] = 0.3
config.linking['train_count_threshold'] = 10

# Крок 5: Створення CAT
cat = CAT(cdb=cdb, config=config, vocab=vocab)

# Крок 6: Unsupervised навчання
print("Початок unsupervised навчання...")
cat.train(clinical_texts)

print("Навчання завершено!")
cat.create_model_pack('./models/trained_unsupervised.zip')
```

### 3.2 Налаштування параметрів навчання

```python
# Конфігурація для українського тексту
config.general['spacy_model'] = 'uk_core_news_sm'  # або uk_core_news_trf

# Linking параметри
config.linking['similarity_threshold'] = 0.25  # Нижче = більше recall
config.linking['similarity_threshold_type'] = 'dynamic'  # adaptive threshold
config.linking['train_count_threshold'] = 10  # мінімум прикладів для навчання
config.linking['disamb_length_limit'] = 5  # довжина для disambiguation
config.linking['prefer_primary_name'] = 0.35  # перевага primary names
config.linking['negative_probability'] = 0.5  # negative sampling

# NER параметри
config.ner['min_name_len'] = 2  # мінімальна довжина терміну
config.ner['upper_case_limit_len'] = 3  # обмеження для upper case
config.ner['try_reverse_word_order'] = True
```

### 3.3 Інкрементальне навчання

```python
# Завантаження базової моделі
cat = CAT.load_model_pack('./models/trained_unsupervised.zip')

# Додаткове навчання на новій партії текстів
new_texts = load_new_clinical_texts()
cat.train(new_texts)

# Збереження оновленої моделі
cat.create_model_pack('./models/trained_v2.zip')
```

---

## 4. Supervised Training з текстовими патернами

### 4.1 Формат анотацій

MedCAT supervised training використовує JSON формат:

```json
{
  "projects": [{
    "name": "custom_training",
    "id": "proj_001",
    "documents": [{
      "name": "doc_1",
      "text": "Patient has severe chest pain",
      "id": "doc_001",
      "annotations": [{
        "cui": "CUSTOM_000123",
        "value": "severe",
        "start": 12,
        "end": 18,
        "correct": true,
        "deleted": false,
        "alternative": false,
        "killed": false,
        "manually_created": true,
        "acc": 1.0,
        "id": "ann_1",
        "meta_anns": {
          "Severity": {
            "value": "Severe",
            "confidence": 1.0,
            "name": "Severity"
          }
        }
      }, {
        "cui": "CUSTOM_000456",
        "value": "chest pain",
        "start": 19,
        "end": 29,
        "correct": true,
        "deleted": false,
        "alternative": false,
        "killed": false,
        "manually_created": true,
        "acc": 1.0,
        "id": "ann_2",
        "meta_anns": {
          "Localization": {
            "value": "Chest",
            "confidence": 1.0
          }
        }
      }]
    }]
  }]
}
```

### 4.2 Конвертація патернів у навчальні дані

**Автоматична генерація анотацій з ваших хінтів:**

```python
import json
import re

def generate_training_from_patterns(ontology_with_patterns, texts, output_path):
    """
    Генерує supervised training annotations з текстових патернів
    
    Args:
        ontology_with_patterns: словник з CUI → patterns
        texts: список текстів для анотування
        output_path: шлях для збереження JSON
    """
    training_data = {
        'projects': [{
            'name': 'pattern_based_training',
            'id': 'pattern_proj',
            'documents': []
        }]
    }
    
    # Підготовка pattern mapping
    pattern_to_cui = {}
    for category, subcats in ontology_with_patterns.items():
        for subcat, data in subcats.items():
            for keyword, patterns in data['keywords'].items():
                cui = data.get('cui_prefix', 'CUSTOM') + '_' + keyword.upper()
                for pattern in patterns:
                    pattern_to_cui[pattern.lower()] = {
                        'cui': cui,
                        'keyword': keyword,
                        'category': category
                    }
    
    # Анотування текстів
    for doc_idx, text in enumerate(texts):
        annotations = []
        text_lower = text.lower()
        
        # Пошук всіх патернів у тексті
        for pattern, pattern_data in pattern_to_cui.items():
            # Регулярний вираз для пошуку з boundary
            regex = r'\b' + re.escape(pattern) + r'\b'
            
            for match in re.finditer(regex, text_lower):
                start, end = match.span()
                
                annotation = {
                    'cui': pattern_data['cui'],
                    'value': text[start:end],  # Оригінальний регістр
                    'start': start,
                    'end': end,
                    'correct': True,
                    'deleted': False,
                    'alternative': False,
                    'killed': False,
                    'manually_created': True,
                    'acc': 1.0,
                    'id': f'ann_{doc_idx}_{len(annotations)}',
                    'meta_anns': {}
                }
                annotations.append(annotation)
        
        if annotations:  # Додавати тільки документи з анотаціями
            document = {
                'name': f'doc_{doc_idx}',
                'text': text,
                'id': str(doc_idx),
                'annotations': annotations
            }
            training_data['projects'][0]['documents'].append(document)
    
    # Збереження
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Згенеровано {len(training_data['projects'][0]['documents'])} документів")
    print(f"✓ Збережено: {output_path}")
    
    return training_data

# Використання
training_texts = [
    "Patient started taking metformin for diabetes",
    "Severe chest pain radiating to left arm",
    "No history of hypertension",
    "CT scan shows pneumonia in right lower lobe"
]

training_json = generate_training_from_patterns(
    full_ontology,
    training_texts,
    './training_data/supervised_annotations.json'
)
```

### 4.3 Supervised навчання

```python
from medcat.cat import CAT

# Завантаження базової моделі
cat = CAT.load_model_pack('./models/trained_unsupervised.zip')

# Supervised training
results = cat.train_supervised_from_json(
    data_path='./training_data/supervised_annotations.json',
    nepochs=10,
    test_size=0.2,
    print_stats=1,
    use_filters=False,
    reset_cui_count=False
)

# Результати
print(f"Precision: {results['precision']:.3f}")
print(f"Recall: {results['recall']:.3f}")
print(f"F1 Score: {results['f1']:.3f}")

# Збереження fine-tuned моделі
cat.create_model_pack('./models/supervised_trained.zip')
```

---

## 5. Meta-CAT: Контекстні атрибути

### 5.1 Що таке Meta-CAT?

Meta-CAT додає контекстне розуміння до розпізнаних сутностей. Для вашої онтології це означає класифікацію атрибутів:

- **Status**: yes/no/unknown/sometimes
- **Severity**: mild/moderate/severe
- **Pattern**: constant/intermittent
- **Temporality**: present/past/future
- **Action Status**: done/decided/started/canceled

### 5.2 Налаштування Meta-CAT моделей

```python
from medcat.meta_cat import MetaCAT
from medcat.config_meta_cat import ConfigMetaCAT
from medcat.tokenizers.meta_cat_tokenizers import TokenizerWrapperBERT
from transformers import BertTokenizerFast

# Крок 1: Підготовка токенізатора
hf_tokenizer = BertTokenizerFast.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
tokenizer = TokenizerWrapperBERT(
    hf_tokenizers=hf_tokenizer,
    max_seq_length=512
)

# Крок 2: Конфігурація для Status (yes/no/unknown)
status_config = ConfigMetaCAT()
status_config.general['category_name'] = 'Status'
status_config.general['category_value2id'] = {
    'yes': 0,
    'no': 1,
    'unknown': 2,
    'sometimes': 3
}
status_config.general['cntx_left'] = 15
status_config.general['cntx_right'] = 10
status_config.general['device'] = 'cuda'

# Model settings
status_config.model['model_name'] = 'bert'
status_config.model['model_variant'] = 'emilyalsentzer/Bio_ClinicalBERT'

# Training settings
status_config.train['nepochs'] = 20
status_config.train['lr'] = 2e-5
status_config.train['batch_size'] = 32
status_config.train['test_size'] = 0.2

# Створення та навчання
meta_status = MetaCAT(tokenizer=tokenizer, config=status_config)
```

### 5.3 Навчання множини Meta-моделей

```python
# Визначення всіх meta-tasks для вашої онтології
meta_tasks_config = {
    'Status': {
        'values': {'yes': 0, 'no': 1, 'unknown': 2, 'sometimes': 3},
        'applies_to': ['Symptoms', 'Finding', 'Problem']
    },
    'Severity': {
        'values': {'mild': 0, 'moderate': 1, 'severe': 2},
        'applies_to': ['Symptoms']
    },
    'ActionStatus': {
        'values': {'done': 0, 'decided': 1, 'started': 2, 'canceled': 3, 'refusal': 4},
        'applies_to': ['Action', 'Drug']
    },
    'Pattern': {
        'values': {'constant': 0, 'intermittent': 1},
        'applies_to': ['Symptoms']
    },
    'Level': {
        'values': {'high': 0, 'normal': 1, 'low': 2, 'not_known': 3},
        'applies_to': ['Lab']
    }
}

def train_all_meta_models(meta_tasks, training_json_path, output_dir):
    """Навчає всі meta-annotation моделі"""
    trained_models = {}
    
    for task_name, task_config in meta_tasks.items():
        print(f"\n=== Навчання {task_name} ===")
        
        # Конфігурація
        config = ConfigMetaCAT()
        config.general['category_name'] = task_name
        config.general['category_value2id'] = task_config['values']
        config.general['cntx_left'] = 15
        config.general['cntx_right'] = 10
        config.general['device'] = 'cuda'
        
        config.model['model_name'] = 'bert'
        config.model['model_variant'] = 'emilyalsentzer/Bio_ClinicalBERT'
        
        config.train['nepochs'] = 25
        config.train['lr'] = 2e-5
        config.train['batch_size'] = 32
        
        # Створення моделі
        meta_model = MetaCAT(tokenizer=tokenizer, config=config)
        
        # Навчання
        results = meta_model.train_from_json(
            json_path=training_json_path,
            save_dir_path=f'{output_dir}/{task_name.lower()}'
        )
        
        print(f"✓ {task_name} - F1: {results['f1']:.3f}")
        
        # Збереження
        meta_model.save(f'{output_dir}/{task_name.lower()}', full_save=True)
        trained_models[task_name] = meta_model
    
    return trained_models

# Використання
trained_metas = train_all_meta_models(
    meta_tasks_config,
    './training_data/meta_annotations.json',
    './models/meta_models'
)
```

### 5.4 Формат даних для Meta-CAT

**Створення meta-annotations для вашого тренувального JSON:**

```python
def add_meta_annotations_to_training_data(training_json_path, output_path):
    """Додає meta-annotations до існуючих анотацій"""
    with open(training_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Правила для автоматичного додавання meta-annotations
    meta_rules = {
        'severe': {'Severity': 'severe'},
        'mild': {'Severity': 'mild'},
        'moderate': {'Severity': 'moderate'},
        'no': {'Status': 'no'},
        'denies': {'Status': 'no'},
        'has': {'Status': 'yes'},
        'started': {'ActionStatus': 'started'},
        'completed': {'ActionStatus': 'done'},
        'cancelled': {'ActionStatus': 'canceled'}
    }
    
    for project in data['projects']:
        for doc in project['documents']:
            for ann in doc['annotations']:
                # Аналіз значення для визначення meta-annotations
                value_lower = ann['value'].lower()
                
                # Додавання meta-annotations на основі правил
                if 'meta_anns' not in ann:
                    ann['meta_anns'] = {}
                
                for trigger, meta_values in meta_rules.items():
                    if trigger in value_lower:
                        for meta_name, meta_value in meta_values.items():
                            ann['meta_anns'][meta_name] = {
                                'value': meta_value,
                                'confidence': 1.0,
                                'name': meta_name
                            }
    
    # Збереження
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Meta-annotations додано")
    print(f"✓ Збережено: {output_path}")
    
    return data

# Використання
enhanced_data = add_meta_annotations_to_training_data(
    './training_data/supervised_annotations.json',
    './training_data/with_meta_annotations.json'
)
```

---

## 6. Повний pipeline: Інтеграція всіх компонентів

### 6.1 Створення production-ready моделі

```python
from medcat.cat import CAT
from medcat.cdb import CDB
from medcat.vocab import Vocab
from medcat.meta_cat import MetaCAT

# Крок 1: Завантаження базових компонентів
vocab = Vocab.load('./models/vocab.dat')
cdb = CDB.load('./models/custom_cdb.dat')

# Крок 2: Завантаження всіх meta-models
meta_models = []
meta_task_names = ['status', 'severity', 'actionstatus', 'pattern', 'level']

for task_name in meta_task_names:
    try:
        meta_model = MetaCAT.load(f'./models/meta_models/{task_name}')
        meta_models.append(meta_model)
        print(f"✓ Завантажено Meta-CAT: {task_name}")
    except Exception as e:
        print(f"⚠ Не вдалося завантажити {task_name}: {e}")

# Крок 3: Створення повного CAT з усіма моделями
cat = CAT(
    cdb=cdb,
    config=cdb.config,
    vocab=vocab,
    meta_cats=meta_models
)

# Крок 4: Збереження як єдиний model pack
cat.create_model_pack('./models/production_model_pack.zip')

print("✓ Production модель створена!")
```

### 6.2 Використання моделі

```python
# Завантаження production моделі
cat = CAT.load_model_pack('./models/production_model_pack.zip')

# Аналіз тексту
text = """
Patient presents with severe chest pain radiating to left arm.
Started metformin 500mg BID for diabetes.
No history of hypertension.
Lab results show elevated glucose levels.
"""

# Екстракція сутностей з мета-анотаціями
entities = cat.get_entities(text)

# Виведення результатів
for ent_id, entity in entities['entities'].items():
    print(f"\n{'='*60}")
    print(f"Текст: {entity['source_value']}")
    print(f"CUI: {entity['cui']}")
    print(f"Type: {entity.get('types', 'N/A')}")
    print(f"Confidence: {entity['acc']:.3f}")
    print(f"Позиція: {entity['start']}-{entity['end']}")
    
    # Meta-annotations
    if 'meta_anns' in entity and entity['meta_anns']:
        print(f"\nКонтекстні атрибути:")
        for meta_name, meta_data in entity['meta_anns'].items():
            print(f"  {meta_name}: {meta_data['value']} (conf: {meta_data.get('confidence', 'N/A'):.2f})")
```

---

## 7. Best Practices та Оптимізація

### 7.1 Evaluation та метрики

```python
from sklearn.metrics import precision_recall_fscore_support, classification_report

def evaluate_model(cat, test_data_path):
    """Оцінка моделі на тестових даних"""
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    y_true_all = []
    y_pred_all = []
    
    for project in test_data['projects']:
        for doc in project['documents']:
            text = doc['text']
            gold_anns = doc['annotations']
            
            # Prediction
            pred_entities = cat.get_entities(text)
            
            # Порівняння
            gold_cuis = {(ann['start'], ann['end'], ann['cui']) for ann in gold_anns if ann['correct']}
            pred_cuis = {(e['start'], e['end'], e['cui']) for e in pred_entities['entities'].values()}
            
            # Metrics
            tp = len(gold_cuis & pred_cuis)
            fp = len(pred_cuis - gold_cuis)
            fn = len(gold_cuis - pred_cuis)
            
            y_true_all.extend([1] * len(gold_cuis))
            y_pred_all.extend([1 if cui in pred_cuis else 0 for cui in gold_cuis])
    
    # Обчислення метрик
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_all, y_pred_all, average='binary'
    )
    
    print(f"\n{'='*50}")
    print(f"MODEL EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"{'='*50}\n")
    
    return {'precision': precision, 'recall': recall, 'f1': f1}

# Використання
results = evaluate_model(cat, './test_data/test_annotations.json')
```

### 7.2 Performance оптимізація

```python
# Оптимізація для великих обсягів даних
config.general['workers'] = 8  # Multiprocessing

# Filtering для підвищення precision
config.linking['filters'] = {
    'threshold': 0.30,  # Вище = більше precision, менше recall
    'threshold_type': 'confidence'
}

# Оптимізація пам'яті
config.general['spacy_model'] = 'en_core_web_sm'  # Менша модель
config.general['spacy_disabled_components'] = ['parser', 'tagger']

# Batch processing
def batch_process_documents(cat, documents, batch_size=100):
    """Batch processing для великих обсягів"""
    results = []
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        batch_results = cat.multiprocessing(batch, nproc=8)
        results.extend(batch_results)
    
    return results
```

### 7.3 Моніторинг та логування

```python
import logging
from datetime import datetime

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'medcat_{datetime.now():%Y%m%d}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('MedCAT-Custom')

# Моніторинг performance
class MedCATMonitor:
    def __init__(self, cat):
        self.cat = cat
        self.stats = {
            'documents_processed': 0,
            'entities_extracted': 0,
            'avg_confidence': 0,
            'processing_times': []
        }
    
    def process_with_monitoring(self, text):
        import time
        start = time.time()
        
        entities = self.cat.get_entities(text)
        
        elapsed = time.time() - start
        self.stats['documents_processed'] += 1
        self.stats['entities_extracted'] += len(entities['entities'])
        self.stats['processing_times'].append(elapsed)
        
        if len(entities['entities']) > 0:
            avg_conf = sum(e['acc'] for e in entities['entities'].values()) / len(entities['entities'])
            self.stats['avg_confidence'] = (
                self.stats['avg_confidence'] * (self.stats['documents_processed'] - 1) + avg_conf
            ) / self.stats['documents_processed']
        
        return entities
    
    def print_stats(self):
        import numpy as np
        times = np.array(self.stats['processing_times'])
        
        logger.info(f"\n{'='*60}")
        logger.info(f"MEDCAT PROCESSING STATISTICS")
        logger.info(f"{'='*60}")
        logger.info(f"Documents processed: {self.stats['documents_processed']}")
        logger.info(f"Total entities: {self.stats['entities_extracted']}")
        logger.info(f"Avg entities/doc: {self.stats['entities_extracted']/max(1, self.stats['documents_processed']):.2f}")
        logger.info(f"Avg confidence: {self.stats['avg_confidence']:.3f}")
        logger.info(f"Processing time - Mean: {times.mean():.3f}s, Median: {np.median(times):.3f}s")
        logger.info(f"{'='*60}\n")

# Використання
monitor = MedCATMonitor(cat)
for text in documents:
    entities = monitor.process_with_monitoring(text)
monitor.print_stats()
```

---

## 8. Deployment та Integration

### 8.1 REST API з Flask

```python
from flask import Flask, request, jsonify
from medcat.cat import CAT
import logging

app = Flask(__name__)

# Завантаження моделі при старті
logger.info("Завантаження MedCAT моделі...")
cat = CAT.load_model_pack('./models/production_model_pack.zip')
logger.info("Модель завантажена успішно!")

@app.route('/api/extract', methods=['POST'])
def extract_entities():
    """API endpoint для екстракції сутностей"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Екстракція
        entities = cat.get_entities(text)
        
        # Форматування результату
        result = {
            'text': text,
            'entities': [
                {
                    'text': ent['source_value'],
                    'cui': ent['cui'],
                    'start': ent['start'],
                    'end': ent['end'],
                    'type': ent.get('types', []),
                    'confidence': ent['acc'],
                    'meta_annotations': ent.get('meta_anns', {})
                }
                for ent in entities['entities'].values()
            ],
            'count': len(entities['entities'])
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': cat is not None,
        'version': '1.0'
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

### 8.2 Docker deployment

**Dockerfile:**

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Встановлення залежностей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Завантаження spacy model
RUN python -m spacy download en_core_web_md

# Копіювання коду та моделей
COPY . .
COPY models/ /app/models/

# Експозиція порту
EXPOSE 5000

# Запуск
CMD ["python", "api.py"]
```

**docker-compose.yml:**

```yaml
version: '3.8'

services:
  medcat-api:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./models:/app/models:ro
    environment:
      - WORKERS=4
      - MODEL_PATH=/app/models/production_model_pack.zip
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G
```

---

## 9. Troubleshooting та FAQ

### Проблема: Низька Precision (багато false positives)

```python
# Рішення 1: Підвищити threshold
cat.config.linking['filters']['threshold'] = 0.5

# Рішення 2: Використати CUI фільтри
cat.config.linking['filters']['cuis'] = allowed_cui_set

# Рішення 3: Додати Meta-CAT для контекстної фільтрації
```

### Проблема: Низька Recall (пропускає entities)

```python
# Рішення 1: Додати синоніми до CDB
cat.cdb.add_names(cui='CUSTOM_001', names={'new_synonym': {}})

# Рішення 2: Знизити threshold
cat.config.linking['filters']['threshold'] = 0.2

# Рішення 3: Збільшити context window
cat.config.ner['max_skip'] = 5
```

### Проблема: Високе споживання пам'яті

```python
# Рішення 1: Фільтрувати vocabulary
def filter_vocab(vocab, min_freq=10):
    filtered = Vocab()
    for word, count in vocab.word_count.items():
        if count >= min_freq:
            filtered.add_word(word, vocab.get_vector(word))
    return filtered

# Рішення 2: Використати меншу spacy модель
config.general['spacy_model'] = 'en_core_web_sm'

# Рішення 3: Фільтрувати CDB за Type IDs
filtered_cdb = cdb.filter_by_cui(target_cui_list)
```

---

## 10. Корисні ресурси

**Офіційна документація:**
- GitHub: https://github.com/CogStack/MedCAT
- Tutorials: https://github.com/CogStack/MedCATtutorials
- ReadTheDocs: https://medcat.readthedocs.io/
- MedCATtrainer: https://github.com/CogStack/MedCATtrainer

**Research Papers:**
- Kraljevic et al. (2021): "Multi-domain Clinical Natural Language Processing with MedCAT"
- F1 scores: 0.448-0.738 на UMLS datasets
- Cross-hospital transferability: F1 > 0.94

**Моделі та дані:**
- UMLS models (потрібна UMLS license)
- SNOMED-CT models
- Bio_ClinicalBERT для Meta-CAT

**Community:**
- Discussion Forum: https://discourse.cogstack.org
- GitHub Issues для багів та feature requests

---

## Висновки

Створення кастомної MedCAT моделі на основі вашої онтології включає:

1. **Підготовка CDB** з вашої ієрархічної структури (кластери → keywords → патерни)
2. **Unsupervised training** на великих обсягах клінічних текстів
3. **Supervised training** з анотованими даними або згенерованими з патернів
4. **Meta-CAT** для екстракції контекстних атрибутів (status, severity, etc.)
5. **Evaluation та оптимізація** для production deployment
6. **Integration** через REST API або вбудовану бібліотеку

**Ключові метрики для успіху:**
- F1 score > 0.85 на тестових даних
- Precision/Recall балансування через threshold tuning
- Cross-validation на різних типах текстів
- Production-ready deployment з моніторингом

Ваша онтологія ідеально підходить для MedCAT, оскільки вона вже має чітку структуру type IDs (кластери) та синоніми (патерни). Використовуючи цей гайд, ви зможете створити high-performance систему для екстракції медичних сутностей з контекстним розумінням.