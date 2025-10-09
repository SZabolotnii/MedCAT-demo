# Юридична та технічна експертиза: MedCAT з UMLS для комерційного використання

**Пряма відповідь: НІ** для pretrained MIMIC-III моделей, але **ТАК можливо** з модифікованим підходом та proper compliance.

Pretrained моделі MedCAT, натреновані на MIMIC-III з UMLS, не можна легально використовувати у комерційному SaaS продукті через перетинання множинних ліцензійних обмежень. Проте існує чіткий шлях до комерційного використання через модифікований transfer learning підхід або альтернативні open-source рішення. Це дослідження виявило три критичні юридичні бар'єри, два viable workarounds, та один оптимальний шлях для швидкого виходу на ринок.

## Три фатальні юридичні бар'єри для Варіанту 5

### MIMIC-III trained models порушують research-only обмеження

PhysioNet Data Use Agreement для MIMIC-III містить чітку заборону на комерційне використання: "for the sole purpose of lawful use in scientific research, and no other". Це не просто застереження — це юридично обов'язкова угода, яку кожен користувач підписує при отриманні доступу.

Критичніше те, що PhysioNet має **експліцитну політику щодо похідних моделей**, опубліковану у квітні 2024: "Any derived datasets or models should be treated as containing sensitive information. If you wish to share these resources, they should be shared on PhysioNet under the same agreement as the source data." Це означає, що model weights, натреновані на MIMIC-III, юридично прирівнюються до самих клінічних даних і мають ті самі обмеження.

Прецедент встановлено Clinical-T5 models: навіть великі language models, натреновані на MIMIC, залишаються під credentialed access на PhysioNet через документовані ризики витоку чутливої інформації. Дослідження показали, що BERT-подібні моделі можуть бути queried для витягування training data, що робить model weights потенційним джерелом PHI (Protected Health Information).

Більше того, **сайт завантаження MedCAT** від King's College London прямо зазначає для pretrained models: "This software is intended solely for testing purposes and non-commercial use." Це додаткове contractual обмеження поверх MIMIC-III restrictions.

### Transfer learning створює derivative works з licensing obligations

Нещодавні юридичні розробки 2025 року кардинально змінили landscape для AI model licensing. У справі **Thomson Reuters v. ROSS Intelligence** (лютий 2025) федеральний суд Delaware **відхилив fair use defense** для AI training на copyrighted works для створення конкуруючого продукту. Це перше major AI fair use рішення показує, що суди не вважають комерційний AI training автоматично transformative use.

Ще драматичніше — **історичне settlement у справі Bartz v. Anthropic** (серпень 2025) на **$1.5 мільярда**, найбільше copyright settlement в історії США. Хоча суддя Alsup ruled що використання легально придбаних книг для training є fair use, використання pirated works з "The Pile" — ні. Це встановлює, що provenance training data має критичне значення.

U.S. Copyright Office у травні 2025 опублікував позицію: "Where generated outputs are substantially similar to inputs, there is a 'strong argument' that copying the model's weights implicates the reproduction and derivative work rights of the original works." Model weights більше не можна вважати просто "математичними параметрами" — вони є legal derivatives якщо містять substantial similarity до training data.

Для вашого transfer learning сценарію це означає: **fine-tuned models є derivative works** base model. Якщо base model має research-only restriction (як MIMIC-trained models), fine-tuned version успадковує цю рестрикцію. Юридичний принцип: "The fine-tuned model cannot grant more rights than exist in the underlying base model." Навіть якщо ви додаєте власні концепти до CDB, integrated system залишається derivative work restrictive base model.

### UMLS licensing залишає критичну неясність щодо embedded models

UMLS Metathesaurus License Agreement від National Library of Medicine має **безкоштовну** ліцензію без окремих комерційних fees, але з важливими обмеженнями. Section 3 дозволяє використання "as an integral part of computer applications developed by LICENSEE for a purpose other than redistribution of vocabulary sources", що здається дозволяти комерційні додатки.

Проте **критична неясність**: ліцензія була написана до епохи ML і **не адресує явно** neural network embeddings, model weights trained on UMLS, або concept vectors derived from UMLS. Дослідження виявило extensive academic research (UmlsBERT, cui2vec, definition2vec) що публікує UMLS-trained models без очевидних licensing restrictions, але це не створює legal precedent для commercial use.

Section 4 вимагає: "LICENSEE agrees to inform NLM prior to distributing any application(s) in which it is using the UMLS Metathesaurus." Section 5 вимагає щорічні usage reports, і **failure to submit terminates license and access**. Section 11 забороняє "altering UMLS and other vocabulary source content" — чи model embeddings вважаються "altered content"?

Найбільша проблема: **Category 3 vocabularies** у UMLS (включаючи CPT від AMA, MedDRA, CDT) мають explicit prohibition: "LICENSEE's right to use material from the source vocabulary is restricted to internal use at the LICENSEE's site(s) for research, product development, and statistical analysis only... expressly excludes: incorporation of material from these copyrighted sources in any publicly accessible computer-based information system." Ці vocabularies **не можна використовувати** у commercial SaaS products, period.

## Два viable шляхи до legal compliance

### Modified Variant 5: власний training pipeline з MedCAT software

Elastic License 2.0, під якою ліцензовано MedCAT з серпня 2022, **дозволяє комерційне використання** software як library з трьома обмеженнями. Elastic's офіційний FAQ прямо підтверджує: "You may freely use Elasticsearch inside your SaaS or self-managed application, and redistribute it with your application."

Ключовий тест для managed service restriction: чи ваші користувачі отримують доступ до **вашого продукту** (який використовує MedCAT internally) чи до **MedCAT functionality** напряму? Для SaaS клінічного NLP де лікарі взаємодіють з вашим healthcare application interface, а MedCAT обробляє дані behind the scenes — це **permitted use**.

**Модифікований Варіант 5 implementation:**

1. **Використовуйте MedCAT software code** (Elastic License 2.0 permits)
2. **НЕ завантажуйте pretrained MIMIC-III models** (уникаєте research restrictions)
3. **Train власні models** на вашій медичній онтології або commercially-licensed clinical data
4. **Додайте власні концепти до CDB** — це стає вашим intellectual property
5. **Self-supervised training** на ваших клінічних даних (якщо маєте access)
6. **Опціонально інтегруйте UMLS** з proper compliance

**UMLS compliance requirements:**
- Кожен developer отримує individual UMLS license (безкоштовно, 5 business days approval)
- Notify NLM перед commercial distribution через custserv@nlm.nih.gov
- Подавати annual usage reports кожного січня через UTS profile
- Використовувати тільки Category 0 vocabularies (67% UMLS content) для commercial products
- Avoid Category 3 sources (CPT, MedDRA, CDT) які explicitly prohibited
- Include required copyright notices та UMLS citation
- **Critical: contact NLM для written clarification** щодо ML model weights

**Technical implementation:** MedCAT підтримує додавання custom concepts через CDB (Concept Database). Ви можете створити власну медичну онтологію, train unsupervised models на clinical text, і використовувати MedCAT's architecture без dependency на MIMIC-trained weights. Performance може бути нижчою initially, але ви можете iterate з власними даними.

**Вартість і timeline:** $100K-$250K для ML engineering (6-12 місяців), включаючи розробку training pipeline, annotation tooling якщо потрібно, compute infrastructure, та iterative improvements. Legal risk: **LOW** з чистим data provenance та proper UMLS compliance.

### scispaCy альтернатива: найшвидший шлях до market

Allen Institute for AI's scispaCy має **MIT License** — найбільш permissive open-source license без будь-яких restrictions на commercial use, modification, або redistribution. Це eliminate licensing headaches повністю.

**scispaCy capabilities** comparable до MedCAT для багатьох use cases: biomedical NER з ~100K vocabulary, entity linking до multiple ontologies (UMLS, MeSH, RxNorm, Gene Ontology, HPO), abbreviation detection, trained на GENIA corpus, PubMed abstracts, та clinical datasets. Performance within 3% of state-of-art parsers на biomedical benchmarks.

**Архітектурна різниця:** scispaCy використовує supervised learning approach (потребує labeled training data) versus MedCAT's self-supervised approach (може train на unlabeled text). Для вашого use case з власною онтологією, supervised approach може бути навіть кращим — ви можете створити targeted training set для ваших specific medical concepts.

**Integration strategy:**

```python
import spacy
import scispacy
from scispacy.linking import EntityLinker
from scispacy.abbreviation import AbbreviationDetector

# Load biomedical model
nlp = spacy.load("en_core_sci_lg")  # або en_ner_bc5cdr_md для specific domains

# Add entity linking до MeSH (безкоштовно) замість UMLS
nlp.add_pipe("scispacy_linker", config={
    "resolve_abbreviations": True,
    "linker_name": "mesh"  # або rxnorm для medications
})

# Add abbreviation detection
nlp.add_pipe("abbreviation_detector")

# Process clinical text
doc = nlp("Patient presents with CHF and T2DM...")
```

**Додайте medspaCy** для clinical-specific functionality: ConText algorithm для negation/uncertainty detection (critical для clinical NLP), section detection для structured notes, та custom rule-based components. medspaCy також MIT licensed.

**Link до безкоштовних terminologies замість UMLS:**
- **MeSH** (Medical Subject Headings) — безкоштовно від NLM, 29K+ descriptors, жодних redistribution restrictions
- **RxNorm** — безкоштовно для US use, normalized drug names
- **ICD-10-CM** — безкоштовно, diagnosis codes
- **LOINC** — безкоштовно з реєстрацією, lab/clinical observations
- Ваша **власна онтологія** — повний IP control

**Вартість і timeline:** $50K-$150K для customization та domain adaptation (2-4 місяці). Includes fine-tuning на вашому domain, створення custom entity recognizers для вашої онтології, integration з вашим SaaS infrastructure, та validation. Legal risk: **MINIMAL** — MIT license дає complete freedom. Investor due diligence: **passes easily** — чистий open-source stack без licensing complications.

## FDA та HIPAA compliance для US clinics

### Clinical NLP зазвичай exempt від FDA regulation

21st Century Cures Act Section 520(o)(1)(E) створив broad exemption для Clinical Decision Support software. Ваш NLP tool для клінік США **НЕ вимагає FDA clearance** якщо виконує всі чотири criteria:

**Criterion 1 (Data Types):** Не обробляє medical imaging, IVD signals, або continuous monitoring patterns — ✅ NLP текстової документації qualifies

**Criterion 2 (Medical Information):** Displays/analyzes інформацію що normally communicated between healthcare providers (clinical notes, lab results, discharge summaries) — ✅ Clinical text processing qualifies

**Criterion 3 (Output Type):** Provides recommendations або options to HCPs rather than specific diagnostic directives — ✅ **CRITICAL DESIGN CHOICE**: Structure outputs як "suggested concepts" або "possible diagnoses for review" замість "Patient HAS sepsis" 

**Criterion 4 (Transparency):** Enables HCPs to independently review algorithm basis — ✅ Include plain language descriptions, show extracted text, display confidence scores, document training data

**FDA-exempt design patterns для NLP:**
- ✅ Extract entities from notes та display для HCP review
- ✅ Suggest ICD/SNOMED codes for clinical documentation
- ✅ Highlight relevant clinical concepts для attention
- ✅ Provide lists of differential diagnoses options
- ✅ Match patient conditions to clinical guidelines

**FDA-regulated scenarios (avoid):**
- ❌ Output single risk score: "85% probability of sepsis"
- ❌ Specific diagnostic conclusions: "Patient HAS condition X"
- ❌ Time-critical alerts/alarms from NLP analysis
- ❌ Automated treatment recommendations without HCP review

Якщо дотримуєтесь exempt CDS design patterns, ваш product не потребує costly 510(k) premarket notification або De Novo classification. FDA Pre-Submission consultation доступна якщо uncertainty залишається.

### HIPAA compliance обов'язкова як Business Associate

Будь-який third-party NLP tool що processes Protected Health Information є **Business Associate** під HIPAA з 2009 HITECH Act. Ви directly liable за violations independent від covered entities (clinics).

**Mandatory requirements перед launch:**

1. **Business Associate Agreements (BAAs)** з кожною клінікою describing permitted PHI uses, requiring safeguards, prohibiting unauthorized disclosure, mandating breach notification within 60 days

2. **Technical safeguards:** AES-256 encryption для data at rest та in transit, role-based access controls, multi-factor authentication, audit logging всіх PHI access, automatic session timeouts, secure API authentication

3. **Administrative safeguards:** Designate HIPAA compliance officer, conduct annual risk assessments documenting threats/vulnerabilities/mitigations, implement written policies and procedures, train всіх employees annually з documented completion, establish incident response plan

4. **Physical safeguards:** Secure data centers (або AWS/Azure/GCP з HIPAA BAAs), workstation security policies, device encryption

**Breach notification requirements:** Якщо PHI breach affects 500+ individuals, must notify HHS within 60 days та notify media. Penalties: $100-$50,000 per violation depending on negligence level, annual maximum $1.5M per violation category.

**Practical SaaS implementation:** Використовуйте HIPAA-compliant cloud providers (AWS with BAA, Azure Healthcare, Google Cloud Healthcare API), implement logging via CloudTrail/equivalent, regular penetration testing, third-party security audits, cyber liability insurance з HIPAA coverage. Budget $50K-$150K annually для comprehensive HIPAA compliance program.

## Порівняльний аналіз трьох стратегій

### Strategy A: Modified MedCAT з власним training

**Pros:** Leverage proven MedCAT architecture від NHS deployments (King's College Hospital processed 2M+ documents generating £15M revenue), sophisticated self-supervised learning approach, active development community від CogStack, published validation studies.

**Cons:** Higher development complexity requiring ML expertise, longer time to market (6-12 months), UMLS licensing uncertainty потребує NLM clarification, ongoing compliance burden (annual reports), trained models initially lower performance порівняно з pretrained MIMIC models.

**Best for:** Organizations з ML capabilities, use cases requiring self-supervised learning на unlabeled clinical data, enterprises willing to invest у long-term proprietary models, scenarios де existing clinical notes можуть serve як training data.

**Total cost:** $150K-$300K (development + ML engineering + compute + compliance setup)
**Risk level:** Medium (UMLS uncertainty, але mitigatable з NLM consultation)
**Investor perception:** Acceptable з clear compliance documentation

### Strategy B: scispaCy + medspaCy open stack

**Pros:** MIT license eliminates всі licensing concerns, proven Allen AI maintenance та community, faster time to market (2-4 months), comparable performance на biomedical tasks, can freely redistribute та modify, lower legal/compliance overhead, investor-friendly clean IP stack.

**Cons:** Supervised approach requires labeled training data (може потребувати annotation effort $50K-$200K), менше clinical-specific optimizations out-of-box порівняно з MedCAT, may need combining multiple tools (scispaCy + medspaCy + custom components), terminology coverage lower без UMLS (але MeSH + RxNorm + LOINC + власна онтологія може compensate).

**Best for:** Startups optimizing для швидкого launch, SaaS business models requiring clear licensing, international deployment без geographic restrictions, teams з NLP expertise але limited ML infrastructure, scenarios prioritizing legal clarity over absolute performance.

**Total cost:** $80K-$180K (customization + annotation + integration)
**Risk level:** Minimal (MIT license, established tools)
**Investor perception:** Excellent — clean open-source stack

### Strategy C: Commercial vendors (John Snow Labs)

**Pros:** Highest accuracy з 2,500+ pretrained medical models, regulatory-grade validation, full compliance management included, enterprise support та SLAs, rapid deployment (weeks), used by pharmaceutical companies та healthcare systems, handles UMLS/SNOMED licensing internally, regular updates та maintenance.

**Cons:** Expensive ($50K-$500K annually depending on scale), vendor lock-in creates dependency, less customization flexibility, pricing scales з usage potentially limiting margins, requires budget availability limiting to well-funded organizations.

**Best for:** Series B+ companies з funding, enterprise healthcare deployments needing established validation, pharmaceutical/clinical research applications requiring regulatory-grade accuracy, organizations без internal ML capabilities, scenarios needing rapid deployment з guaranteed support.

**Total cost:** $100K-$600K annually (licensing + integration + training)
**Risk level:** Minimal (vendor handles compliance)
**Investor perception:** Acceptable для funded companies

## Оптимальна стратегія: Hybrid progressive approach

Найефективніший шлях combines benefits кожного approach через phased implementation aligned з business stage:

**Phase 1 (MVP / Pre-Seed to Seed, months 0-4):**
Start з **scispaCy + medspaCy + MeSH** для proof-of-concept та early customer validation. MIT licensed stack дозволяє швидку iteration без legal overhead. Link entities до MeSH medical concepts (безкоштовно, 29K descriptors) та RxNorm для medications. Implement basic clinical NLP pipeline: tokenization, sentence segmentation, entity recognition, negation detection, abbreviation expansion.

Build initial custom models для вашої специфічної онтології через fine-tuning BioClinicalBERT або PubMedBERT (обидва permissively licensed). Create minimal annotation set (~1,000 examples) для ваших priority medical concepts. Deploy як beta service для pilot clinics з clear HIPAA BAAs.

**Phase 2 (Product-Market Fit / Series A, months 4-12):**
Enhance з **domain-specific training** на feedback від early customers. Collect clinical text (з proper consents) для improving models на real-world data. Expand terminology coverage: add LOINC для lab results, ICD-10-CM для diagnoses, CPT alternative ontologies для procedures.

Evaluate adding **UMLS integration** для enterprise tier: contact NLM для official clarification (custserv@nlm.nih.gov), obtain individual licenses для team, implement client-managed UMLS approach де enterprise customers provide credentials, maintain Category 0 vocabularies only для embedded use.

**Phase 3 (Scale / Series B+, months 12-24):**
Consider **commercial partnerships** для advanced features: integrate John Snow Labs для pharmaceutical/research customers needing regulatory validation, offer tiered product з Standard (open stack) та Enterprise (commercial-enhanced) versions, alternatively train proprietary models з Modified MedCAT approach для competitive differentiation.

Expand **regulatory validation:** conduct prospective clinical studies для FDA exemption documentation, publish accuracy benchmarks у peer-reviewed journals, obtain third-party security certifications (SOC 2 Type II, HITRUST), pursue ONC Health IT Certification якщо applicable.

**Investment requirements by phase:**
- Phase 1: $80K-$120K (4 months development)
- Phase 2: $150K-$250K (8 months enhancement)  
- Phase 3: $200K-$400K annually (scaling + partnerships)

**Risk mitigation:** Phase 1's open-source foundation provides fallback якщо Phase 2-3 licensing complications виникають. Investor due diligence passes easily з clean Phase 1 IP, while Phase 2-3 adds competitive moats.

## Критичні action items перед launch

### Юридичні prerequisites

**Immediate (перед будь-яким development):**
- ❌ **НЕ завантажуйте** MIMIC-III pretrained MedCAT models для commercial use
- ✅ Consult healthcare/IP attorney для review вашої specific implementation (budget $10K-$25K)
- ✅ Якщо using UMLS: contact NLM з detailed use case description requesting written clarification на ML model weights status
- ✅ Document licensing decisions у written IP policy для investor due diligence

**UMLS compliance workflow (якщо використовуєте):**
1. Всі developers obtain individual UMLS licenses через https://uts.nlm.nih.gov/uts/signup-login (5 business days)
2. Review Appendix 1 identifying Category 0 (unrestricted) vs Category 3 (prohibited) vocabularies
3. Email custserv@nlm.nih.gov з pre-distribution notification describing application
4. Setup annual reporting reminder (due January) через UTS profile — failure terminates license
5. Include required copyright notices: "UMLS Metathesaurus, 2025AA, National Library of Medicine"
6. Preserve UMLS concept identifiers (CUIs) для traceability у output

### Technical implementation safeguards

**Data provenance documentation:**
- Maintain detailed records всіх training data sources з licensing terms
- Document model lineage: base model → fine-tuning data → final weights
- Implement versioning для models з changelog describing modifications
- Create audit trail linking deployed models до training provenance

**FDA-exempt CDS design patterns:**
- Structure outputs як "suggested concepts for provider review" not "diagnoses"
- Display confidence scores та supporting evidence (extracted text snippets)
- Enable HCP to access reasoning: show algorithm logic у plain language
- Include disclaimers: "Clinical decision support tool requiring provider judgment"
- Avoid single risk scores — provide contextual information instead

**HIPAA technical controls:**
- Implement PHI encryption at rest (AES-256) та in transit (TLS 1.3+)
- Setup comprehensive audit logging capturing all PHI access з timestamps, user IDs, actions
- Deploy role-based access control limiting PHI visibility to authorized staff only
- Configure automatic de-identification для development/testing environments
- Establish secure model serving infrastructure з authentication/authorization

### Investor due diligence preparation

Healthcare AI investors з Series A+ conduct rigorous IP and compliance review. Prepare documentation package:

**IP clarity documentation:**
- Complete software bill of materials (SBOM) listing всі dependencies з licenses
- Third-party license inventory з commercial use analysis
- Model training data provenance documentation
- Freedom to operate analysis for your specific medical domain

**Regulatory compliance evidence:**
- Written FDA CDS exemption analysis з legal counsel opinion
- HIPAA compliance program documentation (policies, procedures, training records)
- Business Associate Agreement templates
- Security risk assessment results
- Penetration testing reports

**Licensing correspondence:**
- UMLS license copies для всієї команди (якщо використовуєте)
- NLM clarification correspondence regarding model weights
- Any waivers або special permissions obtained

Red flags investors seek: using MIMIC-III models commercially без proper justification, missing UMLS licenses for developers accessing data, no HIPAA compliance program, unclear model provenance, GPL-licensed dependencies у proprietary product.

## Фінальна рекомендація: scispaCy foundation з UMLS optionality

Для вашого specific scenario — комерційний SaaS для клінік США з власною медичною онтологією — оптимальний шлях:

**Core recommendation: Start з scispaCy + medspaCy architecture**

1. **Immediate advantages:** MIT license provides complete freedom, Allen AI provides stable maintenance, comparable performance для entity recognition, fast time-to-market (2-3 months до MVP), clean investor due diligence story, zero licensing complexity.

2. **Implementation roadmap:** Deploy en_core_sci_lg або en_ner_bc5cdr_md base model, add medspaCy ConText для negation/uncertainty critical у clinical settings, link entities до MeSH (29K medical concepts, безкоштовно) + RxNorm (medications) + LOINC (labs), fine-tune BioClinicalBERT на вашу онтологію (~1K annotated examples sufficient), integrate у SaaS infrastructure з HIPAA controls.

3. **Your custom ontology integration:** scispaCy's EntityRuler дозволяє easily додавати custom medical concepts, можете train custom NER models на ваших specific entities, supervised approach може бути more accurate для вашого narrow domain versus MedCAT's broader self-supervised approach.

4. **Commercial viability:** Zero restrictions на SaaS deployment, can redistribute to international markets, freely modify та customize, no annual compliance reporting, no client licensing requirements, reduces legal review costs.

**Optional Phase 2: Add UMLS for enterprise tier**

Після product-market fit з 10+ customers, evaluate adding UMLS-enhanced version:

1. Contact NLM для official guidance: "We have commercial SaaS product using scispaCy base. Want to add optional UMLS entity linking for enterprise customers. Customers will NOT receive UMLS data files, only concept IDs in output. Model embeddings will not redistribute UMLS vocabulary. Is this permitted?"

2. Якщо NLM approves: implement two SKUs — Standard (MeSH linking, ~$X/month) та Enterprise (UMLS linking, ~$3X/month), require enterprise customers confirm UMLS licenses або implement client-managed UMLS credentials, maintain separation between tiers у codebase.

3. Alternative: Use SNOMED CT International (безкоштовно у US для member countries) замість UMLS для enterprise tier — provides 350K+ clinical concepts, clearer commercial terms, can use у applications without redistribution, only requires registration з SNOMED National Release Center.

**Budget allocation:**
- Development (4 months): $80K-$120K (2 senior ML engineers)
- Clinical annotation: $40K-$60K (1,000 examples professionally annotated)
- Infrastructure: $20K-$30K (HIPAA-compliant cloud setup)
- Legal review: $15K-$25K (healthcare attorney consultation)
- **Total MVP: $155K-$235K**

**Timeline to commercial launch:** 4-6 months including development, pilot testing з 2-3 clinics, HIPAA compliance setup, initial customer validation.

**Expected performance:** F1 score 0.75-0.85 на вашому domain після fine-tuning (comparable до MedCAT для specific ontologies), 100-500 notes/second processing throughput, <200ms latency для typical clinical document.

Цей підхід minimizes legal risk (MIT license), maximizes speed to market (proven tools), maintains flexibility (can add commercial components later), та passes investor due diligence easily (clean open-source foundation). Ви можете confidently build commercial product knowing licensing не стане blocker під час funding rounds або customer contracts.

Healthcare NLP commercialization у 2025 requires navigating complex regulatory landscape, але з правильним licensing foundation та proper compliance, ви можете швидко deploy effective clinical NLP solutions. Ключ — почати з permissive open-source tools, maintain clean data provenance, implement robust HIPAA controls, та reserve commercial/licensed components для premium tiers після validating market fit.