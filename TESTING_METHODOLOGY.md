# MedCAT Testing Methodology & Framework

## Executive Testing Strategy

### Phase 1: Foundation Validation (Dictionary-Based)
**Objective:** Establish baseline performance metrics for GO/NO-GO decision on Phase 1B
**Timeline:** 1 week
**Success Criteria:** F1 ≥ 0.78, Precision ≥ 0.80

### Phase 2: Enhancement Validation (Semantic Layer)
**Objective:** Quantify semantic similarity improvements
**Timeline:** Conditional on Phase 1 results
**Success Criteria:** ΔF1 ≥ +0.05, ΔRecall ≥ +0.08

## Testing Architecture Components

### 1. Data Layer Validation
```
Component: CSV → CDB Transformation
Test Focus: Data integrity, concept coverage, hint parsing
Metrics: Concept count accuracy, synonym mapping quality
```

### 2. NER Engine Testing
```
Component: CustomCAT entity detection
Test Focus: Precision, recall, gap-tolerant matching
Metrics: Entity-level F1, span-level accuracy
```

### 3. Combined Hints Validation
```
Component: Gap-tolerant pattern matching
Test Focus: Multi-word phrase detection with intervening text
Metrics: Pattern coverage, false positive rate
```

### 4. End-to-End Pipeline Testing
```
Component: Full extraction pipeline
Test Focus: Real-world clinical text processing
Metrics: Processing speed, memory usage, stability
```

## Detailed Testing Procedures

### Procedure 1: Dictionary Coverage Assessment

**Objective:** Validate that custom ontology concepts are correctly loaded and accessible

**Implementation:**
```python
def test_dictionary_coverage():
    """
    Validates CDB contains all expected concepts with correct mappings.
    
    Tests:
    1. Total CUI count matches expectation (7,219)
    2. Each CUI has at least one name
    3. Primary names are correctly identified
    4. Type IDs map to cluster titles
    """
    expected_cui_count = 7219
    
    assert len(cdb.cui2names) == expected_cui_count
    
    for cui in cdb.cui2names:
        assert len(cdb.cui2names[cui]) >= 1
        assert cui in cdb.cui2preferred_name
        
    # Cluster mapping validation
    for type_id in cdb.addl_info.get('type_id2name', {}):
        assert type_id in cluster_mapping
```

**Success Criteria:**
- ✅ 100% CUI coverage
- ✅ All primary names present
- ✅ Type ID mappings complete

### Procedure 2: Entity Detection Accuracy

**Objective:** Measure precision and recall on annotated clinical texts

**Implementation:**
```python
class EntityDetectionValidator:
    """
    Comprehensive entity detection validation framework.
    
    Measures:
    - Exact match accuracy (span boundaries must match exactly)
    - Partial match accuracy (overlap-based scoring)
    - Type-level accuracy (correct cluster assignment)
    """
    
    def calculate_metrics(self, predicted_entities, gold_entities):
        """
        Calculate precision, recall, F1 for entity detection.
        
        Args:
            predicted_entities: List[{text, start, end, cui, type}]
            gold_entities: List[{text, start, end, cui, type}]
        
        Returns:
            Dict with precision, recall, f1, detailed_results
        """
        exact_matches = self._calculate_exact_matches(predicted, gold)
        partial_matches = self._calculate_partial_matches(predicted, gold)
        type_matches = self._calculate_type_accuracy(predicted, gold)
        
        return {
            'exact_match': exact_matches,
            'partial_match': partial_matches,
            'type_accuracy': type_matches,
            'entity_count': len(predicted_entities),
            'gold_count': len(gold_entities)
        }
```

**Success Criteria:**
- ✅ Exact match F1 ≥ 0.75
- ✅ Partial match F1 ≥ 0.80
- ✅ Type accuracy ≥ 0.85

### Procedure 3: Combined Hints Gap Tolerance

**Objective:** Validate gap-tolerant matching for multi-word medical concepts

**Implementation:**
```python
def test_combined_hints_gap_tolerance():
    """
    Tests gap-tolerant matching for combined hint patterns.
    
    Test Cases:
    1. Zero gap: "aerosol intranasally" → exact match
    2. Single gap: "aerosol therapy intranasally" → 1 word gap
    3. Multi gap: "aerosol spray therapy given intranasally" → 3 word gap
    4. Beyond limit: "aerosol many words here intranasally" → should not match
    """
    test_patterns = [
        {
            'text': 'administered aerosol intranasally twice daily',
            'expected_cui': 'AEROSOL_INTRANASAL_CUI',
            'expected_match': True,
            'gap_count': 0
        },
        {
            'text': 'administered aerosol therapy intranasally',
            'expected_cui': 'AEROSOL_INTRANASAL_CUI', 
            'expected_match': True,
            'gap_count': 1
        },
        {
            'text': 'aerosol treatment was given intranasally',
            'expected_cui': 'AEROSOL_INTRANASAL_CUI',
            'expected_match': True,
            'gap_count': 3
        },
        {
            'text': 'aerosol treatment was carefully administered over time intranasally',
            'expected_cui': 'AEROSOL_INTRANASAL_CUI',
            'expected_match': False,  # Exceeds max_gap=3
            'gap_count': 6
        }
    ]
```

**Success Criteria:**
- ✅ 100% accuracy for gap ≤ max_gap
- ✅ 0% false positives for gap > max_gap
- ✅ Configurable gap tolerance

### Procedure 4: Performance Benchmarking

**Objective:** Establish processing speed and resource utilization baselines

**Implementation:**
```python
class PerformanceBenchmark:
    """
    Measures system performance under various load conditions.
    """
    
    def benchmark_processing_speed(self, documents, batch_sizes=[1, 10, 50]):
        """
        Measure processing speed across different batch sizes.
        
        Metrics:
        - Documents per second
        - Memory usage per document
        - Memory peak during processing
        """
        results = {}
        
        for batch_size in batch_sizes:
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss
            
            # Process documents in batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                _ = cat.batch_process(batch)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            
            results[batch_size] = {
                'docs_per_second': len(documents) / (end_time - start_time),
                'memory_delta_mb': (end_memory - start_memory) / (1024*1024),
                'total_time': end_time - start_time
            }
        
        return results
```

**Success Criteria:**
- ✅ Processing speed ≥ 10 docs/second (Phase 1A)
- ✅ Memory usage ≤ 2GB for 1000 documents
- ✅ No memory leaks during extended processing

## Quality Assurance Framework

### Automated Testing Pipeline

```bash
# Daily automated testing
python -m scripts.run_validation_suite \
    --test-set data/annotated_test_cases.json \
    --model models/IEE_MedCAT_v1 \
    --output reports/daily_validation.json

# Weekly performance regression testing
python -m scripts.performance_benchmark \
    --documents data/performance_test_corpus/ \
    --baseline reports/baseline_performance.json
```

### Manual Review Process

1. **Medical Expert Review** (Weekly)
   - Review false positives for clinical relevance
   - Validate new entity types discovered
   - Assess gap tolerance appropriateness

2. **Technical Review** (Bi-weekly)
   - Code quality assessment
   - Architecture compliance validation
   - Performance optimization opportunities

### Continuous Integration Hooks

```yaml
# .github/workflows/medcat_validation.yml
name: MedCAT Validation
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run validation suite
        run: |
          python -m scripts.validate_phase1a
          python -m scripts.test_combined_hints
          python -m scripts.performance_check
```

## Risk Mitigation Testing

### Edge Case Coverage

1. **Malformed Input Handling**
   - Empty documents
   - Non-English text detection
   - Extremely long documents (>50KB)
   - Special characters and encoding issues

2. **Ontology Edge Cases**
   - CUI conflicts and resolution
   - Circular reference detection
   - Missing cluster mappings

3. **Performance Degradation Scenarios**
   - Memory exhaustion protection
   - Processing timeout handling
   - Concurrent access validation

### Fallback Mechanism Testing

```python
def test_graceful_degradation():
    """
    Ensures system maintains functionality under adverse conditions.
    
    Scenarios:
    1. Partial CDB corruption
    2. Missing semantic embeddings (Phase 1B)
    3. spaCy model unavailability
    """
    # Test partial functionality maintenance
    # Validate error reporting and logging
    # Ensure no silent failures
```

## Metrics Collection & Reporting

### Primary Metrics Dashboard

```python
class MetricsDashboard:
    """
    Real-time metrics collection and visualization.
    """
    
    def generate_validation_report(self, results):
        """
        Generate comprehensive validation report.
        
        Includes:
        - Entity detection performance
        - Processing speed benchmarks
        - Error analysis and categorization
        - Trend analysis over time
        """
        return {
            'summary': self._generate_summary_stats(results),
            'detailed_metrics': self._calculate_detailed_metrics(results),
            'error_analysis': self._analyze_errors(results),
            'recommendations': self._generate_recommendations(results)
        }
```

### Key Performance Indicators (KPIs)

**Phase 1A Decision Metrics:**
- Entity Detection F1 Score
- Processing Speed (docs/second)
- Memory Efficiency (MB/doc)
- Error Rate (failures/1000 docs)

**Phase 1B Evaluation Metrics:**
- Semantic Matching Accuracy
- Combined System F1 Improvement
- Latency Impact Assessment
- Resource Utilization Increase

## Testing Timeline & Milestones

### Week 1: Foundation Testing
- [ ] Dictionary coverage validation
- [ ] Basic entity detection testing
- [ ] Combined hints gap tolerance verification
- [ ] Performance baseline establishment

### Week 2: Comprehensive Validation
- [ ] Full test suite execution on annotated dataset
- [ ] Edge case testing and error handling
- [ ] Performance regression testing
- [ ] Medical expert review session

### Week 3: Decision Point Analysis
- [ ] Metrics compilation and analysis
- [ ] GO/NO-GO recommendation for Phase 1B
- [ ] Test suite optimization based on findings
- [ ] Documentation and handoff preparation

## Success Criteria Summary

**Phase 1A Completion Criteria:**
- ✅ All 10 test cases pass with expected results
- ✅ Overall F1 score ≥ 0.78
- ✅ Zero critical failures in edge case testing
- ✅ Performance meets baseline requirements
- ✅ Medical expert validation approval

**Quality Gate for Phase 1B:**
- IF F1 < 0.78 OR Recall < 0.70 → Proceed to Phase 1B
- IF F1 ≥ 0.78 AND Precision ≥ 0.80 → Phase 1A sufficient
- Document decision rationale with supporting metrics

This methodology ensures systematic validation while maintaining flexibility for future enhancements and provides clear decision criteria for project progression.