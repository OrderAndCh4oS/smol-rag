# SmolRAG Test Suite

This test suite provides comprehensive coverage for the SmolRAG implementation, with a focus on identifying and measuring performance bottlenecks before implementing optimizations.

## Test Structure

### Test Files

1. **`test_graph_store.py`** - NetworkX Graph Operations
   - Tests for: Bottleneck #1 (Synchronous graph operations blocking async code)
   - Tests for: Bottleneck #4 (N+1 graph query patterns)
   - Coverage: Node/edge operations, persistence, performance benchmarks

2. **`test_kv_store.py`** - JSON Key-Value Store
   - Tests for: Bottleneck #2 (Full JSON file rewrites on every save)
   - Coverage: CRUD operations, file I/O performance, concurrency, memory usage

3. **`test_vector_store.py`** - Vector Database Operations
   - Tests for: Bottleneck #3 (In-memory vector storage scaling)
   - Coverage: Vector CRUD, similarity search, memory usage, persistence

4. **`test_smol_rag.py`** - Main RAG System Integration
   - Tests for: Bottleneck #5 (No embedding API batching)
   - Tests for: Bottleneck #6 (Unbounded entity description concatenation)
   - Tests for: Bottleneck #7 (Query result caching disabled)
   - Coverage: Document ingestion, querying, integration tests

5. **`test_utilities.py`** - Utility Functions
   - Tests for: Bottleneck #9 (Repeated token counting)
   - Coverage: String operations, tokenization, helper functions

### Test Categories

Tests are organized into three main categories using pytest markers:

- **Baseline Tests**: Verify current functionality works correctly
- **Performance Tests**: Measure and benchmark bottlenecks (marked with `@pytest.mark.performance`)
- **Integration Tests**: Test end-to-end workflows (marked with `@pytest.mark.integration`)
- **Slow Tests**: Long-running tests that may be skipped (marked with `@pytest.mark.slow`)

## Running Tests

### Install Dependencies

```bash
pip install -r requirements.txt
```

The test dependencies (pytest, etc.) are already included in `requirements.txt`.

### Run All Tests

```bash
pytest
```

### Run Specific Test Categories

```bash
# Run only baseline functionality tests (fast)
pytest -m "not performance and not slow"

# Run performance benchmarks
pytest -m performance

# Run integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

### Run Specific Test Files

```bash
# Test only graph store
pytest tests/test_graph_store.py

# Test only KV store
pytest tests/test_kv_store.py

# Test with verbose output
pytest tests/test_graph_store.py -v

# Show print statements (useful for performance metrics)
pytest tests/test_graph_store.py -v -s
```

### Run Specific Tests

```bash
# Run a specific test
pytest tests/test_graph_store.py::TestGraphStorePerformance::test_sequential_node_additions_timing -v -s

# Run all tests in a class
pytest tests/test_kv_store.py::TestKvStoreFileIOPerformance -v -s
```

## Understanding Test Results

### Performance Benchmarks

Performance tests print timing information to help establish baselines:

```
Sequential additions: 0.0023s for 3 nodes
N+1 queries for 3 entities: 0.0015s
```

These metrics will be used to measure improvements after implementing optimizations.

### Expected Failures

Some tests may fail initially if they test behavior that isn't yet implemented:

- Tests for batched embedding calls (currently sequential)
- Tests for async graph operations (currently synchronous)
- Tests for query caching (currently disabled)

**This is expected!** These tests serve as specifications for the improvements we'll make.

## Test Development Workflow

The test suite follows a Test-Driven Development (TDD) approach:

1. **Baseline Tests** - Verify current functionality
2. **Performance Tests** - Measure current bottlenecks
3. **Implement Fixes** - Optimize the code
4. **Verify Improvements** - Re-run performance tests to measure gains

## Key Test Fixtures

The `conftest.py` file provides shared fixtures:

- `temp_dir` - Temporary directory for test files
- `graph_store` - Initialized NetworkXGraphStore
- `kv_store` - Initialized JsonKvStore
- `vector_store` - Initialized NanoVectorStore
- `mock_openai_llm` - Mocked LLM (no API calls)
- `sample_entities/relationships/excerpts` - Test data
- `large_entity_set` - Large dataset for performance testing
- `performance_timer` - Helper for timing measurements

## Continuous Integration

To run tests in CI/CD:

```bash
# Run with coverage
pytest --cov=app --cov-report=html

# Run with strict mode (treat warnings as errors)
pytest --strict-markers

# Run with multiple workers (parallel)
pytest -n auto
```

## Performance Benchmarking

To get detailed performance metrics:

```bash
# Run all performance tests with output
pytest -m performance -v -s > performance_baseline.txt

# Compare before/after optimization
pytest -m performance -v -s > performance_after_fix.txt
diff performance_baseline.txt performance_after_fix.txt
```

## Troubleshooting

### Tests Fail with "File not found"

Ensure you're running pytest from the project root directory:

```bash
cd /Users/orderandchaos/code/minimal-light-rag
pytest
```

### Tests Hang or Take Too Long

Skip slow tests:

```bash
pytest -m "not slow"
```

### Import Errors

Make sure the app module is in your Python path:

```bash
export PYTHONPATH=$PYTHONPATH:/Users/orderandchaos/code/minimal-light-rag
pytest
```

Or use pytest with the correct path resolution (pytest handles this automatically if run from project root).

## Next Steps

After establishing baseline performance metrics:

1. Review performance test output to confirm bottlenecks
2. Implement optimizations (async graph ops, batching, caching, etc.)
3. Re-run performance tests to verify improvements
4. Add regression tests to prevent performance degradation

## Contributing

When adding new tests:

1. Follow the existing test structure (Baseline, Performance, EdgeCases classes)
2. Add appropriate pytest markers (`@pytest.mark.performance`, etc.)
3. Include print statements in performance tests to show metrics
4. Document what bottleneck or feature the test covers
5. Update this README if adding new test files
