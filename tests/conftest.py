import asyncio
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import numpy as np

from app.graph_store import NetworkXGraphStore
from app.kv_store import JsonKvStore
from app.vector_store import NanoVectorStore


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_graph_path(temp_dir):
    """Provide a temporary path for graph storage."""
    return os.path.join(temp_dir, "test_graph.graphml")


@pytest.fixture
def temp_kv_path(temp_dir):
    """Provide a temporary path for KV storage."""
    return os.path.join(temp_dir, "test_kv.json")


@pytest.fixture
def temp_vector_db_path(temp_dir):
    """Provide a temporary path for vector DB storage."""
    return os.path.join(temp_dir, "test_vectors")


@pytest.fixture
def graph_store(temp_graph_path):
    """Create a NetworkXGraphStore instance for testing."""
    store = NetworkXGraphStore(temp_graph_path)
    # Loads automatically in __init__
    yield store


@pytest.fixture
def kv_store(temp_kv_path):
    """Create a JsonKvStore instance for testing."""
    store = JsonKvStore(temp_kv_path)
    # Loads automatically in __init__
    yield store


@pytest.fixture
def vector_store(temp_vector_db_path):
    """Create a NanoVectorStore instance for testing."""
    store = NanoVectorStore(temp_vector_db_path, dimensions=1536)
    yield store


@pytest.fixture
def mock_openai_llm():
    """Mock OpenAI LLM for testing without API calls."""
    mock_llm = MagicMock()

    # Mock embedding generation (plural - for batches)
    async def mock_get_embeddings(texts):
        return [np.random.rand(1536).tolist() for _ in texts]

    # Mock embedding generation (singular - for single text)
    async def mock_get_embedding(text):
        return np.random.rand(1536).tolist()

    # Mock completion
    async def mock_get_completion(text, **kwargs):
        return "Mock completion response"

    mock_llm.get_embeddings = AsyncMock(side_effect=mock_get_embeddings)
    mock_llm.get_embedding = AsyncMock(side_effect=mock_get_embedding)
    mock_llm.get_completion = AsyncMock(side_effect=mock_get_completion)

    return mock_llm


@pytest.fixture
def sample_entities():
    """Provide sample entity data for testing."""
    return [
        {
            "entity_name": "Python",
            "entity_type": "Programming Language",
            "description": "A high-level programming language",
            "source_id": "doc1"
        },
        {
            "entity_name": "JavaScript",
            "entity_type": "Programming Language",
            "description": "A scripting language for web development",
            "source_id": "doc1"
        },
        {
            "entity_name": "FastAPI",
            "entity_type": "Framework",
            "description": "A modern web framework for Python",
            "source_id": "doc2"
        }
    ]


@pytest.fixture
def sample_relationships():
    """Provide sample relationship data for testing."""
    return [
        {
            "source": "FastAPI",
            "target": "Python",
            "description": "FastAPI is built with Python",
            "keywords": "framework, built-with",
            "weight": 1.0,
            "source_id": "doc2"
        },
        {
            "source": "Python",
            "target": "JavaScript",
            "description": "Both are popular programming languages",
            "keywords": "comparison, alternatives",
            "weight": 0.5,
            "source_id": "doc1"
        }
    ]


@pytest.fixture
def sample_document_content():
    """Provide sample document content for testing."""
    return """# Python Programming

Python is a high-level, interpreted programming language known for its simplicity and readability.

## Features
- Easy to learn syntax
- Extensive standard library
- Large ecosystem of third-party packages

## Popular Frameworks
FastAPI is a modern web framework built with Python that enables building APIs quickly.

## Code Example
```python
def hello_world():
    print("Hello, World!")
```
"""


@pytest.fixture
def sample_excerpts():
    """Provide sample excerpts for testing."""
    return [
        "Python is a high-level programming language.",
        "FastAPI is a modern web framework for Python.",
        "JavaScript is used for web development.",
        "Both Python and JavaScript are popular languages."
    ]


@pytest.fixture
def large_entity_set():
    """Generate a large set of entities for performance testing."""
    entities = []
    for i in range(1000):
        entities.append({
            "entity_name": f"Entity_{i}",
            "entity_type": f"Type_{i % 10}",
            "description": f"Description for entity {i} " * 10,  # Make descriptions longer
            "source_id": f"doc_{i % 100}"
        })
    return entities


@pytest.fixture
def performance_timer():
    """Helper fixture to measure execution time."""
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = asyncio.get_event_loop().time()

        def stop(self):
            self.end_time = asyncio.get_event_loop().time()

        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None

    return Timer()
