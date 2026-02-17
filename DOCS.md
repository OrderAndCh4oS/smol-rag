# SmolRAG Documentation

This document provides comprehensive information for users and developers working with the SmolRAG project.

## Table of Contents

- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Standard Installation](#standard-installation)
  - [Docker Installation](#docker-installation)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
  - [Directory Structure](#directory-structure)
- [Usage](#usage)
  - [Document Ingestion](#document-ingestion)
  - [Querying Documents](#querying-documents)
  - [Query Types](#query-types)
- [API Reference](#api-reference)
  - [Endpoints](#endpoints)
  - [Request/Response Format](#requestresponse-format)
  - [Example Requests](#example-requests)
- [Architecture](#architecture)
  - [System Components](#system-components)
  - [Data Flow](#data-flow)
- [Testing](#testing)
  - [Running Tests](#running-tests)
  - [Creating Tests](#creating-tests)
- [Development](#development)
  - [Code Organization](#code-organization)
  - [Debugging Tips](#debugging-tips)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- Python 3.10 or higher
- OpenAI API key
- Git (for cloning the repository)

### Standard Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/minimal-light-rag.git
   cd minimal-light-rag
   ```

2. **Python Environment Setup**:
   - Create a virtual environment:
     ```bash
     python -m venv .venv
     ```
   - Activate the virtual environment:
     - Windows:
       ```bash
       .venv\Scripts\activate
       ```
     - macOS/Linux:
       ```bash
       source .venv/bin/activate
       ```
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```

3. **Create Required Directories**:
   ```bash
   mkdir -p app/data app/cache app/logs app/input_docs
   ```

4. **Configure Environment Variables**:
   - Copy the example environment file:
     ```bash
     cp .env.example .env
     ```
   - Edit the `.env` file and add your OpenAI API key

5. **Run the Application**:
   - Start the API server:
     ```bash
     uvicorn api.main:app --reload
     ```
   - The API will be available at http://localhost:8000

### Docker Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/minimal-light-rag.git
   cd minimal-light-rag
   ```

2. **Configure Environment Variables**:
   - Copy the example environment file:
     ```bash
     cp .env.example .env
     ```
   - Edit the `.env` file and add your OpenAI API key

3. **Build and Run the Docker Container**:
   ```bash
   docker build -t smol-rag .
   docker run -p 8000:8000 --env-file .env smol-rag
   ```

4. **Access the API**:
   - The API will be available at http://localhost:8000

## Configuration

### Environment Variables

SmolRAG requires the following environment variables:

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | Yes | None |
| `COMPLETION_MODEL` | OpenAI model for completions | No | gpt-4.1-mini |
| `EMBEDDING_MODEL` | OpenAI model for embeddings | No | text-embedding-3-small |

You can set these variables in a `.env` file in the project root.

### Directory Structure

SmolRAG expects the following directory structure:

```
minimal-light-rag/
├── app/
│   ├── data/         # Stores vector databases, knowledge graph, and key-value stores
│   ├── cache/        # Stores query and embedding caches
│   ├── logs/         # Stores log files
│   └── input_docs/   # Place documents here to be ingested by the system
├── api/              # API implementation
└── ...
```

## Usage

### Document Ingestion

To ingest documents into SmolRAG:

1. Place your documents in the `app/input_docs/` directory
2. Documents can be in various formats (txt, md, etc.)
3. Run the ingestion process:

```python
import asyncio
from app.smol_rag import SmolRag

# Initialize SmolRAG
rag = SmolRag()

# Import documents from the input_docs directory
# This is an async method, so it needs to be awaited
async def import_docs():
    await rag.import_documents()

# Run the async function
asyncio.run(import_docs())

# Or in an existing async context:
# await rag.import_documents()
```

### Full Rebuild (Wipe + Rescan)

Use the rebuild script when you want to reset cache/index state and fully rescan source documents:

```bash
# Recommended after cache/index behavior changes
python rebuild_rag.py

# Keep existing state and only run import
python rebuild_rag.py --keep-existing
```

### Querying Documents

After ingesting documents, you can query them:

```python
import asyncio
from app.smol_rag import SmolRag

# Initialize SmolRAG
rag = SmolRag()

# Query using the default method (vector search)
# All query methods are async and need to be awaited
async def run_queries():
    # Vector search query
    result = await rag.query("What is SmolRAG?")
    print(result)

    # Query using the knowledge graph
    kg_result = await rag.hybrid_kg_query("How does SmolRAG handle document updates?")
    print(kg_result)

# Run the async function
asyncio.run(run_queries())

# Or in an existing async context:
# result = await rag.query("What is SmolRAG?")
# kg_result = await rag.hybrid_kg_query("How does SmolRAG handle document updates?")
```

### Query Types

SmolRAG supports multiple query types, each optimized for different use cases:

1. **Vector Search Query** (`query`):
   ```python
   # Async method - must be awaited
   result = await rag.query("What is SmolRAG?")
   ```

2. **Local Knowledge Graph Query** (`local_kg_query`):
   ```python
   # Async method - must be awaited
   result = await rag.local_kg_query("What entities are related to document chunking?")
   ```

3. **Global Knowledge Graph Query** (`global_kg_query`):
   ```python
   # Async method - must be awaited
   result = await rag.global_kg_query("How are different components connected?")
   ```

4. **Hybrid Knowledge Graph Query** (`hybrid_kg_query`):
   ```python
   # Async method - must be awaited
   result = await rag.hybrid_kg_query("What is the relationship between embeddings and queries?")
   ```

5. **Mix Query** (`mix_query`):
   ```python
   # Async method - must be awaited
   result = await rag.mix_query("How does SmolRAG process and retrieve information?")
   ```

## API Reference

### Endpoints

SmolRAG exposes a REST API with the following endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Process a query using SmolRAG |

### Request/Response Format

**Request Format:**

```json
{
  "text": "Your query text here",
  "query_type": "standard"
}
```

The `query_type` parameter can be one of:
- `standard` (default): Uses vector search query
- `hybrid_kg`: Uses hybrid knowledge graph query
- `local_kg`: Uses local knowledge graph query
- `global_kg`: Uses global knowledge graph query
- `mix`: Uses mix query (combines vector search and knowledge graph)

**Response Format:**

```json
{
  "result": "The response text from SmolRAG"
}
```

### Example Requests

Using curl:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"text": "What is SmolRAG?", "query_type": "standard"}'
```

Using Python requests:

```python
import requests
import json

url = "http://localhost:8000/query"
payload = {
    "text": "What is SmolRAG?",
    "query_type": "standard"
}
headers = {"Content-Type": "application/json"}

response = requests.post(url, data=json.dumps(payload), headers=headers)
print(response.json())
```

## Architecture

### System Components

SmolRAG consists of the following main components:

1. **Document Processor**: Handles document ingestion, chunking, and summarization
2. **Vector Store**: Stores and retrieves document embeddings for semantic search
3. **Knowledge Graph**: Stores entities and relationships extracted from documents
4. **Query Processor**: Processes different types of queries and retrieves relevant information
5. **LLM Interface**: Communicates with OpenAI's API for embeddings and completions
6. **API Layer**: Exposes the functionality through a REST API

### Data Flow

1. **Document Ingestion**:
   - Documents are read from the input directory
   - Each document is split into overlapping chunks
   - Chunks are summarized and embedded
   - Entities and relationships are extracted and stored in the knowledge graph
   - The entire process is fully asynchronous, with both key-value store (JsonKvStore) and vector store (NanoVectorStore) operations completely asynchronous, using async/await patterns. This optimizes performance, especially for large document collections

2. **Query Processing**:
   - User submits a query through the API
   - Query is processed based on the specified query type
   - Relevant information is retrieved from the vector store and/or knowledge graph
   - Retrieved information is used to generate a response using the LLM

## Testing

### Running Tests

1. **Evaluation Tests**:
   - The system includes a test evaluation framework in `app/evaluation/`
   - Run evaluation on test set:
     ```bash
     python app/evaluation/evaluate_test_set.py
     ```
   - This evaluates the system's responses against a predefined test set

### Creating Tests

1. **Generate Test Set**:
   ```bash
   python app/evaluation/generate_test_set.py
   ```
   - This extracts pull quotes from documents and generates queries and expected responses

2. **Evaluate Test Set**:
   ```bash
   python app/evaluation/evaluate_test_set.py
   ```
   - The script generates responses, compares them against expected pull quotes, and computes an accuracy score

3. **Writing Unit Tests**:
   ```python
   # Import required modules
   from app.smol_rag import SmolRag
   import unittest

   class TestSmolRag(unittest.TestCase):
       def setUp(self):
           self.rag = SmolRag()

       def test_query(self):
           result = self.rag.query("Test query")
           self.assertIsNotNone(result, "Query should return a response")

   if __name__ == "__main__":
       unittest.main()
   ```

## Development

### Code Organization

SmolRAG's codebase is organized as follows:

```
minimal-light-rag/
├── app/
│   ├── smol_rag.py         # Main RAG implementation
│   ├── openai_llm.py       # OpenAI API integration
│   ├── vector_store.py     # Vector database implementation
│   ├── graph_store.py      # Knowledge graph implementation
│   ├── kv_store.py         # Key-value store implementation
│   ├── chunking.py         # Document chunking utilities
│   ├── prompts.py          # System prompts for various operations
│   ├── utilities.py        # General utility functions
│   ├── logger.py           # Logging configuration
│   ├── definitions.py      # Constants and path definitions
│   └── evaluation/         # Evaluation framework
├── api/
│   └── main.py             # FastAPI implementation
└── ...
```

### Key Classes and Methods

1. **SmolRag** (`app/smol_rag.py`):
   - `__init__()`: Initialize the RAG system
   - `async import_documents()`: Import documents from the input directory (asynchronous)
   - `async query()`: Vector search query (asynchronous)
   - `async local_kg_query()`: Local knowledge graph query (asynchronous)
   - `async global_kg_query()`: Global knowledge graph query (asynchronous)
   - `async hybrid_kg_query()`: Hybrid knowledge graph query (asynchronous)
   - `async mix_query()`: Mix query (combines vector search and knowledge graph) (asynchronous)
   - `async remove_document_by_id()`: Remove a document from the system (asynchronous)

2. **NanoVectorStore** (`app/vector_store.py`):
   - Handles vector embeddings and similarity search with asynchronous operations
   - Provides async methods for upsert, delete, query, and save operations

3. **NetworkXGraphStore** (`app/graph_store.py`):
   - Manages the knowledge graph

4. **JsonKvStore** (`app/kv_store.py`):
   - Manages key-value storage with asynchronous operations
   - Provides async methods for add, remove, has, equal, get_all, get_by_key, and save operations

5. **OpenAiLlm** (`app/openai_llm.py`):
   - Interfaces with OpenAI's API for embeddings and completions

### Debugging Tips

1. **Logging**:
   - The application uses Python's logging module
   - Logs are stored in `app/logs/`
   - The main log file is `main.log`
   - Log levels can be adjusted in `app/logger.py`

2. **Caching**:
   - Query and embedding results are cached to improve performance and reduce API costs
   - If you need a clean state, run `python rebuild_rag.py` to wipe cache/index files and rescan input docs
   - Cache implementations are in `app/kv_store.py`

3. **API Testing**:
   - Use the `requests.http` file to test API endpoints
   - The API exposes a single endpoint at `/query` that accepts POST requests
   - For more complex testing, use tools like Postman or write Python scripts using the requests library

4. **Common Issues**:
   - If you encounter OpenAI API rate limits, consider implementing a retry mechanism or reducing the number of concurrent requests
   - Large documents may cause memory issues; consider adjusting the chunking parameters
   - If the knowledge graph becomes too large, consider pruning less important entities and relationships

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to SmolRAG.

## License

SmolRAG is licensed under the MIT License. See [LICENSE](LICENSE) for the full license text.
