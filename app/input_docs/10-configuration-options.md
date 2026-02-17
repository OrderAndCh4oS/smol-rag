**Title: SmolRAG Configuration Options**

---

### **1. Introduction to SmolRAG Configuration**

SmolRAG offers a variety of configuration options that allow you to customize its behavior to suit your specific needs. These options range from basic settings like model selection and directory paths to advanced customizations of the chunking, embedding, and query processes.

This document provides a comprehensive overview of the available configuration options, explaining what each option does, its default value, and how to set it. Understanding these options will help you optimize SmolRAG for your particular use case, whether you're prioritizing accuracy, speed, or resource efficiency.

---

### **2. Environment Variables**

SmolRAG uses environment variables for core configuration settings. These can be set in your system environment or in a `.env` file in the project root directory.

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | Yes | None |
| `COMPLETION_MODEL` | OpenAI model for completions | No | gpt-4.1-mini |
| `EMBEDDING_MODEL` | OpenAI model for embeddings | No | text-embedding-3-small |

**Example `.env` file**:
```
OPENAI_API_KEY=sk-your-api-key
COMPLETION_MODEL=gpt-4.1-mini
EMBEDDING_MODEL=text-embedding-3-large
```

These environment variables are loaded automatically when SmolRAG starts, and they take precedence over the default values defined in the code.

---

### **3. Directory Structure Configuration**

SmolRAG uses a specific directory structure for storing documents, data, and logs. These paths are defined in `app/definitions.py` and can be customized if needed.

| Path Constant | Default Value | Description |
|---------------|---------------|-------------|
| `INPUT_DOCS_DIR` | `app/input_docs` | Directory where input documents are stored |
| `DATA_DIR` | `app/data` | Directory for storing vector databases and other data |
| `CACHE_DIR` | `app/cache` | Directory for storing caches |
| `LOG_DIR` | `app/logs` | Directory for storing log files |

If you need to change these paths, you can modify the `definitions.py` file or override them when initializing SmolRAG by providing custom paths to the constructor.

---

### **4. SmolRag Initialization Parameters**

The `SmolRag` class constructor accepts several parameters that allow you to customize its behavior:

```python
def __init__(
    self,
    excerpt_fn=None,
    llm=None,
    embeddings_db=None,
    entities_db=None,
    relationships_db=None,
    source_to_doc_kv=None,
    doc_to_source_kv=None,
    doc_to_excerpt_kv=None,
    excerpt_kv=None,
    query_cache_kv=None,
    embedding_cache_kv=None,
    graph_db=None,
    dimensions=None,
    excerpt_size=2000,
    overlap=200
)
```

**Key Parameters**:

- `excerpt_fn`: Function for chunking documents (default: `preserve_markdown_code_excerpts`)
- `llm`: LLM interface instance (default: new `OpenAiLlm` instance)
- `dimensions`: Embedding dimensions (default: 1536)
- `excerpt_size`: Target size for document chunks in characters (default: 2000)
- `overlap`: Overlap between chunks in characters (default: 200)

These parameters allow you to customize the core functionality of SmolRAG without modifying the source code.

---

### **5. Chunking Configuration**

Document chunking is a critical part of the RAG process, and SmolRAG provides several options for customizing it:

**Built-in Chunking Functions**:

1. `naive_overlap_excerpts(text, excerpt_size, overlap)`: A simple chunking strategy that splits text at regular intervals with overlap.
2. `preserve_markdown_code_excerpts(text, excerpt_size, overlap)`: An advanced strategy that respects Markdown structure and code blocks, which:
   - Identifies and extracts fenced code blocks (``` ... ```) from the document
   - Keeps entire code blocks intact, ensuring they remain functional and readable
   - Merges code blocks with neighboring paragraphs when they fit within the chunk size limit
   - Splits plain-text paragraphs at sentence boundaries when necessary
   - Applies optional overlap between chunks to maintain context continuity

**Chunking Parameters**:

- `excerpt_size`: Target size for document chunks in characters (default: 2000)
- `overlap`: Overlap between chunks in characters (default: 200)

**Custom Chunking Function**:

You can implement your own chunking function and pass it to the `SmolRag` constructor:

```python
def custom_chunking_strategy(text, excerpt_size, overlap):
    # Your custom chunking logic here
    # ...
    return chunks

rag = SmolRag(excerpt_fn=custom_chunking_strategy)
```

Your custom function should accept three parameters (`text`, `excerpt_size`, `overlap`) and return a list of text chunks.

---

### **6. LLM Configuration**

SmolRAG uses OpenAI's API for language model capabilities. You can configure the models used and other LLM-related settings:

**Model Selection**:

- `COMPLETION_MODEL`: The model used for completions (default: gpt-4.1-mini)
- `EMBEDDING_MODEL`: The model used for embeddings (default: text-embedding-3-small)

**Custom LLM Interface**:

You can create a custom `OpenAiLlm` instance with specific settings:

```python
from app.openai_llm import OpenAiLlm
from app.kv_store import JsonKvStore

llm = OpenAiLlm(
    completion_model="gpt-4.1-mini",
    embedding_model="text-embedding-3-large",
    query_cache_kv=JsonKvStore("custom/path/to/query_cache.json"),
    embedding_cache_kv=JsonKvStore("custom/path/to/embedding_cache.json")
)

rag = SmolRag(llm=llm)
```

This allows you to customize the models used, caching behavior, and other LLM-related settings.

---

### **7. Vector Store Configuration**

The vector store is responsible for storing and retrieving embeddings. SmolRAG uses a lightweight implementation called `NanoVectorStore` that can be configured in several ways:

**Basic Configuration**:

- `dimensions`: The dimensionality of the embeddings (default: 1536)

**Custom Vector Store**:

You can create a custom `NanoVectorStore` instance with specific settings:

```python
from app.vector_store import NanoVectorStore

embeddings_db = NanoVectorStore("custom/path/to/embeddings.db", dimensions=1536)
entities_db = NanoVectorStore("custom/path/to/entities.db", dimensions=1536)
relationships_db = NanoVectorStore("custom/path/to/relationships.db", dimensions=1536)

rag = SmolRag(
    embeddings_db=embeddings_db,
    entities_db=entities_db,
    relationships_db=relationships_db
)
```

This allows you to customize the storage location and dimensionality of the vector stores.

---

### **8. Knowledge Graph Configuration**

The knowledge graph stores entities and relationships extracted from documents. SmolRAG uses a NetworkX-based implementation that can be configured:

**Custom Graph Store**:

```python
from app.graph_store import NetworkXGraphStore

graph_db = NetworkXGraphStore("custom/path/to/kg.db")

rag = SmolRag(graph_db=graph_db)
```

This allows you to customize the storage location of the knowledge graph.

**Graph Extraction Parameters**:

The entity and relationship extraction process is guided by prompts defined in `prompts.py`. If you need to customize this process, you can modify these prompts or extend the `SmolRag` class to override the extraction methods.

---

### **9. Key-Value Store Configuration**

SmolRAG uses several key-value stores for metadata, mappings, and caching. These can be customized:

**Custom Key-Value Stores**:

```python
from app.kv_store import JsonKvStore

source_to_doc_kv = JsonKvStore("custom/path/to/source_to_doc.json")
doc_to_source_kv = JsonKvStore("custom/path/to/doc_to_source.json")
doc_to_excerpt_kv = JsonKvStore("custom/path/to/doc_to_excerpt.json")
excerpt_kv = JsonKvStore("custom/path/to/excerpt.json")

rag = SmolRag(
    source_to_doc_kv=source_to_doc_kv,
    doc_to_source_kv=doc_to_source_kv,
    doc_to_excerpt_kv=doc_to_excerpt_kv,
    excerpt_kv=excerpt_kv
)
```

This allows you to customize the storage location of the key-value stores.

---

### **10. Query Configuration**

SmolRAG's query processing can be configured in several ways:

**Query Types**:

- `query()`: Vector search query
- `local_kg_query()`: Local knowledge graph query
- `global_kg_query()`: Global knowledge graph query
- `hybrid_kg_query()`: Hybrid knowledge graph query
- `mix_query()`: Mix query (combines vector search and knowledge graph)

**Query Parameters**:

The query methods don't have explicit parameters for customization, but you can modify their behavior by customizing the underlying components (LLM, vector store, etc.) or by extending the `SmolRag` class to override the query methods.

**System Prompts**:

The query processing is guided by prompts defined in `prompts.py`. If you need to customize this process, you can modify these prompts or extend the `SmolRag` class to override the query methods.

---

### **11. Logging Configuration**

SmolRAG includes a logging system that can be configured:

**Log Level**:

The log level can be set in `app/logger.py`:

```python
def set_logger(log_file_name):
    # ...
    logger.setLevel(logging.INFO)  # Change to logging.DEBUG for more verbose logging
    # ...
```

**Log File**:

The log file is specified when calling `set_logger()`:

```python
from app.logger import set_logger

set_logger("custom_log.log")
```

**Custom Logging**:

If you need more advanced logging configuration, you can modify `app/logger.py` or implement your own logging system.

---

### **12. API Configuration**

If you're using the SmolRAG API, you can configure it in several ways:

**Server Configuration**:

When starting the API server, you can specify the host and port:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**FastAPI Configuration**:

The FastAPI application in `api/main.py` can be customized with additional middleware, error handlers, etc.:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="SmolRag API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware, error handlers, etc.
```

**Query Endpoint Configuration**:

The query endpoint in `api/main.py` can be customized to add additional parameters, validation, etc.:

```python
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    text: str = Field(..., description="The query text")
    query_type: str = Field("standard", description="The query type")
    max_results: int = Field(5, description="Maximum number of results to return")

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    # Process query with custom parameters
    # ...
```

---

### **13. Parallel Processing Configuration**

SmolRAG uses Python's asyncio library for parallel processing during document ingestion, which significantly improves performance:

**Asyncio Implementation**:

- Multiple documents are processed simultaneously
- Embedding and completion requests are executed concurrently
- The `asyncio.gather()` function combines multiple asynchronous tasks
- Rate limiting is applied to API calls to prevent throttling

**Rate Limiting Configuration**:

```python
# In SmolRag.__init__
self.llm_limiter = AsyncLimiter(max_rate=100, time_period=1)
```

You can adjust the rate limiting parameters by extending the `SmolRag` class:

```python
from app.smol_rag import SmolRag
from aiolimiter import AsyncLimiter

class CustomRateLimitedSmolRag(SmolRag):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Customize rate limiting
        self.llm_limiter = AsyncLimiter(max_rate=50, time_period=1)
```

This parallel processing approach dramatically reduces ingestion time, especially for large document collections.

---

### **14. Advanced Customization**

For more advanced customization, you can extend the core classes of SmolRAG:

**Extending SmolRag**:

```python
from app.smol_rag import SmolRag

class CustomSmolRag(SmolRag):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add custom initialization

    def custom_query(self, text):
        # Implement custom query method
        # ...

    # Override existing methods
    def query(self, text):
        # Custom implementation of query
        # ...
```

**Custom Components**:

You can also implement custom versions of the core components:

- Custom LLM interface
- Custom vector store
- Custom knowledge graph store
- Custom key-value store
- Custom chunking strategies

These custom components can be passed to the `SmolRag` constructor to replace the default implementations.

---

### **15. Configuration Best Practices**

Here are some best practices for configuring SmolRAG:

**Performance Optimization**:
- Use a smaller embedding model if speed is more important than accuracy
- Adjust chunk size and overlap based on your document characteristics
- Implement caching for frequent queries
- Use the appropriate query type for each use case

**Resource Efficiency**:
- Monitor memory usage, especially for large document collections
- Consider using a more efficient chunking strategy for large documents
- Implement rate limiting for the API to prevent excessive API calls
- Use appropriate token limits for summarization and query processing

**Customization Strategy**:
- Start with the default configuration and adjust as needed
- Test different configurations to find the optimal settings for your use case
- Document your custom configuration for future reference
- Consider creating a configuration file or module for your specific deployment

By following these best practices, you can optimize SmolRAG for your specific needs while maintaining its core functionality and performance.
