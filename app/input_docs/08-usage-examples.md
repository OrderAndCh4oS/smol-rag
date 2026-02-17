**Title: SmolRAG Usage Examples**

---

### **1. Introduction to SmolRAG Usage**

This document provides practical examples of how to use SmolRAG in various scenarios. These examples demonstrate the flexibility and power of SmolRAG for different use cases, from simple document querying to more complex knowledge extraction and reasoning tasks.

The examples are designed to be easy to follow and adapt to your specific needs. Each example includes code snippets and explanations to help you understand how to implement similar functionality in your own applications.

---

### **2. Basic Setup and Initialization**

Before using SmolRAG, you need to set up your environment and initialize the system. Here's a basic example:

```python
# Import the SmolRag class
from app.smol_rag import SmolRag

# Initialize SmolRAG with default settings
rag = SmolRag()

# SmolRAG now uses preserve_markdown_code_excerpts as the default chunking strategy
# This strategy keeps code blocks intact and splits text at sentence boundaries
from app.chunking import preserve_markdown_code_excerpts, naive_overlap_excerpts

# The default is already preserve_markdown_code_excerpts, but you can specify it explicitly:
rag = SmolRag(excerpt_fn=preserve_markdown_code_excerpts)

# Or use the simpler naive chunking if preferred:
# rag = SmolRag(excerpt_fn=naive_overlap_excerpts)

# If you want to customize dimensions or other parameters
rag = SmolRag(
    dimensions=1536,
    excerpt_size=2000,
    overlap=200
)
```

This basic setup creates a SmolRAG instance with either default or custom settings. The instance is now ready to import documents and process queries.

---

### **3. Document Ingestion Example**

To use SmolRAG, you first need to ingest documents. Here's how to do it:

```python
# Import the SmolRag class
from app.smol_rag import SmolRag

# Initialize SmolRAG
rag = SmolRag()

# Import documents from the input_docs directory
rag.import_documents()

# You can also manually add documents to the input_docs directory before importing
import os
import shutil
from app.definitions import INPUT_DOCS_DIR

# Copy a document to the input_docs directory
source_file = "path/to/your/document.md"
destination = os.path.join(INPUT_DOCS_DIR, "document.md")
shutil.copy(source_file, destination)

# Then import documents
rag.import_documents()
```

This example shows how to import documents from the default input_docs directory. You can add documents to this directory manually or programmatically before importing.

---

### **4. Simple Query Example**

Once you have ingested documents, you can query them using the default vector search query:

```python
# Import the SmolRag class
from app.smol_rag import SmolRag

# Initialize SmolRAG
rag = SmolRag()

# Make sure documents are imported
rag.import_documents()

# Perform a simple query
result = rag.query("What is SmolRAG?")
print(result)

# Ask about specific features
result = rag.query("How does document chunking work in SmolRAG?")
print(result)

# Ask about use cases
result = rag.query("What are the main use cases for SmolRAG?")
print(result)
```

This example demonstrates how to perform simple queries using the default vector search method. This method is fast and works well for straightforward questions that are directly addressed in the documents.

---

### **5. Knowledge Graph Query Examples**

SmolRAG offers several knowledge graph-based query methods for more complex questions:

```python
# Import the SmolRag class
from app.smol_rag import SmolRag

# Initialize SmolRAG
rag = SmolRag()

# Make sure documents are imported
rag.import_documents()

# Local knowledge graph query (entity-focused)
result = rag.local_kg_query("What entities are related to document chunking?")
print(result)

# Global knowledge graph query (relationship-focused)
result = rag.global_kg_query("How are different components of SmolRAG connected?")
print(result)

# Hybrid knowledge graph query (combines local and global)
result = rag.hybrid_kg_query("What is the relationship between embeddings and queries?")
print(result)
```

These examples show how to use different knowledge graph query methods for different types of questions. Each method has its strengths and is suited to different types of queries.

---

### **6. Mix Query Example**

For the most comprehensive results, you can use the mix query method, which combines vector search and knowledge graph approaches:

```python
# Import the SmolRag class
from app.smol_rag import SmolRag

# Initialize SmolRAG
rag = SmolRag()

# Make sure documents are imported
rag.import_documents()

# Mix query (combines vector search and knowledge graph)
result = rag.mix_query("How does SmolRAG process and retrieve information?")
print(result)

# Complex question requiring both semantic search and structured knowledge
result = rag.mix_query("What are the advantages and limitations of different query types in SmolRAG?")
print(result)
```

The mix query method provides the most comprehensive results by combining the strengths of both vector search and knowledge graph approaches. It's particularly useful for complex questions that benefit from both semantic relevance and structured knowledge.

---

### **7. API Usage Example**

If you're using SmolRAG through its API, here's how to interact with it:

```python
import requests
import json

# Define the API endpoint
url = "http://localhost:8000/query"

# Vector search query
payload = {
    "text": "What is SmolRAG?",
    "query_type": "standard"
}
headers = {"Content-Type": "application/json"}
response = requests.post(url, data=json.dumps(payload), headers=headers)
print(response.json())

# Knowledge graph query
payload = {
    "text": "What entities are related to document chunking?",
    "query_type": "local_kg"
}
response = requests.post(url, data=json.dumps(payload), headers=headers)
print(response.json())

# Mix query
payload = {
    "text": "How does SmolRAG process and retrieve information?",
    "query_type": "mix"
}
response = requests.post(url, data=json.dumps(payload), headers=headers)
print(response.json())
```

This example shows how to interact with SmolRAG through its REST API using the Python requests library. The API supports all the same query types as the Python interface.

---

### **8. Document Management Example**

SmolRAG includes functionality for managing documents, including detecting changes and removing documents:

```python
# Import the SmolRag class
from app.smol_rag import SmolRag

# Initialize SmolRAG
rag = SmolRag()

# Import documents
rag.import_documents()

# If you update a document in the input_docs directory,
# SmolRAG will automatically detect the change and update its internal representation
# when you call import_documents() again
rag.import_documents()

# If you know the document ID, you can remove it directly
doc_id = "doc_4c3f8100da0b90c1a44c94e6b4ffa041"
rag.remove_document_by_id(doc_id)

# To get the document ID for a file path, you can use the source_to_doc_kv store
file_path = "app/input_docs/document.md"
if rag.source_to_doc_kv.has(file_path):
    doc_id = rag.source_to_doc_kv.get_by_key(file_path)
    rag.remove_document_by_id(doc_id)
```

This example demonstrates how to manage documents in SmolRAG, including handling updates and removing documents when needed.

---

### **9. Custom Chunking Strategy Example**

If the default chunking strategies don't meet your needs, you can implement a custom strategy:

```python
# Import the SmolRag class
from app.smol_rag import SmolRag

# Define a custom chunking function
def custom_chunking_strategy(text, excerpt_size, overlap):
    # Your custom chunking logic here
    # For example, a simple paragraph-based chunking:
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) <= excerpt_size:
            current_chunk += paragraph + "\n\n"
        else:
            chunks.append(current_chunk)
            current_chunk = paragraph + "\n\n"

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

# Initialize SmolRAG with the custom chunking strategy
rag = SmolRag(excerpt_fn=custom_chunking_strategy)

# Import documents
rag.import_documents()
```

This example shows how to implement and use a custom chunking strategy. This can be useful if you have specific requirements for how documents should be divided into chunks.

---

### **10. Integration with Web Applications**

Here's an example of integrating SmolRAG with a simple web application using Flask:

```python
from flask import Flask, request, jsonify
from app.smol_rag import SmolRag

app = Flask(__name__)
rag = SmolRag()

# Make sure documents are imported
rag.import_documents()

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query_text = data.get('text', '')
    query_type = data.get('query_type', 'standard')

    if not query_text:
        return jsonify({'error': 'Query text is required'}), 400

    try:
        if query_type == 'standard':
            result = rag.query(query_text)
        elif query_type == 'local_kg':
            result = rag.local_kg_query(query_text)
        elif query_type == 'global_kg':
            result = rag.global_kg_query(query_text)
        elif query_type == 'hybrid_kg':
            result = rag.hybrid_kg_query(query_text)
        elif query_type == 'mix':
            result = rag.mix_query(query_text)
        else:
            return jsonify({'error': f'Invalid query type: {query_type}'}), 400

        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

This example demonstrates how to integrate SmolRAG with a Flask web application, providing a simple API for querying documents.

---

### **11. Batch Processing Example**

If you need to process multiple queries in batch, here's how to do it efficiently:

```python
# Import the SmolRag class
from app.smol_rag import SmolRag

# Initialize SmolRAG
rag = SmolRag()

# Make sure documents are imported
rag.import_documents()

# Define a list of queries
queries = [
    {"text": "What is SmolRAG?", "type": "standard"},
    {"text": "How does document chunking work?", "type": "local_kg"},
    {"text": "What are the advantages of knowledge graphs?", "type": "global_kg"},
    {"text": "How do vector embeddings and knowledge graphs work together?", "type": "mix"}
]

# Process queries
results = []
for query in queries:
    if query["type"] == "standard":
        result = rag.query(query["text"])
    elif query["type"] == "local_kg":
        result = rag.local_kg_query(query["text"])
    elif query["type"] == "global_kg":
        result = rag.global_kg_query(query["text"])
    elif query["type"] == "hybrid_kg":
        result = rag.hybrid_kg_query(query["text"])
    elif query["type"] == "mix":
        result = rag.mix_query(query["text"])
    else:
        result = f"Invalid query type: {query['type']}"

    results.append({"query": query["text"], "type": query["type"], "result": result})

# Print results
for i, result in enumerate(results):
    print(f"Query {i+1}: {result['query']}")
    print(f"Type: {result['type']}")
    print(f"Result: {result['result']}")
    print("-" * 50)
```

This example shows how to process multiple queries in batch, which can be useful for testing or bulk processing.

---

### **12. Parallel Processing with Asyncio**

SmolRAG now uses asyncio for parallel processing during document ingestion, which significantly improves performance:

```python
import asyncio
from app.smol_rag import SmolRag

# Initialize SmolRAG
rag = SmolRag()

# The import_documents method uses asyncio internally
# It processes multiple documents, embeddings, and entity extractions in parallel
async def import_docs():
    await rag.import_documents()

# Run the async function
asyncio.run(import_docs())

# Or in an existing async context:
# await rag.import_documents()
```

This example demonstrates how SmolRAG leverages asyncio for parallel processing. The system automatically:
- Processes multiple documents simultaneously
- Generates embeddings for multiple excerpts concurrently
- Extracts entities and relationships from different excerpts in parallel

This parallel processing approach dramatically reduces ingestion time, especially for large document collections.

---

### **13. Advanced Configuration Example**

For advanced users, here's how to configure SmolRAG with custom components:

```python
# Import necessary classes
from app.smol_rag import SmolRag
from app.openai_llm import OpenAiLlm
from app.vector_store import NanoVectorStore
from app.graph_store import NetworkXGraphStore
from app.kv_store import JsonKvStore
from app.chunking import preserve_markdown_code_excerpts

# Custom paths
embeddings_path = "custom/path/to/embeddings.db"
entities_path = "custom/path/to/entities.db"
relationships_path = "custom/path/to/relationships.db"
kg_path = "custom/path/to/kg.db"
source_to_doc_path = "custom/path/to/source_to_doc.json"
doc_to_source_path = "custom/path/to/doc_to_source.json"
doc_to_excerpt_path = "custom/path/to/doc_to_excerpt.json"
excerpt_path = "custom/path/to/excerpt.json"
query_cache_path = "custom/path/to/query_cache.json"
embedding_cache_path = "custom/path/to/embedding_cache.json"

# Custom dimensions
dimensions = 1536

# Initialize custom components
llm = OpenAiLlm(
    "gpt-4.1-mini",
    "text-embedding-3-small",
    query_cache_kv=JsonKvStore(query_cache_path),
    embedding_cache_kv=JsonKvStore(embedding_cache_path)
)

embeddings_db = NanoVectorStore(embeddings_path, dimensions)
entities_db = NanoVectorStore(entities_path, dimensions)
relationships_db = NanoVectorStore(relationships_path, dimensions)

source_to_doc_kv = JsonKvStore(source_to_doc_path)
doc_to_source_kv = JsonKvStore(doc_to_source_path)
doc_to_excerpt_kv = JsonKvStore(doc_to_excerpt_path)
excerpt_kv = JsonKvStore(excerpt_path)

graph_db = NetworkXGraphStore(kg_path)

# Initialize SmolRAG with custom components
rag = SmolRag(
    excerpt_fn=preserve_markdown_code_excerpts,
    llm=llm,
    embeddings_db=embeddings_db,
    entities_db=entities_db,
    relationships_db=relationships_db,
    source_to_doc_kv=source_to_doc_kv,
    doc_to_source_kv=doc_to_source_kv,
    doc_to_excerpt_kv=doc_to_excerpt_kv,
    excerpt_kv=excerpt_kv,
    graph_db=graph_db,
    dimensions=dimensions,
    excerpt_size=2000,
    overlap=200
)

# Import documents and use as normal
rag.import_documents()
result = rag.query("What is SmolRAG?")
print(result)
```

This advanced example demonstrates how to configure SmolRAG with custom components and paths, providing maximum flexibility for integration into existing systems.

---

### **14. Conclusion**

These examples demonstrate the versatility and power of SmolRAG for various use cases. Whether you need simple document querying, complex knowledge extraction, or integration with web applications, SmolRAG provides the tools you need.

By understanding these examples and adapting them to your specific needs, you can leverage the full potential of SmolRAG to build powerful document retrieval and question-answering systems.
