### SmolRAG

SmolRAG is a lightweight retrieval‑augmented generation system inspired by LightRAG, focused on fast, up‑to‑date
querying of your own documents.

Documents are split into ~2,000‑character overlapping chunks that keep Markdown code blocks intact. Long paragraphs are
shortened at sentence boundaries so words never split. Each chunk is summarised to raise context quality before being
embedded in NanoVectorDB, and its entities and relationships are stored in a local NetworkX knowledge graph for
structured querying.

Change detection is automatic. Every file’s full content is hashed. If the path already exists but the hash changes, the
old embeddings and graph entries are deleted and the new content is reingested, so answers always reflect the latest
versions of the documents.

## Docs

For setup instructions and general guidance, please refer to [SmolRAG Docs](DOCS.md).

### Example RAG Documentation

Additional AI-generated documentation for SmolRag can be found in `app/input_docs`, they've been ingested into SmolRag 
for to demo purposes. 

If you want to learn more about SmolRAG's abilities and internal workings, you can run it and ask some questions. 

The docs provide detailed information about SmolRAG's architecture, features, and implementation details, including:

- Project overview
- Document ingestion process
- Excerpt embeddings
- Similarity search functionality
- Knowledge graph implementation
- Query types and usage
- Implementation details
- Usage examples
- API documentation
- Configuration options
- Performance considerations
- Troubleshooting guide


## Features

### Document Ingestion & Update Handling

Each document is split into overlapping chunks (approximately 2000 characters) using our
`preserve_markdown_code_excerpts` function, which is optimised for code documentation by keeping code blocks intact.

- Markdown code blocks are preserved in their entirety, ensuring code examples remain functional and readable.
- Text is then segmented into paragraphs and, if necessary, further divided into sentences at whitespace boundaries to
  avoid splitting words.
- Each excerpt is then individually summarised with the whole document provided as context; the summary preserves
  details about where the excerpt fits in.
- Documents are embedded and checked for updates using hash-based deduplication. Each document is tracked using a
  combination of its file path and a hash of its content. If a file path already exists but the hash has changed,
  SmolRAG automatically removes the old version and re-ingests the updated content to ensure that queries reflect the
  most recent state of your source materials without unnecessary reprocessing.
- The ingestion process is fully asynchronous, using asyncio and gather for parallel processing. This significantly
  improves data ingestion speed by processing multiple documents, generating embeddings, and extracting entities
  concurrently. Both the key-value store (JsonKvStore) and vector store (NanoVectorStore) operations are completely
  asynchronous, with all methods implemented using async/await patterns. This fully asynchronous approach optimizes the
  entire document ingestion pipeline, providing significant performance improvements especially for large document
  collections.

### Query Types

SmolRAG supports multiple query methods that leverage both semantic embeddings and a structured knowledge graph (KG).
These methods are designed to balance accuracy, flexibility, and reasoning depth depending on the type of user query.

#### 1. **Vector Search Query** (`query`)

This method uses pure semantic similarity:

- The query is embedded into a vector using the OpenAI embedding API.
- The embedding is compared against excerpt embeddings in the vector DB.
- Top-k excerpts are selected and their summaries are retrieved.
- These excerpts and summaries form the context for a prompt to the LLM.

Use this when:

- You want quick results.
- The question is directly answerable from document content.
- You don’t need deeper relational understanding.

#### 2. **Local Knowledge Graph Query** (`local_kg_query`)

This method focuses on **low-level keywords** from the query:

- Embeds keywords to search for relevant entities in the KG.
- Ranks entities by graph degree and relevance.
- Extracts connected relationships and associated excerpts.
- Provides structured CSV context about entities, relationships, and text.

Use this when:

- You care about **fine-grained entity-level** information.
- You need a graph-aware answer that highlights links between terms.

#### 3. **Global Knowledge Graph Query** (`global_kg_query`)

This focuses on **high-level keywords**:

- Embeds and matches keywords with relationships in the KG.
- Retrieves top-ranked relationships based on weight and connectivity.
- Extracts linked entities and supporting excerpts.

Use this when:

- You're looking for a **bird’s-eye view** of topic interconnections.
- You want to reason about high-level concepts or broad themes.

#### 4. **Hybrid Knowledge Graph Query** (`hybrid_kg_query`)

A combination of **both low-level and high-level** keyword approaches:

- Combines local and global KG query outputs.
- Selects top entities, relationships, and excerpts.
- Presents them together to the LLM.

Use this when:

- You want the **best of both KG worlds**: precision + abstraction.
- You have a query that blends specific terms with general topics.

#### 5. **Mix Query** (`mix_query`)

This is the most comprehensive query type:

- Combines both **semantic search** (via vector embeddings) and **KG-based reasoning**.
- Retrieves relevant excerpts via vector search.
- Merges this with KG data from the hybrid KG query.
- Constructs a combined context for the LLM.

Use this when:

- You want **maximum context and coverage**.
- You have a complex query that benefits from both literal excerpts and conceptual links.

