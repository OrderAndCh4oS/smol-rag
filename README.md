# Smol RAG

Smol RAG is a lightweight retrieval-augmented generation system heavily inspired by and borrowing prompts and patterns from LightRAG. It supports multiple query types and includes a robust pipeline for ingesting, embedding, and updating documents.

## Document Ingestion & Update Handling

The document ingestion process is central to the system. Documents are split into excerpts, summarised, embedded, and checked for updates using hash-based deduplication. Each document is tracked using a combination of its file path and a hash of its content. If a file path already exists but the hash has changed, Smol RAG automatically removes the old version and re-ingests the updated content. This ensures that queries always reflect the most recent state of your source materials without unnecessary reprocessing.

- **Smart Hashing for Change Detection**  
  Each document's full content is hashed to generate a unique ID. This ensures:
  - Identical content isn't reprocessed.
  - Changes in the file will trigger a re-ingestion.

- **Excerpt and Summary**  
  Each document is split into overlapping chunks (~2000 characters), which are individually summarised to improve downstream context quality.

- **Embeddings**  
  Each excerpt and its summary are embedded using OpenAI’s embedding API and stored in NanoVectorDB.

- **Entity & Relationship Extraction**  
  Each excerpt is analysed for structured entities and relationships. These are saved in a local knowledge graph using NetworkX and also embedded for similarity queries.

- **Update Lifecycle**  
  When a document is updated:
  1. The old doc is removed (its excerpts, embeddings, and KG entries).
  2. The new content is reprocessed and indexed.

---

## Query Types

Smol RAG supports multiple query methods that leverage both semantic embeddings and a structured knowledge graph (KG). These methods are designed to balance accuracy, flexibility, and reasoning depth depending on the type of user query.

### 1. **Vector Search Query** (`query`)

This method uses pure semantic similarity:

- The query is embedded into a vector using the OpenAI embedding API.
- The embedding is compared against excerpt embeddings in the vector DB.
- Top-k excerpts are selected and their summaries are retrieved.
- These excerpts and summaries form the context for a prompt to the LLM.

Use this when:
- You want quick results.
- The question is directly answerable from document content.
- You don’t need deeper relational understanding.

### 2. **Local Knowledge Graph Query** (`local_kg_query`)

This method focuses on **low-level keywords** from the query:

- Embeds keywords to search for relevant entities in the KG.
- Ranks entities by graph degree and relevance.
- Extracts connected relationships and associated excerpts.
- Provides structured CSV context about entities, relationships, and text.

Use this when:
- You care about **fine-grained entity-level** information.
- You need a graph-aware answer that highlights links between terms.

### 3. **Global Knowledge Graph Query** (`global_kg_query`)

This focuses on **high-level keywords**:

- Embeds and matches keywords with relationships in the KG.
- Retrieves top-ranked relationships based on weight and connectivity.
- Extracts linked entities and supporting excerpts.

Use this when:
- You're looking for a **bird’s-eye view** of topic interconnections.
- You want to reason about high-level concepts or broad themes.

### 4. **Hybrid Knowledge Graph Query** (`hybrid_kg_query`)

A combination of **both low-level and high-level** keyword approaches:

- Combines local and global KG query outputs.
- Selects top entities, relationships, and excerpts.
- Presents them together to the LLM.

Use this when:
- You want the **best of both KG worlds**: precision + abstraction.
- You have a query that blends specific terms with general topics.

### 5. **Mix Query** (`mix_query`)

This is the most comprehensive query type:

- Combines both **semantic search** (via vector embeddings) and **KG-based reasoning**.
- Retrieves relevant excerpts via vector search.
- Merges this with KG data from the hybrid KG query.
- Constructs a combined context for the LLM.

Use this when:
- You want **maximum context and coverage**.
- You have a complex query that benefits from both literal excerpts and conceptual links.

---

Each query type is optimised for a different information need. You can start simple with vector search and progressively move toward hybrid or mix queries when interpretability and deeper reasoning matter.


