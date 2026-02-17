**Title: SmolRAG Troubleshooting Guide**

---

### **1. Introduction to Troubleshooting SmolRAG**

Even with careful implementation and configuration, you may encounter issues when working with SmolRAG. This troubleshooting guide aims to help you identify and resolve common problems that can occur during document ingestion, querying, and general system operation.

The guide is organized by problem category, with each section describing common symptoms, potential causes, and recommended solutions. By following this guide, you should be able to diagnose and fix most issues you encounter with SmolRAG, ensuring smooth operation of your retrieval-augmented generation system.

---

### **2. Installation and Setup Issues**

**Symptom: Import errors when trying to use SmolRAG**

*Potential Causes:*
- Missing dependencies
- Incorrect Python version
- Incorrect installation path

*Solutions:*
1. Ensure you're using Python 3.10 or higher:
   ```bash
   python --version
   ```

2. Reinstall dependencies with:
   ```bash
   pip install -r requirements.txt
   ```

3. Check your Python path:
   ```python
   import sys
   print(sys.path)
   ```

**Symptom: Directory structure errors**

*Potential Causes:*
- Missing required directories
- Incorrect permissions

*Solutions:*
1. Create the required directories:
   ```bash
   mkdir -p app/data app/cache app/logs app/input_docs
   ```

2. Check directory permissions:
   ```bash
   ls -la app/
   ```

3. Ensure the directories are writable by the current user:
   ```bash
   chmod -R u+w app/data app/cache app/logs app/input_docs
   ```

**Symptom: Environment variable errors**

*Potential Causes:*
- Missing or incorrect OpenAI API key
- Environment variables not loaded

*Solutions:*
1. Check if your `.env` file exists and contains the required variables:
   ```bash
   cat .env
   ```

2. Ensure the API key is correctly formatted and valid:
   ```
   OPENAI_API_KEY=sk-your-api-key
   ```

3. Try setting the environment variable directly:
   ```bash
   export OPENAI_API_KEY=sk-your-api-key
   ```

4. Verify the environment variable is loaded:
   ```python
   import os
   print(os.environ.get("OPENAI_API_KEY"))
   ```

---

### **3. Document Ingestion Issues**

**Symptom: Documents not being ingested**

*Potential Causes:*
- Documents not in the correct directory
- Unsupported file formats
- Permission issues

*Solutions:*
1. Verify documents are in the correct directory:
   ```bash
   ls -la app/input_docs/
   ```

2. Ensure documents are in supported formats (e.g., .txt, .md)

3. Check file permissions:
   ```bash
   chmod -R u+r app/input_docs/
   ```

**Symptom: Slow document ingestion**

*Potential Causes:*
- Large documents
- Many documents being processed at once
- API rate limits
- Insufficient resources

*Solutions:*
1. Process documents in smaller batches:
   ```python
   # Process documents in batches
   import os
   from app.definitions import INPUT_DOCS_DIR

   files = [f for f in os.listdir(INPUT_DOCS_DIR) if os.path.isfile(os.path.join(INPUT_DOCS_DIR, f))]
   batch_size = 5

   for i in range(0, len(files), batch_size):
       batch = files[i:i+batch_size]
       # Process batch
       print(f"Processing batch {i//batch_size + 1}/{(len(files) + batch_size - 1)//batch_size}")
   ```

2. Implement rate limiting for API calls:
   ```python
   import time

   def rate_limited_api_call(func, *args, **kwargs):
       result = func(*args, **kwargs)
       time.sleep(0.1)  # Sleep to avoid hitting rate limits
       return result
   ```

3. Monitor resource usage and adjust accordingly:
   ```python
   import psutil

   def check_resources():
       cpu_percent = psutil.cpu_percent()
       memory_percent = psutil.virtual_memory().percent
       print(f"CPU: {cpu_percent}%, Memory: {memory_percent}%")

       if memory_percent > 90:
           print("Warning: High memory usage")
   ```

**Symptom: Errors during entity extraction**

*Potential Causes:*
- LLM API errors
- Malformed content
- Timeout issues

*Solutions:*
1. Implement robust error handling:
   ```python
   try:
       result = llm.get_completion(prompt)
   except Exception as e:
       logger.error(f"Error during entity extraction: {e}")
       result = ""  # Provide a default or retry
   ```

2. Check for and handle malformed content:
   ```python
   def sanitize_content(content):
       # Remove problematic characters or patterns
       return content
   ```

3. Implement retries with exponential backoff:
   ```python
   import time

   def retry_with_backoff(func, max_retries=3):
       retries = 0
       while retries < max_retries:
           try:
               return func()
           except Exception as e:
               wait_time = 2 ** retries
               print(f"Error: {e}. Retrying in {wait_time} seconds...")
               time.sleep(wait_time)
               retries += 1

       # If we get here, all retries failed
       raise Exception("Max retries exceeded")
   ```

---

### **4. Vector Store Issues**

**Symptom: Vector store errors or corruption**

*Potential Causes:*
- Disk space issues
- File permission problems
- Concurrent access issues
- Power loss during write operations

*Solutions:*
1. Check disk space:
   ```bash
   df -h
   ```

2. Verify file permissions:
   ```bash
   ls -la app/data/
   ```

3. Implement file locking for concurrent access:
   ```python
   import fcntl

   def with_file_lock(file_path, callback):
       with open(file_path, 'r+') as f:
           try:
               fcntl.flock(f, fcntl.LOCK_EX)
               return callback(f)
           finally:
               fcntl.flock(f, fcntl.LOCK_UN)
   ```

4. Implement backup and recovery:
   ```python
   import shutil
   import os

   def backup_vector_store(vector_store_path, backup_dir):
       os.makedirs(backup_dir, exist_ok=True)
       backup_path = os.path.join(backup_dir, os.path.basename(vector_store_path) + '.bak')
       shutil.copy2(vector_store_path, backup_path)
       return backup_path
   ```

**Symptom: Missing or incorrect embeddings**

*Potential Causes:*
- Embedding generation failures
- Embedding model issues
- Caching problems

*Solutions:*
1. Verify embeddings exist:
   ```python
   def check_embeddings(rag, doc_id):
       excerpt_ids = rag.doc_to_excerpt_kv.get_by_key(doc_id)
       if not excerpt_ids:
           print(f"No excerpts found for document {doc_id}")
           return False

       for excerpt_id in excerpt_ids:
           results = rag.embeddings_db.query(query=None, filter_func=lambda x: x["__id__"] == excerpt_id)
           if not results:
               print(f"No embedding found for excerpt {excerpt_id}")
               return False

       return True
   ```

2. Clear embedding cache and regenerate:
   ```python
   import os
   from app.definitions import CACHE_DIR

   def clear_embedding_cache():
       cache_path = os.path.join(CACHE_DIR, "embedding_cache.json")
       if os.path.exists(cache_path):
           os.remove(cache_path)
           print(f"Removed embedding cache at {cache_path}")
   ```

3. Check embedding dimensions:
   ```python
   def verify_embedding_dimensions(rag, expected_dim=1536):
       # Get a sample embedding
       results = rag.embeddings_db.query(query=None, top_k=1)
       if not results:
           print("No embeddings found")
           return False

       sample_id = results[0]["__id__"]
       sample_vector = results[0]["__vector__"]

       actual_dim = len(sample_vector)
       if actual_dim != expected_dim:
           print(f"Dimension mismatch: expected {expected_dim}, got {actual_dim}")
           return False

       return True
   ```

---

### **5. Knowledge Graph Issues**

**Symptom: Missing or incorrect entities and relationships**

*Potential Causes:*
- Entity extraction failures
- Graph storage issues
- Prompt engineering problems

*Solutions:*
1. Check entity extraction prompt:
   ```python
   from app.prompts import get_extract_entities_prompt

   # Print the prompt for a sample excerpt
   sample_excerpt = "Your sample text here"
   print(get_extract_entities_prompt(sample_excerpt))
   ```

2. Manually test entity extraction:
   ```python
   from app.openai_llm import OpenAiLlm
   from app.prompts import get_extract_entities_prompt

   llm = OpenAiLlm()
   sample_excerpt = "Your sample text here"
   prompt = get_extract_entities_prompt(sample_excerpt)
   result = llm.get_completion(prompt)
   print(result)
   ```

3. Verify graph structure:
   ```python
   def inspect_graph(rag):
       print(f"Graph has {rag.graph.graph.number_of_nodes()} nodes and {rag.graph.graph.number_of_edges()} edges")

       # Print some sample nodes
       for i, node in enumerate(list(rag.graph.graph.nodes())[:5]):
           print(f"Node {i}: {node}")
           print(f"  Attributes: {rag.graph.graph.nodes[node]}")

       # Print some sample edges
       for i, edge in enumerate(list(rag.graph.graph.edges())[:5]):
           print(f"Edge {i}: {edge}")
           print(f"  Attributes: {rag.graph.graph.edges[edge]}")
   ```

**Symptom: Knowledge graph query returns unexpected results**

*Potential Causes:*
- Keyword extraction issues
- Graph traversal problems
- Ranking issues

*Solutions:*
1. Debug keyword extraction:
   ```python
   from app.prompts import get_high_low_level_keywords_prompt
   from app.openai_llm import OpenAiLlm
   from app.utilities import extract_json_from_text

   llm = OpenAiLlm()
   query = "Your query here"
   prompt = get_high_low_level_keywords_prompt(query)
   result = llm.get_completion(prompt)
   keyword_data = extract_json_from_text(result)
   print(keyword_data)
   ```

2. Trace graph traversal:
   ```python
   def trace_kg_query(rag, query):
       # Get keywords
       prompt = rag.prompts.get_high_low_level_keywords_prompt(query)
       result = rag.llm.get_completion(prompt)
       keyword_data = rag.utilities.extract_json_from_text(result)

       print(f"Low-level keywords: {keyword_data.get('low_level_keywords', [])}")
       print(f"High-level keywords: {keyword_data.get('high_level_keywords', [])}")

       # Trace entity search
       ll_keywords = keyword_data.get("low_level_keywords", [])
       if ll_keywords:
           ll_embedding = rag.llm.get_embedding(ll_keywords)
           ll_results = rag.entities_db.query(query=ll_embedding, top_k=5)
           print("Top entity matches:")
           for r in ll_results:
               print(f"  {r['__entity_name__']} (score: {r['__score__']:.4f})")

       # Trace relationship search
       hl_keywords = keyword_data.get("high_level_keywords", [])
       if hl_keywords:
           hl_embedding = rag.llm.get_embedding(hl_keywords)
           hl_results = rag.relationships_db.query(query=hl_embedding, top_k=5)
           print("Top relationship matches:")
           for r in hl_results:
               print(f"  {r['__source__']} -> {r['__target__']} (score: {r['__score__']:.4f})")
   ```

3. Adjust ranking parameters:
   ```python
   # Modify the ranking logic in your custom SmolRag subclass
   class CustomSmolRag(SmolRag):
       def _get_entities_from_relationships(self, kg_dataset):
           # ... existing code ...

           # Adjust ranking to prioritize degree more than before
           data = sorted(data, key=lambda x: (x["rank"] * 2, x["weight"]), reverse=True)

           # ... rest of the method ...
   ```

---

### **6. Query Issues**

**Symptom: Queries return irrelevant or incorrect information**

*Potential Causes:*
- Poor document quality
- Inappropriate query type
- Embedding quality issues
- Prompt engineering problems

*Solutions:*
1. Try different query types:
   ```python
   query = "Your query here"

   # Try all query types
   standard_result = rag.query(query)
   local_kg_result = rag.local_kg_query(query)
   global_kg_result = rag.global_kg_query(query)
   hybrid_kg_result = rag.hybrid_kg_query(query)
   mix_result = rag.mix_query(query)

   # Compare results
   print("Standard Query Result:", standard_result)
   print("Local KG Query Result:", local_kg_result)
   print("Global KG Query Result:", global_kg_result)
   print("Hybrid KG Query Result:", hybrid_kg_result)
   print("Mix Query Result:", mix_result)
   ```

2. Inspect retrieved excerpts:
   ```python
   def inspect_query_excerpts(rag, query):
       embedding = rag.llm.get_embedding(query)
       embedding_array = np.array(embedding)
       results = rag.embeddings_db.query(query=embedding_array, top_k=5)

       print(f"Top {len(results)} excerpts for query: '{query}'")
       for i, result in enumerate(results):
           excerpt_id = result["__id__"]
           score = result["__score__"]
           excerpt_data = rag.excerpt_kv.get_by_key(excerpt_id)

           print(f"\nExcerpt {i+1} (ID: {excerpt_id}, Score: {score:.4f}):")
           print(f"Summary: {excerpt_data['summary']}")
           print(f"Excerpt: {excerpt_data['excerpt'][:200]}...")
   ```

3. Adjust similarity threshold:
   ```python
   # Modify the threshold in your custom SmolRag subclass
   class CustomSmolRag(SmolRag):
       def _get_query_excerpts(self, text):
           embedding = self.llm.get_embedding(text)
           embedding_array = np.array(embedding)
           # Increase threshold for higher precision
           results = self.embeddings_db.query(query=embedding_array, top_k=5, better_than_threshold=0.05)
           excerpts = [self.excerpt_kv.get_by_key(result["__id__"]) for result in results]
           excerpts = truncate_list_by_token_size(excerpts, get_text_for_row=lambda x: x["excerpt"], max_token_size=4000)
           return excerpts
   ```

**Symptom: Queries are slow**

*Potential Causes:*
- Large vector database
- Complex knowledge graph
- API latency
- Resource constraints

*Solutions:*
1. Implement query caching:
   ```python
   import hashlib
   import json
   import os

   class QueryCache:
       def __init__(self, cache_dir):
           self.cache_dir = cache_dir
           os.makedirs(cache_dir, exist_ok=True)

       def get_cache_key(self, query_text, query_type, model, context):
           key = f"{query_text}_{query_type}_{model}_{context}"
           return hashlib.md5(key.encode()).hexdigest()

       def get_from_cache(self, query_text, query_type, model, context):
           key = self.get_cache_key(query_text, query_type, model, context)
           cache_file = os.path.join(self.cache_dir, f"{key}.json")

           if os.path.exists(cache_file):
               with open(cache_file, 'r') as f:
                   return json.load(f)
           return None

       def save_to_cache(self, query_text, query_type, model, context, result):
           key = self.get_cache_key(query_text, query_type, model, context)
           cache_file = os.path.join(self.cache_dir, f"{key}.json")

           with open(cache_file, 'w') as f:
               json.dump(result, f)
   ```

2. Optimize vector search:
   ```python
   # Use approximate nearest neighbor search for larger datasets
   # This is a simplified example - in practice, you would use a library like FAISS
   def approximate_vector_search(vectors, query_vector, top_k=5):
       # Randomly sample a subset of vectors for initial filtering
       sample_size = min(1000, len(vectors))
       sample_indices = random.sample(range(len(vectors)), sample_size)
       sample_vectors = [vectors[i] for i in sample_indices]

       # Compute similarities for the sample
       similarities = [np.dot(v, query_vector) / (np.linalg.norm(v) * np.linalg.norm(query_vector)) for v in sample_vectors]

       # Get top candidates from the sample
       top_sample_indices = np.argsort(similarities)[-top_k*2:]
       top_candidates = [sample_indices[i] for i in top_sample_indices]

       # Compute exact similarities for the top candidates
       candidate_vectors = [vectors[i] for i in top_candidates]
       candidate_similarities = [np.dot(v, query_vector) / (np.linalg.norm(v) * np.linalg.norm(query_vector)) for v in candidate_vectors]

       # Get final top-k
       top_k_indices = np.argsort(candidate_similarities)[-top_k:]
       return [top_candidates[i] for i in top_k_indices]
   ```

3. Use a simpler query type for time-sensitive applications:
   ```python
   def fast_query(rag, query_text):
       # Use the standard vector search query which is typically faster
       return rag.query(query_text)
   ```

**Symptom: Out of memory errors during queries**

*Potential Causes:*
- Large context size
- Memory leaks
- Insufficient system resources

*Solutions:*
1. Limit context size:
   ```python
   # Modify the context truncation in your custom SmolRag subclass
   class CustomSmolRag(SmolRag):
       def _get_excerpt_context(self, excerpts):
           # Limit to fewer excerpts
           excerpts = excerpts[:3]

           context = ""
           for excerpt in excerpts:
               context += inspect.cleandoc(f"""
                   ## Excerpt

                   {excerpt["excerpt"]}

                   ## Summary

                   {excerpt["summary"]} 
               """)
               context += "\n\n"

           return context
   ```

2. Implement garbage collection:
   ```python
   import gc

   def memory_efficient_query(rag, query_text):
       # Force garbage collection before query
       gc.collect()

       result = rag.query(query_text)

       # Force garbage collection after query
       gc.collect()

       return result
   ```

3. Monitor memory usage:
   ```python
   import psutil

   def memory_safe_query(rag, query_text, max_memory_percent=90):
       # Check memory before query
       memory_percent = psutil.virtual_memory().percent
       if memory_percent > max_memory_percent:
           return "System is low on memory. Please try again later."

       # Proceed with query
       return rag.query(query_text)
   ```

---

### **7. API Issues**

**Symptom: OpenAI API errors**

*Potential Causes:*
- Invalid API key
- Rate limiting
- Quota exceeded
- API service disruption

*Solutions:*
1. Verify API key:
   ```python
   import os
   from openai import OpenAI

   def verify_openai_api_key():
       api_key = os.environ.get("OPENAI_API_KEY")
       if not api_key:
           print("API key not found in environment variables")
           return False

       client = OpenAI(api_key=api_key)
       try:
           # Make a simple API call to verify the key
           response = client.embeddings.create(
               model="text-embedding-3-small",
               input="test"
           )
           print("API key is valid")
           return True
       except Exception as e:
           print(f"API key verification failed: {e}")
           return False
   ```

2. Implement rate limiting:
   ```python
   import time

   class RateLimitedOpenAiLlm:
       def __init__(self, base_llm, requests_per_minute=60):
           self.base_llm = base_llm
           self.min_seconds_per_request = 60 / requests_per_minute
           self.last_request_time = 0

       def get_embedding(self, text):
           self._wait_for_rate_limit()
           return self.base_llm.get_embedding(text)

       def get_completion(self, prompt, context=None, use_cache=True):
           self._wait_for_rate_limit()
           return self.base_llm.get_completion(prompt, context, use_cache)

       def _wait_for_rate_limit(self):
           current_time = time.time()
           time_since_last_request = current_time - self.last_request_time

           if time_since_last_request < self.min_seconds_per_request:
               sleep_time = self.min_seconds_per_request - time_since_last_request
               time.sleep(sleep_time)

           self.last_request_time = time.time()
   ```

3. Implement exponential backoff for retries:
   ```python
   import time
   import random

   def retry_with_exponential_backoff(
       func,
       initial_delay=1,
       exponential_base=2,
       jitter=True,
       max_retries=10,
       errors=(Exception,),
   ):
       """Retry a function with exponential backoff."""

       def wrapper(*args, **kwargs):
           # Initialize variables
           num_retries = 0
           delay = initial_delay

           # Loop until a successful response or max_retries is hit
           while True:
               try:
                   return func(*args, **kwargs)

               # Retry on specified errors
               except errors as e:
                   num_retries += 1

                   # Check if max retries has been reached
                   if num_retries > max_retries:
                       raise Exception(f"Maximum number of retries ({max_retries}) exceeded.")

                   # Increment the delay
                   delay *= exponential_base * (1 + jitter * random.random())

                   # Sleep for the delay
                   time.sleep(delay)

                   # Log the retry
                   print(f"Retry {num_retries}/{max_retries} after {delay:.2f} seconds delay")

       return wrapper
   ```

**Symptom: FastAPI errors**

*Potential Causes:*
- Incorrect request format
- Server configuration issues
- Resource constraints

*Solutions:*
1. Validate request format:
   ```python
   from fastapi import FastAPI, HTTPException
   from pydantic import BaseModel, validator

   class QueryRequest(BaseModel):
       text: str
       query_type: str = "standard"

       @validator("text")
       def text_must_not_be_empty(cls, v):
           if not v.strip():
               raise ValueError("Query text cannot be empty")
           return v

       @validator("query_type")
       def query_type_must_be_valid(cls, v):
           valid_types = ["standard", "local_kg", "global_kg", "hybrid_kg", "mix"]
           if v not in valid_types:
               raise ValueError(f"Invalid query_type: {v}. Valid types are: {', '.join(valid_types)}")
           return v
   ```

2. Implement proper error handling:
   ```python
   @app.post("/query")
   async def query_endpoint(request: QueryRequest):
       try:
           # Process query
           query_func = query_map.get(request.query_type.lower())
           result = query_func(request.text)
           return {"result": result}
       except Exception as e:
           # Log the error
           logger.error(f"Error processing query: {e}")
           # Return a user-friendly generic message
           raise HTTPException(status_code=500, detail="Internal server error")
   ```

3. Implement request timeouts:
   ```python
   from fastapi import FastAPI, HTTPException, BackgroundTasks
   import asyncio

   app = FastAPI()

   async def process_query_with_timeout(query_text, query_type, timeout=30):
       try:
           # Run the query with a timeout
           result = await asyncio.wait_for(
               asyncio.to_thread(query_map[query_type], query_text),
               timeout=timeout
           )
           return result
       except asyncio.TimeoutError:
           raise HTTPException(status_code=504, detail="Query processing timed out")

   @app.post("/query")
   async def query_endpoint(request: QueryRequest):
       result = await process_query_with_timeout(request.text, request.query_type)
       return {"result": result}
   ```

---

### **8. Performance Issues**

**Symptom: System becomes slower over time**

*Potential Causes:*
- Growing vector database
- Expanding knowledge graph
- Cache bloat
- Memory leaks

*Solutions:*
1. Implement database pruning:
   ```python
   def prune_old_documents(rag, days_threshold=90):
       """Remove documents older than the threshold."""
       import time

       current_time = time.time()
       threshold_time = current_time - (days_threshold * 24 * 60 * 60)

       # Get all documents
       all_doc_ids = list(rag.doc_to_source_kv.data.keys())

       for doc_id in all_doc_ids:
           # Check if we have timestamp information
           excerpt_ids = rag.doc_to_excerpt_kv.get_by_key(doc_id)
           if not excerpt_ids:
               continue

           # Get the first excerpt to check its timestamp
           excerpt_data = rag.excerpt_kv.get_by_key(excerpt_ids[0])
           if not excerpt_data or "indexed_at" not in excerpt_data:
               continue

           # Check if the document is older than the threshold
           if excerpt_data["indexed_at"] < threshold_time:
               print(f"Removing old document: {doc_id}")
               rag.remove_document_by_id(doc_id)
   ```

2. Implement cache cleanup:
   ```python
   import os
   import time

   def clean_old_cache_files(cache_dir, days_threshold=30):
       """Remove cache files older than the threshold."""
       current_time = time.time()
       threshold_time = current_time - (days_threshold * 24 * 60 * 60)

       for filename in os.listdir(cache_dir):
           file_path = os.path.join(cache_dir, filename)
           if os.path.isfile(file_path):
               file_mtime = os.path.getmtime(file_path)
               if file_mtime < threshold_time:
                   os.remove(file_path)
                   print(f"Removed old cache file: {file_path}")
   ```

3. If index or cache behavior changes, run a full rebuild:
   ```bash
   # Wipe cache/index files and fully rescan app/input_docs
   python rebuild_rag.py
   ```

3. Monitor and optimize memory usage:
   ```python
   import psutil
   import gc

   def optimize_memory():
       """Force garbage collection and report memory usage."""
       # Get initial memory usage
       initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB

       # Force garbage collection
       gc.collect()

       # Get memory usage after garbage collection
       final_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB

       print(f"Memory before: {initial_memory:.2f} MB")
       print(f"Memory after: {final_memory:.2f} MB")
       print(f"Memory freed: {initial_memory - final_memory:.2f} MB")
   ```

**Symptom: High CPU usage**

*Potential Causes:*
- Inefficient vector operations
- Large batch processing
- Excessive parallel processing
- Background tasks

*Solutions:*
1. Profile CPU usage:
   ```python
   import cProfile
   import pstats

   def profile_function(func, *args, **kwargs):
       profiler = cProfile.Profile()
       profiler.enable()

       result = func(*args, **kwargs)

       profiler.disable()
       stats = pstats.Stats(profiler).sort_stats('cumtime')
       stats.print_stats(20)  # Print top 20 functions by cumulative time

       return result

   # Usage
   profile_function(rag.query, "What is SmolRAG?")
   ```

2. Optimize vector operations:
   ```python
   import numpy as np

   # Use vectorized operations instead of loops
   def optimized_similarity(vectors, query_vector):
       # Compute dot products in a single operation
       dot_products = np.dot(vectors, query_vector)

       # Compute norms in a single operation
       vector_norms = np.linalg.norm(vectors, axis=1)
       query_norm = np.linalg.norm(query_vector)

       # Compute similarities
       similarities = dot_products / (vector_norms * query_norm)

       return similarities
   ```

3. Implement batch size control:
   ```python
   def process_with_controlled_batches(items, process_func, batch_size=100):
       """Process items in controlled batch sizes to manage CPU usage."""
       results = []

       for i in range(0, len(items), batch_size):
           batch = items[i:i+batch_size]
           batch_results = process_func(batch)
           results.extend(batch_results)

           # Optional: Add a small delay between batches
           time.sleep(0.1)

       return results
   ```

**Symptom: Disk I/O bottlenecks**

*Potential Causes:*
- Frequent vector store saves
- Large log files
- Inefficient file operations
- Slow storage media

*Solutions:*
1. Reduce save frequency:
   ```python
   class BufferedVectorStore:
       def __init__(self, base_store, buffer_size=100):
           self.base_store = base_store
           self.buffer_size = buffer_size
           self.buffer = []

       def upsert(self, items):
           self.buffer.extend(items)

           if len(self.buffer) >= self.buffer_size:
               self.base_store.upsert(self.buffer)
               self.buffer = []
               self.base_store.save()

       def save(self):
           if self.buffer:
               self.base_store.upsert(self.buffer)
               self.buffer = []
           self.base_store.save()
   ```

2. Implement log rotation:
   ```python
   import logging
   from logging.handlers import RotatingFileHandler

   def setup_rotating_logger(log_path, max_bytes=10485760, backup_count=5):
       """Set up a rotating logger to prevent large log files."""
       logger = logging.getLogger("smolrag")
       handler = RotatingFileHandler(
           log_path,
           maxBytes=max_bytes,
           backupCount=backup_count
       )
       formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
       handler.setFormatter(formatter)
       logger.addHandler(handler)
       return logger
   ```

3. Use memory-mapped files for large datasets:
   ```python
   import numpy as np
   import os

   class MemmapVectorStore:
       def __init__(self, file_path, dimensions, max_vectors=10000):
           self.file_path = file_path
           self.dimensions = dimensions
           self.max_vectors = max_vectors
           self.metadata_file = file_path + ".meta"
           self.initialize()

       def initialize(self):
           if not os.path.exists(self.file_path):
               # Create an empty memory-mapped array
               self.vectors = np.memmap(
                   self.file_path,
                   dtype=np.float32,
                   mode='w+',
                   shape=(self.max_vectors, self.dimensions)
               )
               self.count = 0
               self.save_metadata()
           else:
               # Load metadata
               self.load_metadata()
               # Load existing memory-mapped array
               self.vectors = np.memmap(
                   self.file_path,
                   dtype=np.float32,
                   mode='r+',
                   shape=(self.max_vectors, self.dimensions)
               )

       def save_metadata(self):
           with open(self.metadata_file, 'w') as f:
               f.write(str(self.count))

       def load_metadata(self):
           with open(self.metadata_file, 'r') as f:
               self.count = int(f.read().strip())
   ```

---

### **9. General Troubleshooting Strategies**

When you encounter issues that aren't covered by the specific scenarios above, these general troubleshooting strategies can help:

**1. Check the Logs**

SmolRAG logs important information to the `app/logs/` directory. Examining these logs can provide valuable insights into what's happening:

```bash
# View the main log file
cat app/logs/main.log

# Search for errors
grep "ERROR" app/logs/main.log

# Follow the log in real-time
tail -f app/logs/main.log
```

**2. Enable Debug Logging**

For more detailed information, you can enable debug logging:

```python
# In app/logger.py
def set_logger(log_file_name):
    # ...
    logger.setLevel(logging.DEBUG)  # Change from INFO to DEBUG
    # ...
```

**3. Inspect Data Files**

Examining the data files can help identify issues:

```bash
# List data files
ls -la app/data/

# Check file sizes
du -h app/data/

# Check if vector store files exist
ls -la app/data/embeddings.db
```

**4. Test Components in Isolation**

When troubleshooting complex issues, it can be helpful to test components in isolation:

```python
# Test document loading
from app.utilities import read_file
content = read_file("app/input_docs/sample.md")
print(f"Loaded document with {len(content)} characters")

# Test chunking
from app.chunking import preserve_markdown_code_excerpts
chunks = preserve_markdown_code_excerpts(content, 2000, 200)
print(f"Created {len(chunks)} chunks")

# Test embedding generation
from app.openai_llm import OpenAiLlm
llm = OpenAiLlm()
embedding = llm.get_embedding("Test text")
print(f"Generated embedding with {len(embedding)} dimensions")
```

**5. Create a Minimal Reproduction**

Creating a minimal reproduction of the issue can help isolate the problem:

```python
# Minimal SmolRAG setup
from app.smol_rag import SmolRag
import os
from app.definitions import DATA_DIR, CACHE_DIR, LOG_DIR

# Clean start
for dir_path in [DATA_DIR, CACHE_DIR, LOG_DIR]:
    os.makedirs(dir_path, exist_ok=True)
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

# Initialize with minimal configuration
rag = SmolRag(excerpt_size=1000, overlap=100)

# Test basic functionality
rag.import_documents()
result = rag.query("Test query")
print(result)
```

**6. Check System Resources**

System resource constraints can cause various issues:

```python
import psutil

def check_system_resources():
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"CPU Usage: {cpu_percent}%")

    # Memory usage
    memory = psutil.virtual_memory()
    print(f"Memory Usage: {memory.percent}%")
    print(f"Available Memory: {memory.available / (1024 * 1024):.2f} MB")

    # Disk usage
    disk = psutil.disk_usage('/')
    print(f"Disk Usage: {disk.percent}%")
    print(f"Free Disk Space: {disk.free / (1024 * 1024 * 1024):.2f} GB")

# Check resources
check_system_resources()
```

---

### **10. Conclusion**

This troubleshooting guide covers common issues you might encounter when working with SmolRAG. By following the diagnostic steps and implementing the suggested solutions, you should be able to resolve most problems and ensure smooth operation of your retrieval-augmented generation system.

Remember that troubleshooting is often an iterative process. Start with the simplest possible explanation and solution, and gradually work your way toward more complex possibilities. Keep detailed notes about what you've tried and the results, as this can help identify patterns and root causes.

If you encounter issues not covered in this guide, consider:

1. Checking the project's issue tracker for similar problems and solutions
2. Consulting the community forums or discussion groups
3. Reviewing the source code to understand the underlying behavior
4. Reaching out to the project maintainers with a detailed description of the issue and steps to reproduce it

With persistence and systematic troubleshooting, most issues can be resolved, allowing you to take full advantage of SmolRAG's capabilities for retrieval-augmented generation.
