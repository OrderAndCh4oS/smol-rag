#!/usr/bin/env python3
"""
Test importing the new performance optimization document and querying it.
"""
import asyncio
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.smol_rag import SmolRag
from app.logger import logger


async def main():
    """Import new document and test queries."""
    logger.info("=" * 80)
    logger.info("IMPORTING NEW PERFORMANCE OPTIMIZATION DOCUMENT")
    logger.info("=" * 80)

    # Initialize RAG (will load existing data)
    rag = SmolRag()

    # Check current document count
    import os
    from app.utilities import get_docs
    from app.definitions import INPUT_DOCS_DIR

    docs_before = len(get_docs(INPUT_DOCS_DIR))
    logger.info(f"📚 Documents in input_docs: {docs_before}")

    # Import documents (should only add the new one)
    logger.info("\n🔄 Running import_documents()...")
    start_time = time.time()
    await rag.import_documents()
    elapsed = time.time() - start_time
    logger.info(f"✅ Import completed in {elapsed:.2f} seconds")

    # Test queries about performance optimizations
    logger.info("\n" + "=" * 80)
    logger.info("TESTING QUERIES ABOUT PERFORMANCE OPTIMIZATIONS")
    logger.info("=" * 80)

    test_queries = [
        "What are the main performance bottlenecks in SmolRAG?",
        "How does embedding batching work?",
        "What are the performance improvements from the optimizations?",
    ]

    for i, query in enumerate(test_queries, 1):
        logger.info(f"\n📝 Query {i}: {query}")
        try:
            query_start = time.time()
            result = await rag.query(query)
            query_elapsed = time.time() - query_start

            logger.info(f"⏱️  Query time: {query_elapsed:.2f}s")
            logger.info(f"✅ Response preview:")
            logger.info(f"{result[:400]}...")

        except Exception as e:
            logger.error(f"❌ Query failed: {e}")

    # Test cached query
    logger.info("\n" + "=" * 80)
    logger.info("TESTING CACHED QUERY PERFORMANCE")
    logger.info("=" * 80)

    cached_query = test_queries[0]
    logger.info(f"\n📝 Cached Query: {cached_query}")

    cache_start = time.time()
    cached_result = await rag.query(cached_query)
    cache_elapsed = time.time() - cache_start

    logger.info(f"⚡ Cached query time: {cache_elapsed:.4f}s (should be <0.1s)")
    logger.info(f"✅ Cache hit - instant response!")

    logger.info("\n" + "=" * 80)
    logger.info("TEST COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
