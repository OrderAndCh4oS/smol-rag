#!/usr/bin/env python3
"""
Script to rebuild the RAG from sample documents and verify it works.
"""
import asyncio
import os
import sys

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.smol_rag import SmolRag
from app.logger import logger


async def main():
    """Rebuild the RAG and test it."""
    logger.info("=" * 80)
    logger.info("REBUILDING RAG FROM SAMPLE DOCUMENTS")
    logger.info("=" * 80)

    # Initialize RAG with default settings
    rag = SmolRag()

    # Import all documents
    logger.info("\n🔄 Starting document import...")
    await rag.import_documents()
    logger.info("✅ Document import completed!")

    # Test query functionality
    logger.info("\n" + "=" * 80)
    logger.info("TESTING QUERY FUNCTIONALITY")
    logger.info("=" * 80)

    test_queries = [
        "What is SmolRag?",
        "How does document ingestion work?",
        "What are the different query types?",
    ]

    for query in test_queries:
        logger.info(f"\n📝 Query: {query}")
        try:
            result = await rag.query(query)
            logger.info(f"✅ Response: {result[:200]}..." if len(result) > 200 else f"✅ Response: {result}")
        except Exception as e:
            logger.error(f"❌ Error: {e}")

    logger.info("\n" + "=" * 80)
    logger.info("RAG REBUILD AND TEST COMPLETED!")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
