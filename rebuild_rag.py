#!/usr/bin/env python3
"""
Script to rebuild the RAG from sample documents and verify it works.
"""
import asyncio
import argparse
import os
import sys

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.smol_rag import SmolRag
from app.logger import logger
from app.definitions import SOURCE_TO_DOC_ID_KV_PATH, DOC_ID_TO_SOURCE_KV_PATH, DOC_ID_TO_EXCERPT_KV_PATH, \
    DOC_ID_TO_ENTITY_IDS_KV_PATH, DOC_ID_TO_RELATIONSHIP_IDS_KV_PATH, ENTITY_ID_TO_DOC_IDS_KV_PATH, \
    RELATIONSHIP_ID_TO_DOC_IDS_KV_PATH, EXCERPT_KV_PATH, EMBEDDINGS_DB, ENTITIES_DB, RELATIONSHIPS_DB, KG_DB, \
    QUERY_CACHE_KV_PATH, EMBEDDING_CACHE_KV_PATH


STATE_FILES = [
    SOURCE_TO_DOC_ID_KV_PATH,
    DOC_ID_TO_SOURCE_KV_PATH,
    DOC_ID_TO_EXCERPT_KV_PATH,
    DOC_ID_TO_ENTITY_IDS_KV_PATH,
    DOC_ID_TO_RELATIONSHIP_IDS_KV_PATH,
    ENTITY_ID_TO_DOC_IDS_KV_PATH,
    RELATIONSHIP_ID_TO_DOC_IDS_KV_PATH,
    EXCERPT_KV_PATH,
    EMBEDDINGS_DB,
    ENTITIES_DB,
    RELATIONSHIPS_DB,
    KG_DB,
    QUERY_CACHE_KV_PATH,
    EMBEDDING_CACHE_KV_PATH,
]


def wipe_state():
    removed = 0
    for path in STATE_FILES:
        if os.path.exists(path):
            os.remove(path)
            removed += 1
            logger.info("Removed state file: %s", path)
    logger.info("State wipe complete. Removed %s files.", removed)


async def main(wipe=True):
    """Rebuild the RAG and test it."""
    logger.info("=" * 80)
    logger.info("REBUILDING RAG FROM SAMPLE DOCUMENTS")
    logger.info("=" * 80)

    if wipe:
        logger.info("Wiping existing cache/index state before rebuild...")
        wipe_state()

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

    failures = []
    for query in test_queries:
        logger.info(f"\n📝 Query: {query}")
        try:
            result = await rag.query(query)
            logger.info(f"✅ Response: {result[:200]}..." if len(result) > 200 else f"✅ Response: {result}")
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            failures.append((query, str(e)))

    if failures:
        logger.error("\nVerification failed for %s queries.", len(failures))
        for query, error in failures:
            logger.error("Failed query '%s': %s", query, error)
        return 1

    logger.info("\n" + "=" * 80)
    logger.info("RAG REBUILD AND TEST COMPLETED!")
    logger.info("=" * 80)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wipe local SmolRAG state and rebuild by rescanning input docs.")
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not delete current cache/index files before import."
    )
    args = parser.parse_args()
    exit_code = asyncio.run(main(wipe=not args.keep_existing))
    sys.exit(exit_code)
