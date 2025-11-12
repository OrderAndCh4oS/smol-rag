#!/usr/bin/env python3
"""Quick test to check if queries work during import."""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.smol_rag import SmolRag


async def test_query():
    """Test a quick query."""
    rag = SmolRag()

    try:
        print("🔍 Testing query: 'What is SmolRag?'")
        result = await rag.query("What is SmolRag?")
        print(f"✅ Query successful!")
        print(f"Response preview: {result[:300]}...")
        return True
    except Exception as e:
        print(f"❌ Query failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_query())
    sys.exit(0 if success else 1)
