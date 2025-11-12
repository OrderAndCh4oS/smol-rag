#!/usr/bin/env python3
"""
Test concurrent request handling to verify no race conditions.
"""
import asyncio
import aiohttp
import time
import sys


async def make_query(session, query_num, query_type="standard"):
    """Make a query to the API."""
    url = "http://localhost:8000/query"

    # Vary queries to test different code paths
    queries = [
        "What is SmolRag?",
        "How does document ingestion work?",
        "What are the query types?",
        "How does embedding batching work?",
        "What are the performance optimizations?",
    ]

    payload = {
        "text": queries[query_num % len(queries)],
        "query_type": query_type
    }

    start = time.time()
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as response:
            result = await response.json()
            elapsed = time.time() - start
            return query_num, elapsed, response.status, None
    except Exception as e:
        elapsed = time.time() - start
        return query_num, elapsed, None, str(e)


async def test_concurrent_queries(num_requests=10, query_type="standard"):
    """Send multiple concurrent requests."""
    print(f"🔥 Sending {num_requests} concurrent {query_type} requests...\n")

    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        tasks = [make_query(session, i, query_type) for i in range(num_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    total_elapsed = time.time() - start_time

    print("📊 Results:")
    success_count = 0
    error_count = 0
    query_times = []

    for result in results:
        if isinstance(result, Exception):
            print(f"❌ Request failed with exception: {result}")
            error_count += 1
        else:
            num, elapsed, status, error = result
            if error:
                print(f"❌ Request {num}: Error - {error}")
                error_count += 1
            elif status == 200:
                success_count += 1
                query_times.append(elapsed)
                print(f"✅ Request {num}: {status} - {elapsed:.2f}s")
            else:
                print(f"⚠️  Request {num}: {status} - {elapsed:.2f}s")
                error_count += 1

    print(f"\n📈 Summary:")
    print(f"  Total time: {total_elapsed:.2f}s")
    print(f"  Success: {success_count}/{num_requests}")
    print(f"  Errors: {error_count}/{num_requests}")

    if query_times:
        print(f"  Avg query time: {sum(query_times)/len(query_times):.2f}s")
        print(f"  Min query time: {min(query_times):.2f}s")
        print(f"  Max query time: {max(query_times):.2f}s")

    if success_count == num_requests:
        print(f"\n✅ All requests completed successfully without errors!")
        return True
    else:
        print(f"\n❌ {error_count} requests failed - possible race conditions or errors")
        return False


async def test_mixed_query_types(num_each=3):
    """Test concurrent requests with different query types."""
    print("🎯 Testing mixed query types concurrently...\n")

    query_types = ["standard", "hybrid_kg", "local_kg", "global_kg", "mix"]

    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, qtype in enumerate(query_types):
            for j in range(num_each):
                tasks.append(make_query(session, i * num_each + j, qtype))

        results = await asyncio.gather(*tasks, return_exceptions=True)

    success = sum(1 for r in results if not isinstance(r, Exception) and r[2] == 200)
    total = len(results)

    print(f"📊 Mixed query results: {success}/{total} successful\n")
    return success == total


async def test_rapid_fire(duration=5):
    """Send queries as fast as possible for a duration."""
    print(f"⚡ Rapid fire test for {duration} seconds...\n")

    end_time = time.time() + duration
    request_count = 0
    success_count = 0

    async with aiohttp.ClientSession() as session:
        while time.time() < end_time:
            try:
                async with session.post(
                    "http://localhost:8000/query",
                    json={"text": "What is SmolRag?", "query_type": "standard"},
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        success_count += 1
                    request_count += 1
            except Exception as e:
                request_count += 1

    print(f"📊 Rapid fire results:")
    print(f"  Requests sent: {request_count}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {request_count - success_count}")
    print(f"  Rate: {request_count/duration:.1f} req/sec\n")

    return success_count / request_count > 0.9 if request_count > 0 else False


async def main():
    """Run all concurrency tests."""
    print("=" * 80)
    print("CONCURRENT REQUEST SAFETY TESTS")
    print("=" * 80)
    print()

    # Check if server is running
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:8000/query",
                json={"text": "test", "query_type": "standard"},
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status not in [200, 500]:
                    print("❌ Server not responding correctly")
                    return False
    except Exception as e:
        print(f"❌ Could not connect to server: {e}")
        print("\nMake sure the server is running:")
        print("  uvicorn api.main:app --host 0.0.0.0 --port 8000\n")
        return False

    print("✅ Server is running\n")
    print("=" * 80)
    print()

    all_passed = True

    # Test 1: Basic concurrent queries
    print("Test 1: Basic Concurrent Queries")
    print("-" * 80)
    passed = await test_concurrent_queries(10, "standard")
    all_passed = all_passed and passed
    print()

    # Test 2: Mixed query types
    print("Test 2: Mixed Query Types")
    print("-" * 80)
    passed = await test_mixed_query_types(3)
    all_passed = all_passed and passed
    print()

    # Test 3: Rapid fire (commented out to avoid API rate limits)
    # print("Test 3: Rapid Fire")
    # print("-" * 80)
    # passed = await test_rapid_fire(5)
    # all_passed = all_passed and passed
    # print()

    print("=" * 80)
    if all_passed:
        print("✅ ALL TESTS PASSED - No race conditions detected!")
    else:
        print("❌ SOME TESTS FAILED - Check for race conditions")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(1)
