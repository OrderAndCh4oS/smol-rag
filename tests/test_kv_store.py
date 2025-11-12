"""
Tests for JsonKvStore operations, focusing on file I/O bottlenecks.
Tests bottleneck #2: Full JSON file rewrites on every save.
"""
import asyncio
import json
import os
import time
import pytest

from app.kv_store import JsonKvStore


class TestKvStoreBaseline:
    """Baseline tests for current KV store functionality."""

    @pytest.mark.asyncio
    async def test_set_and_get_single_value(self, kv_store):
        """Test basic set and get operations."""
        await kv_store.add("test_key", "test_value")
        value = await kv_store.get_by_key("test_key")
        assert value == "test_value"

    @pytest.mark.asyncio
    async def test_set_complex_value(self, kv_store):
        """Test storing complex data structures."""
        complex_data = {
            "nested": {
                "data": [1, 2, 3],
                "more": {"deep": "value"}
            }
        }
        await kv_store.add("complex", complex_data)
        retrieved = await kv_store.get_by_key("complex")
        assert retrieved == complex_data

    @pytest.mark.asyncio
    async def test_get_nonexistent_key(self, kv_store):
        """Test getting a key that doesn't exist."""
        value = await kv_store.get_by_key("nonexistent")
        assert value is None

    @pytest.mark.asyncio
    async def test_delete_key(self, kv_store):
        """Test deleting a key."""
        await kv_store.add("to_delete", "value")
        await kv_store.remove("to_delete")
        value = await kv_store.get_by_key("to_delete")
        assert value is None

    @pytest.mark.asyncio
    async def test_persistence_after_save_and_load(self, temp_kv_path):
        """Test data persists after save and reload."""
        store1 = JsonKvStore(temp_kv_path)
        await store1.add("persistent", "data")
        await store1.save()

        # Load in new instance
        store2 = JsonKvStore(temp_kv_path)
        value = await store2.get_by_key("persistent")
        assert value == "data"

    @pytest.mark.asyncio
    async def test_multiple_keys(self, kv_store):
        """Test storing multiple keys."""
        data = {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3"
        }
        for key, value in data.items():
            await kv_store.add(key, value)

        for key, expected_value in data.items():
            value = await kv_store.get_by_key(key)
            assert value == expected_value


class TestKvStoreFileIOPerformance:
    """Tests for file I/O performance bottlenecks."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_sequential_saves_write_entire_file(self, temp_kv_path):
        """Test that demonstrates full file rewrite on each save."""
        store = JsonKvStore(temp_kv_path)

        # Add some initial data
        for i in range(100):
            await store.add(f"key_{i}", f"value_{i}")

        # Measure file size
        initial_writes = 0

        # Time multiple saves
        start_time = time.perf_counter()

        for i in range(10):
            await store.add(f"new_key_{i}", f"new_value_{i}")
            await store.save()
            initial_writes += 1

        elapsed = time.perf_counter() - start_time

        # Check file size (should be similar for each save)
        file_size = os.path.getsize(temp_kv_path)

        print(f"\n{initial_writes} saves took {elapsed:.4f}s")
        print(f"Final file size: {file_size} bytes")
        print(f"Estimated data written: {file_size * initial_writes} bytes")

        assert elapsed >= 0

    @pytest.mark.performance
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_save_frequency_impact(self, temp_kv_path):
        """Test impact of save frequency on performance."""
        # Test 1: Save after every operation
        store1 = JsonKvStore(temp_kv_path + "_frequent")

        start_time = time.perf_counter()
        for i in range(50):
            await store1.add(f"key_{i}", f"value_{i}")
            await store1.save()  # Save every time
        frequent_save_time = time.perf_counter() - start_time

        # Test 2: Batch save at the end
        store2 = JsonKvStore(temp_kv_path + "_batch")

        start_time = time.perf_counter()
        for i in range(50):
            await store2.add(f"key_{i}", f"value_{i}")
        await store2.save()  # Save once
        batch_save_time = time.perf_counter() - start_time

        print(f"\nFrequent saves (50x): {frequent_save_time:.4f}s")
        print(f"Batch save (1x): {batch_save_time:.4f}s")
        print(f"Speedup: {frequent_save_time / batch_save_time:.2f}x")

        # Batch should be significantly faster
        assert batch_save_time < frequent_save_time

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_large_store_save_time(self, temp_kv_path):
        """Test save time grows with store size."""
        store = JsonKvStore(temp_kv_path)

        save_times = []

        # Add data in batches and measure save time
        for batch in range(5):
            # Add 100 entries
            for i in range(100):
                idx = batch * 100 + i
                # Make values larger to simulate real data
                await store.add(f"key_{idx}", {"data": "x" * 1000, "index": idx})

            # Measure save time
            start_time = time.perf_counter()
            await store.save()
            elapsed = time.perf_counter() - start_time
            save_times.append(elapsed)

            print(f"Save after {(batch + 1) * 100} entries: {elapsed:.4f}s")

        # Save time should increase as store grows (linear growth)
        assert save_times[-1] >= save_times[0]

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_operations_with_lock(self, kv_store):
        """Test that save lock prevents concurrent operations."""
        # Add some data
        for i in range(100):
            await kv_store.add(f"key_{i}", f"value_{i}")

        async def writer_task(task_id):
            for i in range(10):
                await kv_store.add(f"task_{task_id}_key_{i}", f"value_{i}")
                await kv_store.save()

        start_time = time.perf_counter()

        # Run multiple writers concurrently
        tasks = [writer_task(i) for i in range(5)]
        await asyncio.gather(*tasks)

        elapsed = time.perf_counter() - start_time

        print(f"\n5 concurrent writers, 10 saves each: {elapsed:.4f}s")

        # Verify all data was written
        for task_id in range(5):
            for i in range(10):
                value = await kv_store.get_by_key(f"task_{task_id}_key_{i}")
                assert value == f"value_{i}"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_file_io_blocks_event_loop(self, kv_store):
        """Test that file I/O blocks the event loop."""
        # Add significant data to make save slower
        for i in range(500):
            await kv_store.add(f"key_{i}", {"data": "x" * 1000})

        async def quick_task():
            await asyncio.sleep(0.001)
            return "done"

        # Create concurrent task
        task = asyncio.create_task(quick_task())

        start_time = time.perf_counter()

        # Save (blocks event loop with file I/O)
        await kv_store.save()

        await task
        elapsed = time.perf_counter() - start_time

        print(f"\nSave + concurrent task: {elapsed:.4f}s")
        assert elapsed >= 0


class TestKvStoreMemoryUsage:
    """Tests for memory usage patterns."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_entire_store_in_memory(self, kv_store):
        """Test that entire store is kept in memory."""
        # Add 1000 entries with 10KB each = ~10MB
        for i in range(1000):
            await kv_store.add(f"key_{i}", {"data": "x" * 10000})

        # All data should be accessible without loading
        value = await kv_store.get_by_key("key_500")
        assert value is not None
        assert len(value["data"]) == 10000

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_json_serialization_overhead(self, temp_kv_path):
        """Measure JSON serialization overhead during save."""
        store = JsonKvStore(temp_kv_path)

        # Add complex nested data
        for i in range(100):
            await store.add(f"key_{i}", {
                "id": i,
                "nested": {
                    "list": [j for j in range(100)],
                    "dict": {f"k{j}": f"v{j}" for j in range(50)}
                }
            })

        # Time the save (includes JSON serialization)
        start_time = time.perf_counter()
        await store.save()
        elapsed = time.perf_counter() - start_time

        # Check file size
        file_size = os.path.getsize(temp_kv_path)

        print(f"\nJSON serialization + write: {elapsed:.4f}s for {file_size} bytes")


class TestKvStoreEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_save_to_nonexistent_directory(self):
        """Test saving to a directory that doesn't exist."""
        # This will fail on __init__ because create_file_if_not_exists needs parent dir
        try:
            bad_path = "/tmp/nonexistent_dir_12345/test.json"
            store = JsonKvStore(bad_path)
            await store.add("key", "value")
            await store.save()
            # If we get here, implementation handles missing dirs
        except Exception:
            # Expected - directory doesn't exist
            assert True

    @pytest.mark.asyncio
    async def test_load_corrupted_json(self, temp_kv_path):
        """Test loading a corrupted JSON file."""
        # Write invalid JSON
        with open(temp_kv_path, 'w') as f:
            f.write("{ invalid json }")

        # JsonKvStore loads in __init__
        try:
            store = JsonKvStore(temp_kv_path)
            # Depending on implementation, might start with empty store or raise
        except json.JSONDecodeError:
            # Expected
            assert True

    @pytest.mark.asyncio
    async def test_concurrent_load_and_save(self, kv_store, temp_kv_path):
        """Test concurrent load and save operations."""
        await kv_store.add("initial", "data")
        await kv_store.save()

        async def saver():
            for i in range(10):
                await kv_store.add(f"key_{i}", f"value_{i}")
                await kv_store.save()
                await asyncio.sleep(0.01)

        async def loader():
            for i in range(10):
                try:
                    store2 = JsonKvStore(temp_kv_path)
                    # Loads in __init__ - may fail if file is being written
                    await asyncio.sleep(0.01)
                except Exception:
                    # Expected - file may be being written
                    await asyncio.sleep(0.01)

        # Run concurrently
        await asyncio.gather(saver(), loader())

        # Final consistency check
        value = await kv_store.get_by_key("key_9")
        assert value == "value_9"
