"""
Tests for GraphStore concurrency safety.
Tests the concurrency fixes: async methods with lock protection.
"""
import asyncio
import time
import pytest

from app.graph_store import NetworkXGraphStore


class TestGraphStoreConcurrency:
    """Tests for concurrent access safety."""

    @pytest.mark.asyncio
    async def test_async_add_node_works(self, graph_store):
        """Test that async_add_node works correctly."""
        await graph_store.async_add_node(
            name="AsyncTestEntity",
            node_type="Test",
            description="Async test entity",
            source_id="test_doc"
        )

        node = graph_store.get_node("AsyncTestEntity")
        assert node is not None
        assert node.get("node_type") == "Test"

    @pytest.mark.asyncio
    async def test_async_add_edge_works(self, graph_store):
        """Test that async_add_edge works correctly."""
        # Add nodes first
        await graph_store.async_add_node("AsyncSource", node_type="Type", description="Source", source_id="doc1")
        await graph_store.async_add_node("AsyncTarget", node_type="Type", description="Target", source_id="doc1")

        # Add edge
        await graph_store.async_add_edge(
            source="AsyncSource",
            destination="AsyncTarget",
            description="Async relationship",
            keywords="test",
            weight=1.0
        )

        edge = graph_store.get_edge(("AsyncSource", "AsyncTarget"))
        assert edge is not None

    @pytest.mark.asyncio
    async def test_async_set_field_works(self, graph_store):
        """Test that async_set_field works correctly."""
        await graph_store.async_set_field("test_key", "test_value")

        # Access graph metadata directly
        assert graph_store.graph.graph.get("test_key") == "test_value"

    @pytest.mark.asyncio
    async def test_concurrent_node_additions_no_race_condition(self, graph_store):
        """Test that concurrent node additions don't cause race conditions."""
        # Create 50 concurrent tasks adding nodes
        async def add_node(i):
            await graph_store.async_add_node(
                name=f"ConcurrentNode_{i}",
                node_type="Test",
                description=f"Node {i}",
                source_id="test_doc"
            )

        tasks = [add_node(i) for i in range(50)]
        await asyncio.gather(*tasks)

        # Verify all nodes were added
        for i in range(50):
            node = graph_store.get_node(f"ConcurrentNode_{i}")
            assert node is not None, f"Node {i} was not added correctly"

    @pytest.mark.asyncio
    async def test_concurrent_edge_additions_no_race_condition(self, graph_store):
        """Test that concurrent edge additions don't cause race conditions."""
        # Setup: Add nodes first
        for i in range(20):
            await graph_store.async_add_node(
                f"EdgeNode_{i}",
                node_type="Test",
                description=f"Node {i}",
                source_id="test_doc"
            )

        # Create concurrent tasks adding edges
        async def add_edge(i):
            source = f"EdgeNode_{i}"
            target = f"EdgeNode_{(i+1) % 20}"
            await graph_store.async_add_edge(
                source=source,
                destination=target,
                description=f"Edge {i}",
                keywords="test",
                weight=1.0
            )

        tasks = [add_edge(i) for i in range(20)]
        await asyncio.gather(*tasks)

        # Verify all edges were added
        for i in range(20):
            source = f"EdgeNode_{i}"
            target = f"EdgeNode_{(i+1) % 20}"
            edge = graph_store.get_edge((source, target))
            assert edge is not None, f"Edge {i} was not added correctly"

    @pytest.mark.asyncio
    async def test_concurrent_reads_and_writes_safe(self, graph_store):
        """Test that concurrent reads and writes don't cause corruption."""
        # Setup: Add initial nodes
        for i in range(10):
            await graph_store.async_add_node(
                f"MixedNode_{i}",
                node_type="Test",
                description=f"Initial {i}",
                source_id="test_doc"
            )

        # Mix of concurrent reads and writes
        async def read_node(i):
            node = graph_store.get_node(f"MixedNode_{i % 10}")
            return node is not None

        async def write_node(i):
            await graph_store.async_add_node(
                f"NewNode_{i}",
                node_type="Test",
                description=f"New {i}",
                source_id="test_doc"
            )

        # Create mix of read and write tasks
        tasks = []
        for i in range(30):
            if i % 2 == 0:
                tasks.append(read_node(i))
            else:
                tasks.append(write_node(i))

        results = await asyncio.gather(*tasks)

        # Verify all reads succeeded
        read_results = [r for r in results if r is not None]
        assert all(read_results), "Some reads failed"

        # Verify all writes succeeded
        for i in range(1, 30, 2):
            node = graph_store.get_node(f"NewNode_{i}")
            assert node is not None, f"NewNode_{i} was not added"

    @pytest.mark.asyncio
    async def test_async_save_does_not_block_event_loop(self, temp_graph_path):
        """Test that async_save runs in executor and doesn't block."""
        store = NetworkXGraphStore(temp_graph_path)

        # Add some data
        for i in range(100):
            await store.async_add_node(
                f"SaveTestNode_{i}",
                node_type="Test",
                description=f"Node {i}",
                source_id="test_doc"
            )

        # Create a concurrent task that should complete quickly
        quick_task_completed = False

        async def quick_task():
            nonlocal quick_task_completed
            await asyncio.sleep(0.01)  # 10ms
            quick_task_completed = True

        # Start save and quick task concurrently
        start_time = time.perf_counter()
        task = asyncio.create_task(quick_task())

        # async_save should not block the event loop
        await store.async_save()

        await task
        elapsed = time.perf_counter() - start_time

        # Quick task should have completed despite the save
        assert quick_task_completed, "Quick task was blocked by async_save"

        print(f"\nAsync save + concurrent task: {elapsed:.4f}s")
        print(f"Quick task completed: {quick_task_completed}")

        # Verify data was saved
        store2 = NetworkXGraphStore(temp_graph_path)
        node = store2.get_node("SaveTestNode_0")
        assert node is not None, "Data was not saved correctly"

    @pytest.mark.asyncio
    async def test_lock_prevents_race_condition_on_same_node(self, graph_store):
        """Test that lock prevents race conditions when updating same node."""
        # Add initial node
        await graph_store.async_add_node(
            "SharedNode",
            node_type="Test",
            description="Initial",
            source_id="doc1"
        )

        # Attempt to update same node concurrently
        # Without locks, this could cause corruption
        async def update_node(value):
            await graph_store.async_add_node(
                "SharedNode",
                node_type="Test",
                description=f"Update_{value}",
                source_id=f"doc_{value}"
            )

        # Run 20 concurrent updates
        tasks = [update_node(i) for i in range(20)]
        await asyncio.gather(*tasks)

        # Node should exist and have a valid state (not corrupted)
        node = graph_store.get_node("SharedNode")
        assert node is not None, "Node was corrupted"
        assert "description" in node, "Node attributes were corrupted"

    @pytest.mark.asyncio
    async def test_atomic_entity_upsert_keeps_all_excerpt_ids(self, graph_store):
        """Concurrent upserts to one entity should not drop contributions."""
        async def upsert(i):
            await graph_store.async_upsert_entity_node(
                name="SharedEntity",
                category="Type",
                description=f"Description {i}",
                excerpt_id=f"excerpt-{i}",
                sep=":|:",
            )

        await asyncio.gather(*(upsert(i) for i in range(20)))

        node = graph_store.get_node("SharedEntity")
        assert node is not None
        excerpt_ids = set(node["excerpt_id"].split(":|:"))
        assert len(excerpt_ids) == 20

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_operations_performance(self, graph_store):
        """Measure performance of concurrent operations."""
        num_operations = 100

        # Measure async operations with locks
        start_time = time.perf_counter()

        tasks = []
        for i in range(num_operations):
            tasks.append(graph_store.async_add_node(
                f"PerfNode_{i}",
                node_type="Test",
                description=f"Node {i}",
                source_id="test_doc"
            ))

        await asyncio.gather(*tasks)
        async_elapsed = time.perf_counter() - start_time

        print(f"\n{num_operations} async node additions: {async_elapsed:.4f}s")
        print(f"Average per operation: {async_elapsed/num_operations:.6f}s")

        # Verify all nodes were added
        for i in range(num_operations):
            node = graph_store.get_node(f"PerfNode_{i}")
            assert node is not None

    @pytest.mark.asyncio
    async def test_mixed_sync_and_async_methods(self, graph_store):
        """Test that mixing sync and async methods works (for backward compatibility)."""
        # Sync method
        graph_store.add_node("SyncNode", node_type="Test", description="Sync", source_id="doc1")

        # Async method
        await graph_store.async_add_node("AsyncNode", node_type="Test", description="Async", source_id="doc1")

        # Both should exist
        assert graph_store.get_node("SyncNode") is not None
        assert graph_store.get_node("AsyncNode") is not None

        # Add edges
        graph_store.add_edge("SyncNode", "AsyncNode", description="Sync edge", keywords="test", weight=1.0)
        await graph_store.async_add_edge("AsyncNode", "SyncNode", description="Async edge", keywords="test", weight=1.0)

        # Both edges should exist
        assert graph_store.get_edge(("SyncNode", "AsyncNode")) is not None
        assert graph_store.get_edge(("AsyncNode", "SyncNode")) is not None


class TestGraphStoreStressTest:
    """Stress tests for concurrent access under heavy load."""

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_high_concurrency_stress(self, graph_store):
        """Stress test with high number of concurrent operations."""
        num_tasks = 200

        async def mixed_operation(i):
            # Mix of different operations
            if i % 3 == 0:
                await graph_store.async_add_node(
                    f"StressNode_{i}",
                    node_type="Stress",
                    description=f"Node {i}",
                    source_id="stress_test"
                )
            elif i % 3 == 1:
                # Read operation
                graph_store.get_node(f"StressNode_{i-1}")
            else:
                await graph_store.async_set_field(f"stress_key_{i}", f"value_{i}")

        start_time = time.perf_counter()
        tasks = [mixed_operation(i) for i in range(num_tasks)]
        await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start_time

        print(f"\nStress test: {num_tasks} concurrent operations in {elapsed:.4f}s")
        print(f"Operations per second: {num_tasks/elapsed:.2f}")

        # Verify some operations succeeded
        node_count = sum(1 for i in range(0, num_tasks, 3)
                        if graph_store.get_node(f"StressNode_{i}") is not None)
        assert node_count > 0, "No nodes were added during stress test"
        print(f"Successfully added {node_count} nodes")
