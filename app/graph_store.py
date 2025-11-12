import asyncio
import os

import networkx as nx

from app.logger import logger


class NetworkXGraphStore:
    def __init__(self, file_path):
        self.file_path = file_path
        self._lock = asyncio.Lock()  # Protect concurrent access to graph
        if os.path.exists(file_path):
            try:
                self.graph = nx.read_graphml(file_path)
                logger.info(f"Knowledge graph loaded from {file_path}")
            except Exception as e:
                logger.error(f"Error loading knowledge graph from {file_path}: {e}")
                self.graph = nx.Graph()
        else:
            self.graph = nx.Graph()
            logger.info("No existing knowledge graph found; creating a new one.")

    def get_node(self, name):
        logger.info(f"Getting node {name}")
        # Note: NetworkX dict access is atomic for reads, no lock needed for single dict lookups
        return self.graph.nodes.get(name)

    def get_edge(self, edge):
        logger.info(f"Getting edge {edge}")
        return self.graph.edges.get(edge)

    def get_node_edges(self, name):
        # Returning iterator - safe for concurrent reads
        return self.graph.edges(name)

    def add_node(self, name, **kwargs):
        # Synchronous wrapper for use in sync contexts
        logger.info(f"Adding node {name}")
        self.graph.add_node(name, **kwargs)

    def add_edge(self, source, destination, **kwargs):
        # Synchronous wrapper for use in sync contexts
        logger.info(f"Adding edge {(source, destination)}")
        self.graph.add_edge(source, destination, **kwargs)

    def degree(self, name):
        return self.graph.degree(name)

    def set_field(self, key, value):
        # Synchronous wrapper for metadata updates
        self.graph.graph[key] = value
        logger.info(f"Graph metadata '{key}' updated to: {value}")

    async def async_add_node(self, name, **kwargs):
        """Async version with lock protection for concurrent writes."""
        async with self._lock:
            logger.info(f"Adding node {name}")
            self.graph.add_node(name, **kwargs)

    async def async_add_edge(self, source, destination, **kwargs):
        """Async version with lock protection for concurrent writes."""
        async with self._lock:
            logger.info(f"Adding edge {(source, destination)}")
            self.graph.add_edge(source, destination, **kwargs)

    async def async_set_field(self, key, value):
        """Async version with lock protection for concurrent writes."""
        async with self._lock:
            self.graph.graph[key] = value
            logger.info(f"Graph metadata '{key}' updated to: {value}")

    def save(self):
        # Note: This is synchronous and blocks. For production, should be async
        nx.write_graphml(self.graph, self.file_path)

    async def async_save(self):
        """Async version that runs save in executor to avoid blocking."""
        async with self._lock:
            await asyncio.to_thread(nx.write_graphml, self.graph, self.file_path)
            logger.info(f"Graph saved to {self.file_path}")