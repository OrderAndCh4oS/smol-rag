import asyncio

from nano_vectordb import NanoVectorDB


class NanoVectorStore:
    def __init__(self, storage_file, dimensions):
        self.storage_file = storage_file
        self.dimensions = dimensions
        self.db = NanoVectorDB(self.dimensions, storage_file=storage_file)
        self._lock = asyncio.Lock()

    async def upsert(self, rows):
        async with self._lock:
            self.db.upsert(rows)

    async def delete(self, ids):
        async with self._lock:
            self.db.delete(ids)

    async def query(self, query, top_k=10, better_than_threshold=0.02):
        async with self._lock:
            return self.db.query(query=query, top_k=top_k, better_than_threshold=better_than_threshold)

    async def get(self, ids):
        async with self._lock:
            return self.db.get(ids)

    async def save(self):
        async with self._lock:
            self.db.save()
