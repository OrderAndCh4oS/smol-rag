import os
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("nltk")

import rebuild_rag
from app.graph_store import NetworkXGraphStore
from app.kv_store import JsonKvStore
from app.smol_rag import SmolRag
from app.vector_store import NanoVectorStore


def _build_rag(temp_dir, llm):
    return SmolRag(
        llm=llm,
        excerpt_fn=lambda text, _size, _overlap: [text],
        embeddings_db=NanoVectorStore(os.path.join(temp_dir, "embeddings"), dimensions=1536),
        entities_db=NanoVectorStore(os.path.join(temp_dir, "entities"), dimensions=1536),
        relationships_db=NanoVectorStore(os.path.join(temp_dir, "relationships"), dimensions=1536),
        source_to_doc_kv=JsonKvStore(os.path.join(temp_dir, "source_to_doc.json")),
        doc_to_source_kv=JsonKvStore(os.path.join(temp_dir, "doc_to_source.json")),
        doc_to_excerpt_kv=JsonKvStore(os.path.join(temp_dir, "doc_to_excerpt.json")),
        doc_to_entity_kv=JsonKvStore(os.path.join(temp_dir, "doc_to_entity.json")),
        doc_to_relationship_kv=JsonKvStore(os.path.join(temp_dir, "doc_to_relationship.json")),
        entity_to_doc_kv=JsonKvStore(os.path.join(temp_dir, "entity_to_doc.json")),
        relationship_to_doc_kv=JsonKvStore(os.path.join(temp_dir, "relationship_to_doc.json")),
        excerpt_kv=JsonKvStore(os.path.join(temp_dir, "excerpt.json")),
        graph_db=NetworkXGraphStore(os.path.join(temp_dir, "graph.graphml")),
    )


@pytest.mark.asyncio
async def test_import_documents_skips_unchanged_source(temp_dir, mock_openai_llm):
    doc_path = os.path.join(temp_dir, "doc.md")
    with open(doc_path, "w") as f:
        f.write("original content")

    rag = _build_rag(temp_dir=temp_dir, llm=mock_openai_llm)

    with patch("app.smol_rag.get_docs", return_value=[doc_path]):
        await rag.import_documents()

    rag._embed_document = AsyncMock()
    rag._extract_entities = AsyncMock()

    with patch("app.smol_rag.get_docs", return_value=[doc_path]):
        await rag.import_documents()

    rag._embed_document.assert_not_called()
    rag._extract_entities.assert_not_called()


@pytest.mark.asyncio
async def test_import_documents_reprocesses_changed_source(temp_dir, mock_openai_llm):
    doc_path = os.path.join(temp_dir, "doc.md")
    with open(doc_path, "w") as f:
        f.write("v1")

    rag = _build_rag(temp_dir=temp_dir, llm=mock_openai_llm)

    with patch("app.smol_rag.get_docs", return_value=[doc_path]):
        await rag.import_documents()

    rag.remove_document_by_id = AsyncMock()
    rag._embed_document = AsyncMock()
    rag._extract_entities = AsyncMock()

    with open(doc_path, "w") as f:
        f.write("v2")

    with patch("app.smol_rag.get_docs", return_value=[doc_path]):
        await rag.import_documents()

    rag.remove_document_by_id.assert_awaited_once()
    rag._embed_document.assert_awaited_once()
    rag._extract_entities.assert_awaited_once()


class _FakeRag:
    def __init__(self):
        self.import_called = False
        self.queries = []

    async def import_documents(self):
        self.import_called = True

    async def query(self, text):
        self.queries.append(text)
        return "ok"


@pytest.mark.asyncio
async def test_rebuild_main_keep_existing_does_not_wipe_state(temp_dir, monkeypatch):
    state_files = [os.path.join(temp_dir, "a.json"), os.path.join(temp_dir, "b.json")]
    for path in state_files:
        with open(path, "w") as f:
            f.write("state")

    created = []

    def fake_ctor():
        rag = _FakeRag()
        created.append(rag)
        return rag

    monkeypatch.setattr(rebuild_rag, "STATE_FILES", state_files)
    monkeypatch.setattr(rebuild_rag, "SmolRag", fake_ctor)

    exit_code = await rebuild_rag.main(wipe=False)

    assert exit_code == 0
    assert len(created) == 1
    assert created[0].import_called is True
    assert len(created[0].queries) == 3
    assert all(os.path.exists(path) for path in state_files)


@pytest.mark.asyncio
async def test_rebuild_main_wipe_removes_state(temp_dir, monkeypatch):
    state_files = [os.path.join(temp_dir, "a.json"), os.path.join(temp_dir, "b.json")]
    for path in state_files:
        with open(path, "w") as f:
            f.write("state")

    monkeypatch.setattr(rebuild_rag, "STATE_FILES", state_files)
    monkeypatch.setattr(rebuild_rag, "SmolRag", lambda: _FakeRag())

    exit_code = await rebuild_rag.main(wipe=True)

    assert exit_code == 0
    assert all(not os.path.exists(path) for path in state_files)
