import asyncio
import inspect
import time
from typing import Any

import numpy as np
from aiolimiter import AsyncLimiter

from app.chunking import preserve_markdown_code_excerpts
from app.definitions import INPUT_DOCS_DIR, SOURCE_TO_DOC_ID_KV_PATH, DOC_ID_TO_SOURCE_KV_PATH, EMBEDDINGS_DB, \
    EXCERPT_KV_PATH, DOC_ID_TO_EXCERPT_KV_PATH, DOC_ID_TO_ENTITY_IDS_KV_PATH, DOC_ID_TO_RELATIONSHIP_IDS_KV_PATH, \
    ENTITY_ID_TO_DOC_IDS_KV_PATH, RELATIONSHIP_ID_TO_DOC_IDS_KV_PATH, KG_DB, ENTITIES_DB, RELATIONSHIPS_DB, KG_SEP, \
    TUPLE_SEP, REC_SEP, COMPLETE_TAG, LOG_DIR, COMPLETION_MODEL, EMBEDDING_MODEL
from app.graph_store import NetworkXGraphStore
from app.kv_store import JsonKvStore
from app.logger import logger, set_logger
from app.openai_llm import OpenAiLlm
from app.prompts import get_query_system_prompt, excerpt_summary_prompt, get_extract_entities_prompt, \
    get_high_low_level_keywords_prompt, get_kg_query_system_prompt, get_mix_system_prompt
from app.utilities import read_file, get_docs, make_hash, split_string_by_multi_markers, clean_str, \
    extract_json_from_text, truncate_list_by_token_size, \
    list_of_list_to_csv, delete_all_files
from app.vector_store import NanoVectorStore


class SmolRag:
    def __init__(
            self,
            excerpt_fn=None,
            llm=None,
            embeddings_db=None,
            entities_db=None,
            relationships_db=None,
            source_to_doc_kv=None,
            doc_to_source_kv=None,
            doc_to_excerpt_kv=None,
            doc_to_entity_kv=None,
            doc_to_relationship_kv=None,
            entity_to_doc_kv=None,
            relationship_to_doc_kv=None,
            excerpt_kv=None,
            query_cache_kv=None,
            embedding_cache_kv=None,
            graph_db=None,
            dimensions=None,
            excerpt_size=2000,
            overlap=200,
            ingest_concurrency=4
    ):
        set_logger("main.log")
        self.llm_limiter = AsyncLimiter(max_rate=100, time_period=1)
        self._provenance_lock = asyncio.Lock()

        self.excerpt_fn = excerpt_fn or preserve_markdown_code_excerpts
        self.excerpt_size = excerpt_size
        self.overlap = overlap
        self.ingest_concurrency = max(1, ingest_concurrency)

        self.llm = llm or OpenAiLlm(
            COMPLETION_MODEL,
            EMBEDDING_MODEL,
            query_cache_kv=query_cache_kv,
            embedding_cache_kv=embedding_cache_kv
        )

        self.dimensions = dimensions or 1536
        self.embeddings_db = embeddings_db or NanoVectorStore(EMBEDDINGS_DB, self.dimensions)
        self.entities_db = entities_db or NanoVectorStore(ENTITIES_DB, self.dimensions)
        self.relationships_db = relationships_db or NanoVectorStore(RELATIONSHIPS_DB, self.dimensions)

        self.source_to_doc_kv = source_to_doc_kv or JsonKvStore(SOURCE_TO_DOC_ID_KV_PATH)
        self.doc_to_source_kv = doc_to_source_kv or JsonKvStore(DOC_ID_TO_SOURCE_KV_PATH)
        self.doc_to_excerpt_kv = doc_to_excerpt_kv or JsonKvStore(DOC_ID_TO_EXCERPT_KV_PATH)
        self.doc_to_entity_kv = doc_to_entity_kv or JsonKvStore(DOC_ID_TO_ENTITY_IDS_KV_PATH)
        self.doc_to_relationship_kv = doc_to_relationship_kv or JsonKvStore(DOC_ID_TO_RELATIONSHIP_IDS_KV_PATH)
        self.entity_to_doc_kv = entity_to_doc_kv or JsonKvStore(ENTITY_ID_TO_DOC_IDS_KV_PATH)
        self.relationship_to_doc_kv = relationship_to_doc_kv or JsonKvStore(RELATIONSHIP_ID_TO_DOC_IDS_KV_PATH)
        self.excerpt_kv = excerpt_kv or JsonKvStore(EXCERPT_KV_PATH)

        self.graph = graph_db or NetworkXGraphStore(KG_DB)

    async def rate_limited_get_completion(self, *args, **kwargs):
        async with self.llm_limiter:
            return await self.llm.get_completion(*args, **kwargs)

    async def rate_limited_get_embedding(self, *args, **kwargs):
        async with self.llm_limiter:
            return await self.llm.get_embedding(*args, **kwargs)

    async def rate_limited_get_embeddings(self, *args, **kwargs):
        async with self.llm_limiter:
            return await self.llm.get_embeddings(*args, **kwargs)

    @staticmethod
    def _prune_kg_ids(raw_value: Any, ids_to_remove: set[str]) -> str:
        values = split_string_by_multi_markers(str(raw_value or ""), [KG_SEP])
        filtered = [value for value in values if value not in ids_to_remove]
        deduped = list(dict.fromkeys(filtered))
        return KG_SEP.join(deduped)

    def _find_entity_name_for_id(self, entity_id: str):
        for entity_name in self.graph.graph.nodes:
            if make_hash(entity_name, prefix="ent-") == entity_id:
                return entity_name
        return None

    def _find_relationship_endpoints_for_id(self, relationship_id: str):
        for source, target in self.graph.graph.edges:
            sorted_source, sorted_target = sorted((source, target))
            current_id = make_hash(f"{sorted_source}_{sorted_target}", prefix="rel-")
            if current_id == relationship_id:
                return sorted_source, sorted_target
        return None, None

    async def _cleanup_entity_contributions(self, doc_id: str, doc_excerpt_ids: set[str]) -> bool:
        entity_ids = await self.doc_to_entity_kv.get_by_key(doc_id)
        if entity_ids is None:
            return False
        if not entity_ids:
            await self.doc_to_entity_kv.remove(doc_id)
            return True

        rows = await self.entities_db.get(entity_ids)
        entity_rows_by_id = {row.get("__id__"): row for row in rows}
        ids_to_delete = []

        for entity_id in entity_ids:
            docs = await self.entity_to_doc_kv.get_by_key(entity_id) or []
            remaining_docs = [item for item in docs if item != doc_id]

            entity_row = entity_rows_by_id.get(entity_id)
            entity_name = entity_row.get("__entity_name__") if entity_row else None
            if entity_name is None:
                entity_name = self._find_entity_name_for_id(entity_id)
            node = self.graph.get_node(entity_name) if entity_name else None

            if node and "excerpt_id" in node:
                pruned_excerpt_ids = self._prune_kg_ids(node.get("excerpt_id"), doc_excerpt_ids)
                if remaining_docs and pruned_excerpt_ids:
                    updated_node = {**node, "excerpt_id": pruned_excerpt_ids}
                    await self.graph.async_add_node(entity_name, **updated_node)
                else:
                    await self.graph.async_remove_node(entity_name)

            if remaining_docs:
                await self.entity_to_doc_kv.add(entity_id, remaining_docs)
            else:
                await self.entity_to_doc_kv.remove(entity_id)
                ids_to_delete.append(entity_id)

        await self.doc_to_entity_kv.remove(doc_id)
        if ids_to_delete:
            await self.entities_db.delete(ids_to_delete)
        return True

    async def _cleanup_relationship_contributions(self, doc_id: str, doc_excerpt_ids: set[str]) -> bool:
        relationship_ids = await self.doc_to_relationship_kv.get_by_key(doc_id)
        if relationship_ids is None:
            return False
        if not relationship_ids:
            await self.doc_to_relationship_kv.remove(doc_id)
            return True

        rows = await self.relationships_db.get(relationship_ids)
        relationship_rows_by_id = {row.get("__id__"): row for row in rows}
        ids_to_delete = []

        for relationship_id in relationship_ids:
            docs = await self.relationship_to_doc_kv.get_by_key(relationship_id) or []
            remaining_docs = [item for item in docs if item != doc_id]

            relationship_row = relationship_rows_by_id.get(relationship_id)
            source = relationship_row.get("__source__") if relationship_row else None
            target = relationship_row.get("__target__") if relationship_row else None
            if not source or not target:
                source, target = self._find_relationship_endpoints_for_id(relationship_id)
            edge = self.graph.get_edge((source, target)) if source and target else None

            if edge and "excerpt_id" in edge:
                pruned_excerpt_ids = self._prune_kg_ids(edge.get("excerpt_id"), doc_excerpt_ids)
                if remaining_docs and pruned_excerpt_ids:
                    updated_edge = {**edge, "excerpt_id": pruned_excerpt_ids}
                    await self.graph.async_add_edge(source, target, **updated_edge)
                else:
                    await self.graph.async_remove_edge(source, target)

            if remaining_docs:
                await self.relationship_to_doc_kv.add(relationship_id, remaining_docs)
            else:
                await self.relationship_to_doc_kv.remove(relationship_id)
                ids_to_delete.append(relationship_id)

        await self.doc_to_relationship_kv.remove(doc_id)
        if ids_to_delete:
            await self.relationships_db.delete(ids_to_delete)
        return True

    async def _track_kg_provenance(self, doc_id: str, entity_ids: set[str], relationship_ids: set[str]):
        async with self._provenance_lock:
            await self.doc_to_entity_kv.add(doc_id, sorted(entity_ids))
            await self.doc_to_relationship_kv.add(doc_id, sorted(relationship_ids))

            for entity_id in entity_ids:
                docs = await self.entity_to_doc_kv.get_by_key(entity_id) or []
                if doc_id not in docs:
                    docs.append(doc_id)
                    await self.entity_to_doc_kv.add(entity_id, docs)

            for relationship_id in relationship_ids:
                docs = await self.relationship_to_doc_kv.get_by_key(relationship_id) or []
                if doc_id not in docs:
                    docs.append(doc_id)
                    await self.relationship_to_doc_kv.add(relationship_id, docs)

    async def remove_document_by_id(self, doc_id, persist=True):
        removed_source_map = False
        removed_excerpt_data = False
        removed_kg_data = False
        excerpt_ids = []

        if await self.doc_to_source_kv.has(doc_id):
            source = await self.doc_to_source_kv.get_by_key(doc_id)
            await asyncio.gather(self.doc_to_source_kv.remove(doc_id), self.source_to_doc_kv.remove(source))
            removed_source_map = True

        if await self.doc_to_excerpt_kv.has(doc_id):
            excerpt_ids = await self.doc_to_excerpt_kv.get_by_key(doc_id) or []
            excerpts_to_remove = [self.excerpt_kv.remove(excerpt_id) for excerpt_id in excerpt_ids]
            await asyncio.gather(self.embeddings_db.delete(excerpt_ids), *excerpts_to_remove)
            await self.doc_to_excerpt_kv.remove(doc_id)
            removed_excerpt_data = True

        excerpt_ids_set = set(excerpt_ids)
        async with self._provenance_lock:
            entity_removed = await self._cleanup_entity_contributions(doc_id, excerpt_ids_set)
            relationship_removed = await self._cleanup_relationship_contributions(doc_id, excerpt_ids_set)
        removed_kg_data = entity_removed or relationship_removed

        if not persist:
            return

        save_tasks = []
        if removed_source_map:
            save_tasks.extend([self.doc_to_source_kv.save(), self.source_to_doc_kv.save()])
        if removed_excerpt_data:
            save_tasks.extend([self.excerpt_kv.save(), self.doc_to_excerpt_kv.save(), self.embeddings_db.save()])
        if removed_kg_data:
            save_tasks.extend([
                self.doc_to_entity_kv.save(),
                self.doc_to_relationship_kv.save(),
                self.entity_to_doc_kv.save(),
                self.relationship_to_doc_kv.save(),
                self.entities_db.save(),
                self.relationships_db.save(),
            ])
            save_tasks.append(self.graph.async_save())
        if save_tasks:
            await asyncio.gather(*save_tasks)

    async def import_documents(self):
        sources = get_docs(INPUT_DOCS_DIR)
        semaphore = asyncio.Semaphore(self.ingest_concurrency)
        await asyncio.gather(*(self._process_source(source, semaphore) for source in sources))
        await self._save_stores()

    async def _save_stores(self):
        await asyncio.gather(
            self.source_to_doc_kv.save(),
            self.doc_to_source_kv.save(),
            self.excerpt_kv.save(),
            self.doc_to_excerpt_kv.save(),
            self.doc_to_entity_kv.save(),
            self.doc_to_relationship_kv.save(),
            self.entity_to_doc_kv.save(),
            self.relationship_to_doc_kv.save(),
            self.embeddings_db.save(),
            self.entities_db.save(),
            self.relationships_db.save()
        )
        self.graph.save()

    async def _process_source(self, source, semaphore):
        async with semaphore:
            content = read_file(source)
            doc_id = make_hash(content, "doc_")
            if not await self.source_to_doc_kv.has(source):
                logger.info(f"Importing new document: {source} (ID: {doc_id})")
                await self._add_document_maps(source, content)
                await self._embed_document(content, doc_id)
                await self._extract_entities(content, doc_id)
                return

            if not await self.source_to_doc_kv.equal(source, doc_id):
                logger.info(f"Updating document: {source} (New ID: {doc_id})")
                old_doc_id = await self.source_to_doc_kv.get_by_key(source)
                await self.remove_document_by_id(old_doc_id, persist=False)
                await self._add_document_maps(source, content)
                await self._embed_document(content, doc_id)
                await self._extract_entities(content, doc_id)
                return

            logger.debug(f"No changes detected for document: {source} (ID: {doc_id})")

    async def _add_document_maps(self, source, content):
        doc_id = make_hash(content, "doc_")
        await self.source_to_doc_kv.add(source, doc_id)
        await self.doc_to_source_kv.add(doc_id, source)
        # Saves are batched in import_documents()

    async def _embed_document(self, content, doc_id):
        start_time = time.time()
        excerpts = self.excerpt_fn(content, self.excerpt_size, self.overlap)
        excerpt_ids = []

        summary_tasks = [self._get_excerpt_summary(content, excerpt) for excerpt in excerpts]
        summaries = await asyncio.gather(*summary_tasks)

        # Batch all embeddings into a single API call
        embedding_contents = [f"{excerpt}\n\n{summary}" for excerpt, summary in zip(excerpts, summaries)]
        embedding_results = await self.rate_limited_get_embeddings(embedding_contents)
        storage_tasks = []
        for i, (excerpt, summary, embedding_result) in enumerate(zip(excerpts, summaries, embedding_results)):
            excerpt_id = make_hash(excerpt, "excerpt_id_")
            excerpt_ids.append(excerpt_id)
            vector = np.array(embedding_result, dtype=np.float32)
            storage_tasks.append(self.embeddings_db.upsert([
                {
                    "__id__": excerpt_id,
                    "__vector__": vector,
                    "__doc_id__": doc_id,
                    "__inserted_at__": time.time()
                }
            ]))
            storage_tasks.append(self.excerpt_kv.add(excerpt_id, {
                "doc_id": doc_id,
                "doc_order_index": i,
                "excerpt": excerpt,
                "summary": summary,
                "indexed_at": time.time()
            }))
            logger.info(f"Created embedding for excerpt {excerpt_id} associated with document {doc_id}")
        await asyncio.gather(*storage_tasks)

        await self.doc_to_excerpt_kv.add(doc_id, excerpt_ids)
        # Saves are batched in import_documents()
        elapsed = time.time() - start_time
        logger.info(f"Document {doc_id} processed with {len(excerpts)} excerpts in {elapsed:.2f} seconds.")

    async def _get_excerpt_summary(self, full_doc, excerpt):
        prompt = excerpt_summary_prompt(full_doc, excerpt)
        try:
            summary = await self.rate_limited_get_completion(prompt)
        except Exception as e:
            logger.error(f"LLM call in _get_excerpt_summary failed: {e}")
            summary = "Summary unavailable."
        return summary

    async def _extract_entities(self, content, doc_id):
        start_time = time.time()
        total_entities = 0
        total_relationships = 0
        doc_entity_ids = set()
        doc_relationship_ids = set()
        excerpts = self.excerpt_fn(content, self.excerpt_size, self.overlap)

        extract_entity_tasks = [self.rate_limited_get_completion(get_extract_entities_prompt(excerpt)) for excerpt in
                                excerpts]
        entity_results = await asyncio.gather(*extract_entity_tasks)

        for (excerpt, result) in zip(excerpts, entity_results):
            excerpt_id = make_hash(excerpt, "excerpt_id_")
            data_str = result.replace(COMPLETE_TAG, '').strip()
            records = split_string_by_multi_markers(data_str, [REC_SEP])

            clean_records = []
            for record in records:
                if record.startswith('(') and record.endswith(')'):
                    record = record[1:-1]
                clean_records.append(clean_str(record))
            records = clean_records

            entities_to_upsert = []
            relationships_to_upsert = []
            embedding_contents = []

            for record in records:
                fields = split_string_by_multi_markers(record, [TUPLE_SEP])
                if not fields:
                    continue
                fields = [field[1:-1] if field.startswith('"') and field.endswith('"') else field for field in fields]
                record_type = fields[0].lower()

                if record_type == 'entity':
                    if len(fields) >= 4:
                        _, name, category, description = fields[:4]
                        await self.graph.async_upsert_entity_node(
                            name=name,
                            category=category,
                            description=description,
                            excerpt_id=excerpt_id,
                            sep=KG_SEP,
                        )
                        total_entities += 1
                        entity_id = make_hash(name, prefix="ent-")
                        doc_entity_ids.add(entity_id)
                        embedding_content = f"{name} {description}"
                        embedding_contents.append(embedding_content)
                        entities_to_upsert.append({
                            "__id__": entity_id,
                            "__entity_name__": name,
                            "__inserted_at__": time.time(),
                        })
                elif record_type == 'relationship':
                    if len(fields) >= 6:
                        _, source, target, description, keywords, weight = fields[:6]
                        source, target = sorted([source, target])
                        try:
                            parsed_weight = float(weight)
                        except (TypeError, ValueError):
                            parsed_weight = 1.0

                        await self.graph.async_upsert_relationship_edge(
                            source=source,
                            destination=target,
                            description=description,
                            keywords=keywords,
                            weight=parsed_weight,
                            excerpt_id=excerpt_id,
                            sep=KG_SEP,
                        )
                        total_relationships += 1

                        relationship_id = make_hash(f"{source}_{target}", prefix="rel-")
                        doc_relationship_ids.add(relationship_id)
                        embedding_content = f"{keywords} {source} {target} {description}"
                        embedding_contents.append(embedding_content)
                        relationships_to_upsert.append({
                            "__id__": relationship_id,
                            "__source__": source,
                            "__target__": target,
                            "__inserted_at__": time.time(),
                        })
                elif record_type == 'content_keywords':
                    if len(fields) >= 2:
                        await self.graph.async_set_field('content_keywords', fields[1])

            # Batch all embeddings into a single API call
            if embedding_contents:
                results = await self.rate_limited_get_embeddings(embedding_contents)

                idx = 0
                for entity in entities_to_upsert:
                    vector = np.array(results[idx], dtype=np.float32)
                    entity["__vector__"] = vector
                    idx += 1

                for relation in relationships_to_upsert:
                    vector = np.array(results[idx], dtype=np.float32)
                    relation["__vector__"] = vector
                    idx += 1

            upsert_tasks = []
            if entities_to_upsert:
                upsert_tasks.append(self.entities_db.upsert(entities_to_upsert))
            if relationships_to_upsert:
                upsert_tasks.append(self.relationships_db.upsert(relationships_to_upsert))
            if upsert_tasks:
                await asyncio.gather(*upsert_tasks)

        await self._track_kg_provenance(doc_id, doc_entity_ids, doc_relationship_ids)
        # Saves are batched in import_documents()
        elapsed = time.time() - start_time
        logger.info(f"Extracted {total_entities} entities and {total_relationships} relationships "
                    f"from document {doc_id} in {elapsed:.2f} seconds.")

    async def query(self, text):
        logger.info(f"Received query: {text}")
        excerpts = await self._get_query_excerpts(text)
        logger.info(f"Retrieved {len(excerpts)} excerpts for the query.")
        excerpt_context = self._get_excerpt_context(excerpts)
        system_prompt = get_query_system_prompt(excerpt_context)

        return await self.rate_limited_get_completion(text, context=system_prompt.strip(), use_cache=True)

    def _get_excerpt_context(self, excerpts):
        context = ""
        for excerpt in excerpts:
            context += inspect.cleandoc(f"""
                ## Excerpt
    
                {excerpt["excerpt"]}
    
                ## Summary
    
                {excerpt["summary"]} 
            """)
            context += "\n\n"

        return context

    async def _get_query_excerpts(self, text):
        embedding = await self.rate_limited_get_embedding(text)
        embedding_array = np.array(embedding)
        results = await self.embeddings_db.query(query=embedding_array, top_k=5, better_than_threshold=0.02)
        excerpts = [self.excerpt_kv.get_by_key(result["__id__"]) for result in results]
        excerpts = await asyncio.gather(*excerpts)
        excerpts = [excerpt for excerpt in excerpts if excerpt is not None and "excerpt" in excerpt]
        excerpts = truncate_list_by_token_size(excerpts, get_text_for_row=lambda x: x["excerpt"], max_token_size=4000)
        return excerpts

    async def hybrid_kg_query(self, text):
        prompt = get_high_low_level_keywords_prompt(text)
        result = await self.rate_limited_get_completion(prompt)
        keyword_data = extract_json_from_text(result) or {}
        logger.info("Processed high/low level keywords for hybrid KG query.")

        ll_dataset, ll_entity_excerpts, ll_relations = await self._get_low_level_dataset(keyword_data)
        hl_dataset, hl_entities, hl_entity_excerpts = await self._get_high_level_dataset(keyword_data)

        entities = ll_dataset + hl_entities
        relations = ll_relations + hl_dataset
        excerpts = ll_entity_excerpts + hl_entity_excerpts
        context = self._get_kg_query_context(entities, excerpts, relations)
        system_prompt = get_kg_query_system_prompt(context)
        return await self.rate_limited_get_completion(text, context=system_prompt.strip(), use_cache=True)

    async def local_kg_query(self, text):
        prompt = get_high_low_level_keywords_prompt(text)
        result = await self.rate_limited_get_completion(prompt)
        keyword_data = extract_json_from_text(result) or {}
        logger.info("Processed high/low level keywords for local KG query.")

        ll_dataset, ll_entity_excerpts, ll_relations = await self._get_low_level_dataset(keyword_data)
        entities = ll_dataset
        relations = ll_relations
        excerpts = ll_entity_excerpts
        context = self._get_kg_query_context(entities, excerpts, relations)
        system_prompt = get_kg_query_system_prompt(context)
        return await self.rate_limited_get_completion(text, context=system_prompt.strip(), use_cache=True)

    async def global_kg_query(self, text):
        prompt = get_high_low_level_keywords_prompt(text)
        result = await self.rate_limited_get_completion(prompt)
        keyword_data = extract_json_from_text(result) or {}
        logger.info("Processed high/low level keywords for global KG query.")

        hl_dataset, hl_entities, hl_entity_excerpts = await self._get_high_level_dataset(keyword_data)
        entities = hl_entities
        relations = hl_dataset
        excerpts = hl_entity_excerpts
        context = self._get_kg_query_context(entities, excerpts, relations)
        system_prompt = get_kg_query_system_prompt(context)
        return await self.rate_limited_get_completion(text, context=system_prompt.strip(), use_cache=True)

    async def mix_query(self, text):
        prompt = get_high_low_level_keywords_prompt(text)
        result = await self.rate_limited_get_completion(prompt)
        keyword_data = extract_json_from_text(result) or {}
        logger.info("Processed high/low level keywords for mixed KG query.")

        ll_dataset, ll_entity_excerpts, ll_relations = await self._get_low_level_dataset(keyword_data)
        hl_dataset, hl_entities, hl_entity_excerpts = await self._get_high_level_dataset(keyword_data)

        kg_entities = ll_dataset + hl_entities
        kg_relations = ll_relations + hl_dataset
        kg_excerpts = ll_entity_excerpts + hl_entity_excerpts
        query_excerpts = await self._get_query_excerpts(text)
        kg_context = self._get_kg_query_context(kg_entities, kg_excerpts, kg_relations)
        excerpt_context = self._get_excerpt_context(query_excerpts)
        system_prompt = get_mix_system_prompt(excerpt_context, kg_context)
        return await self.rate_limited_get_completion(text, context=system_prompt.strip(), use_cache=True)

    def _get_kg_query_context(self, entities, excerpts, relations):
        entity_csv = [["entity", "type", "description", "rank"]]
        for entity in entities:
            entity_csv.append([
                entity["entity_name"],
                entity.get("category", "UNKNOWN"),
                entity.get("description", "UNKNOWN"),
                entity["rank"],
            ])
        entity_context = list_of_list_to_csv(entity_csv)
        relation_csv = [["source", "target", "description", "keywords", "weight", "rank"]]
        for relation in relations:
            relation_csv.append([
                relation["src_tgt"][0],
                relation["src_tgt"][1],
                relation["description"],
                relation["keywords"],
                relation["weight"],
                relation["rank"],
            ])
        relations_context = list_of_list_to_csv(relation_csv)
        excerpt_csv = [["excerpt"]]
        for excerpt in excerpts:
            excerpt_csv.append([excerpt["excerpt"]])
        excerpt_context = list_of_list_to_csv(excerpt_csv)
        context = inspect.cleandoc(f"""
            -----Entities-----
            ```csv
            {entity_context}
            ```
            -----Relationships-----
            ```csv
            {relations_context}
            ```
            -----Excerpts-----
            ```csv
            {excerpt_context}
            ```
        """)
        logger.info(f"KG query context built with {len(entities)} entities, {len(relations)} relationships, "
                    f"and {len(excerpts)} excerpts.")
        return context

    async def _get_high_level_dataset(self, keyword_data):
        keyword_data = keyword_data if isinstance(keyword_data, dict) else {}
        hl_keywords = self._normalize_keywords(keyword_data.get("high_level_keywords", []))
        logger.info(f"Found {len(hl_keywords)} high-level keywords.")
        hl_results = []
        if len(hl_keywords):
            hl_embedding = await self.rate_limited_get_embedding(hl_keywords)
            hl_embedding_array = np.array(hl_embedding)
            hl_results = await self.relationships_db.query(query=hl_embedding_array, top_k=25, better_than_threshold=0.02)
        hl_data = [self.graph.get_edge((r["__source__"], r["__target__"])) for r in hl_results]
        hl_degrees = [self.graph.degree(r["__source__"]) + self.graph.degree(r["__target__"]) for r in hl_results]
        hl_dataset = []
        for k, n, d in zip(hl_results, hl_data, hl_degrees):
            if n is None:
                logger.warning(
                    "Skipping stale relationship vector row because graph edge is missing: %s -> %s",
                    k.get("__source__"),
                    k.get("__target__"),
                )
                continue
            if not {"description", "keywords", "weight", "excerpt_id"}.issubset(set(n.keys())):
                logger.warning(
                    "Skipping relationship with incomplete graph payload for edge %s -> %s",
                    k.get("__source__"),
                    k.get("__target__"),
                )
                continue
            hl_dataset.append({"src_tgt": (k["__source__"], k["__target__"]), "rank": d, **n})
        hl_dataset = sorted(hl_dataset, key=lambda x: (x["rank"], x["weight"]), reverse=True)
        hl_dataset = truncate_list_by_token_size(
            hl_dataset,
            get_text_for_row=lambda x: x["description"],
            max_token_size=4000,
        )
        hl_entity_excerpts = await self._get_excerpts_for_relationships(hl_dataset)
        hl_entities = self._get_entities_from_relationships(hl_dataset)
        logger.info(f"High-level dataset: {len(hl_dataset)} relationships, {len(hl_entities)} entities extracted.")
        return hl_dataset, hl_entities, hl_entity_excerpts

    async def _get_low_level_dataset(self, keyword_data):
        keyword_data = keyword_data if isinstance(keyword_data, dict) else {}
        ll_keywords = self._normalize_keywords(keyword_data.get("low_level_keywords", []))
        logger.info(f"Found {len(ll_keywords)} low-level keywords.")
        ll_results = []
        if len(ll_keywords):
            ll_embedding = await self.rate_limited_get_embedding(ll_keywords)
            ll_embedding_array = np.array(ll_embedding)
            ll_results = await self.entities_db.query(query=ll_embedding_array, top_k=25, better_than_threshold=0.02)
        ll_data = [self.graph.get_node(r["__entity_name__"]) for r in ll_results]
        ll_degrees = [self.graph.degree(r["__entity_name__"]) for r in ll_results]
        ll_dataset = []
        for k, n, d in zip(ll_results, ll_data, ll_degrees):
            if n is None:
                logger.warning(
                    "Skipping stale entity vector row because graph node is missing: %s",
                    k.get("__entity_name__"),
                )
                continue
            if "excerpt_id" not in n:
                logger.warning(
                    "Skipping entity with incomplete graph payload: %s",
                    k.get("__entity_name__"),
                )
                continue
            ll_dataset.append({**n, "entity_name": k["__entity_name__"], "rank": d})
        ll_entity_excerpts = await self._get_excerpts_for_entities(ll_dataset)
        ll_relations = self._get_relationships_from_entities(ll_dataset)
        logger.info(f"Low-level dataset: {len(ll_dataset)} entities, {len(ll_relations)} relationships extracted.")
        return ll_dataset, ll_entity_excerpts, ll_relations

    async def _get_excerpts_for_entities(self, kg_dataset):
        excerpt_ids = [split_string_by_multi_markers(row["excerpt_id"], [KG_SEP]) for row in kg_dataset]
        all_edges = [self.graph.get_node_edges(row["entity_name"]) for row in kg_dataset]
        sibling_names = set()
        for edge in all_edges:
            if not edge:
                continue
            sibling_names.update([e[1] for e in edge])
        sibling_nodes = [self.graph.get_node(name) for name in list(sibling_names)]
        sibling_excerpt_lookup = {
            k: set(split_string_by_multi_markers(v["excerpt_id"], [KG_SEP]))
            for k, v in zip(sibling_names, sibling_nodes)
            if v is not None and "excerpt_id" in v
        }
        all_excerpt_data_lookup = {}
        for index, (excerpt_ids, edges) in enumerate(zip(excerpt_ids, all_edges)):
            for excerpt_id in excerpt_ids:
                if excerpt_id in all_excerpt_data_lookup:
                    continue
                relation_counts = 0
                if edges:
                    for edge in edges:
                        sibling_name = edge[1]
                        if sibling_name in sibling_excerpt_lookup and excerpt_id in sibling_excerpt_lookup[
                            sibling_name]:
                            relation_counts += 1
                excerpt_data = await self.excerpt_kv.get_by_key(excerpt_id)
                if excerpt_data is not None and "excerpt" in excerpt_data:
                    all_excerpt_data_lookup[excerpt_id] = {
                        "data": excerpt_data,
                        "order": index,
                        "relation_counts": relation_counts,
                    }

        all_excerpts = [
            {"id": k, **v}
            for k, v in all_excerpt_data_lookup.items()
            if v is not None and v.get("data") is not None and "excerpt" in v["data"]
        ]

        if not all_excerpts:
            logger.warning("No valid excerpts found for low-level entities.")
            return []

        all_excerpts = sorted(all_excerpts, key=lambda x: (x["order"], -x["relation_counts"]))
        all_excerpts = [t["data"] for t in all_excerpts]

        all_excerpts = truncate_list_by_token_size(
            all_excerpts,
            get_text_for_row=lambda x: x["excerpt"],
            max_token_size=4000,
        )
        logger.info(f"Extracted {len(all_excerpts)} excerpts for low-level entities.")
        return all_excerpts

    def _get_relationships_from_entities(self, kg_dataset):
        node_edges_list = [self.graph.get_node_edges(row["entity_name"]) for row in kg_dataset]

        edges = []
        seen = set()

        for node_edges in node_edges_list:
            for edge in node_edges:
                sorted_edge = tuple(sorted(edge))
                if sorted_edge not in seen:
                    seen.add(sorted_edge)
                    edges.append(sorted_edge)

        edges_pack = [self.graph.get_edge((e[0], e[1])) for e in edges]
        edges_degree = [self.graph.degree(e[0]) + self.graph.degree(e[1]) for e in edges]

        edges_data = [
            {"src_tgt": k, "rank": d, **v}
            for k, v, d in zip(edges, edges_pack, edges_degree)
            if v is not None
        ]
        edges_data = sorted(edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True)
        edges_data = truncate_list_by_token_size(
            edges_data,
            get_text_for_row=lambda x: x["description"],
            max_token_size=1000,
        )
        logger.info(f"Extracted {len(edges_data)} relationships from low-level entities.")
        return edges_data

    async def _get_excerpts_for_relationships(self, kg_dataset):
        excerpt_ids = [
            split_string_by_multi_markers(dp["excerpt_id"], [KG_SEP])
            for dp in kg_dataset
        ]

        all_excerpts_lookup = {}

        for index, excerpt_ids in enumerate(excerpt_ids):
            for excerpt_id in excerpt_ids:
                if excerpt_id not in all_excerpts_lookup:
                    all_excerpts_lookup[excerpt_id] = {
                        "data": await self.excerpt_kv.get_by_key(excerpt_id),
                        "order": index,
                    }

        if any([v is None for v in all_excerpts_lookup.values()]):
            logger.warning("Text chunks are missing, maybe the storage is damaged")
        all_excerpts = [
            {"id": k, **v} for k, v in all_excerpts_lookup.items() if v is not None
        ]
        all_excerpts = sorted(all_excerpts, key=lambda x: x["order"])
        # Todo: figure out how t["data"] is None
        all_excerpts = [t["data"] for t in all_excerpts if t["data"] is not None]

        all_excerpts = truncate_list_by_token_size(
            all_excerpts,
            get_text_for_row=lambda x: x["excerpt"],
            max_token_size=4000,
        )

        return all_excerpts

    def _get_entities_from_relationships(self, kg_dataset):
        entity_names = []
        seen = set()

        for e in kg_dataset:
            if e["src_tgt"][0] not in seen:
                entity_names.append(e["src_tgt"][0])
                seen.add(e["src_tgt"][0])
            if e["src_tgt"][1] not in seen:
                entity_names.append(e["src_tgt"][1])
                seen.add(e["src_tgt"][1])

        data = [self.graph.get_node(entity_name) for entity_name in entity_names]
        degrees = [self.graph.degree(entity_name) for entity_name in entity_names]

        # Todo: we need to filter out missing node data (ie no description) in case the node was added as an edge only
        data = [
            {**n, "entity_name": k, "rank": d}
            for k, n, d in zip(entity_names, data, degrees)
            if n is not None and "description" in n
        ]

        # Todo: figure out how we hit a bug here with missing description
        data = truncate_list_by_token_size(
            data,
            get_text_for_row=lambda x: x["description"],
            max_token_size=4000,
        )
        logger.info(f"Extracted {len(data)} entities from relationships.")
        return data

    @staticmethod
    def _normalize_keywords(keywords):
        if keywords is None:
            return []
        if isinstance(keywords, str):
            return [keywords] if keywords.strip() else []
        if not isinstance(keywords, list):
            return []
        return [str(k) for k in keywords if str(k).strip()]


if __name__ == '__main__':
    async def main():
        # delete_all_files(DATA_DIR)
        delete_all_files(LOG_DIR)

        smol_rag = SmolRag()

        await smol_rag.import_documents()

        print(await smol_rag.query("what is SmolRag?"))  # Should answer
        print("=+=+=+=+=+=+=+=+=+=+=+=+=+=")
        print(await smol_rag.query("what do cats eat?"))  # Should reject
        print("=+=+=+=+=+=+=+=+=+=+=+=+=+=")
        print(await smol_rag.query("What subjects we can discuss?"))  # Should answer

        print(await smol_rag.hybrid_kg_query("what is SmolRag?"))  # Should answer
        print("=+=+=+=+=+=+=+=+=+=+=+=+=+=")
        print(await smol_rag.hybrid_kg_query("what do cows eat?"))  # Should reject
        print("=+=+=+=+=+=+=+=+=+=+=+=+=+=")
        print(await smol_rag.hybrid_kg_query("What subjects we can discuss?"))

        print(await smol_rag.local_kg_query("what is SmolRag?"))  # Should answer
        print("=+=+=+=+=+=+=+=+=+=+=+=+=+=")
        print(await smol_rag.local_kg_query("what do ducks eat?"))  # Should reject
        print("=+=+=+=+=+=+=+=+=+=+=+=+=+=")
        print(await smol_rag.local_kg_query("What subjects we can discuss?"))

        print(await smol_rag.global_kg_query("what is SmolRag?"))  # Should answer
        print("=+=+=+=+=+=+=+=+=+=+=+=+=+=")
        print(await smol_rag.global_kg_query("what do frogs eat?"))  # Should reject
        print("=+=+=+=+=+=+=+=+=+=+=+=+=+=")
        print(await smol_rag.global_kg_query("What subjects we can discuss?"))

        print(await smol_rag.mix_query("what is SmolRag?"))  # Should answer
        print("=+=+=+=+=+=+=+=+=+=+=+=+=+=")
        print(await smol_rag.mix_query("what do jellyfish eat?"))  # Should reject
        print("=+=+=+=+=+=+=+=+=+=+=+=+=+=")
        print(await smol_rag.mix_query("What subjects we can discuss?"))

        # await smol_rag.remove_document_by_id("doc_68ee570c562a4cdfb5c37cf96be2898d")



    asyncio.run(main())
