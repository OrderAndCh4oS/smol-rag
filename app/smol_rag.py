import asyncio
import inspect
import time

import numpy as np
from aiolimiter import AsyncLimiter

from app.chunking import preserve_markdown_code_excerpts
from app.definitions import INPUT_DOCS_DIR, SOURCE_TO_DOC_ID_KV_PATH, DOC_ID_TO_SOURCE_KV_PATH, EMBEDDINGS_DB, \
    EXCERPT_KV_PATH, DOC_ID_TO_EXCERPT_KV_PATH, KG_DB, ENTITIES_DB, RELATIONSHIPS_DB, KG_SEP, TUPLE_SEP, REC_SEP, \
    COMPLETE_TAG, LOG_DIR, COMPLETION_MODEL, EMBEDDING_MODEL
from app.graph_store import NetworkXGraphStore
from app.kv_store import JsonKvStore
from app.logger import logger, set_logger
from app.openai_llm import OpenAiLlm
from app.prompts import get_query_system_prompt, excerpt_summary_prompt, get_extract_entities_prompt, \
    get_high_low_level_keywords_prompt, get_kg_query_system_prompt, get_mix_system_prompt
from app.utilities import read_file, get_docs, make_hash, split_string_by_multi_markers, clean_str, \
    extract_json_from_text, is_float_regex, truncate_list_by_token_size, \
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
            excerpt_kv=None,
            query_cache_kv=None,
            embedding_cache_kv=None,
            graph_db=None,
            dimensions=None,
            excerpt_size=2000,
            overlap=200
    ):
        set_logger("main.log")
        self.llm_limiter = AsyncLimiter(max_rate=100, time_period=1)

        self.excerpt_fn = excerpt_fn or preserve_markdown_code_excerpts
        self.excerpt_size = excerpt_size
        self.overlap = overlap

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

    async def remove_document_by_id(self, doc_id):
        if await self.doc_to_source_kv.has(doc_id):
            source = await self.doc_to_source_kv.get_by_key(doc_id)
            await asyncio.gather(self.doc_to_source_kv.remove(doc_id), self.source_to_doc_kv.remove(source))
            await asyncio.gather(self.doc_to_source_kv.save(), self.source_to_doc_kv.save())
        if await self.doc_to_excerpt_kv.has(doc_id):
            excerpt_ids = await self.doc_to_excerpt_kv.get_by_key(doc_id)
            print(excerpt_ids)
            excerpts_to_remove = [self.excerpt_kv.remove(excerpt_id) for excerpt_id in excerpt_ids]
            await asyncio.gather(self.embeddings_db.delete(excerpt_ids), *excerpts_to_remove)
            await self.doc_to_excerpt_kv.remove(doc_id)
            await asyncio.gather(self.excerpt_kv.save(), self.doc_to_excerpt_kv.save())
            await self.embeddings_db.save()

    async def import_documents(self):
        sources = get_docs(INPUT_DOCS_DIR)
        tasks = []
        for source in sources:
            content = read_file(source)
            doc_id = make_hash(content, "doc_")
            if not await self.source_to_doc_kv.has(source):
                logger.info(f"Importing new document: {source} (ID: {doc_id})")
                tasks.append(self._add_document_maps(source, content))
                tasks.append(self._embed_document(content, doc_id))
                tasks.append(self._extract_entities(content, doc_id))
            elif not await self.source_to_doc_kv.equal(source, doc_id):
                logger.info(f"Updating document: {source} (New ID: {doc_id})")
                old_doc_id = await self.source_to_doc_kv.get_by_key(source)
                tasks.append(self.remove_document_by_id(old_doc_id))
                tasks.append(self._add_document_maps(source, content))
                tasks.append(self._embed_document(content, doc_id))
                tasks.append(self._extract_entities(content, doc_id))
            else:
                logger.debug(f"No changes detected for document: {source} (ID: {doc_id})")

        await asyncio.gather(*tasks)

        # Batch save all stores once at the end
        await asyncio.gather(
            self.source_to_doc_kv.save(),
            self.doc_to_source_kv.save(),
            self.excerpt_kv.save(),
            self.doc_to_excerpt_kv.save(),
            self.embeddings_db.save(),
            self.entities_db.save(),
            self.relationships_db.save()
        )
        self.graph.save()

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
                        existing_node = self.graph.get_node(name)
                        if existing_node:
                            existing_descriptions = split_string_by_multi_markers(existing_node["description"],
                                                                                  [KG_SEP])
                            # Use set to remove duplicates and limit to 10 most recent descriptions
                            unique_descriptions = list(set(existing_descriptions + [description]))
                            descriptions = KG_SEP.join(unique_descriptions[-10:])
                            existing_categories = split_string_by_multi_markers(existing_node["category"], [KG_SEP])
                            categories = KG_SEP.join(set(list(existing_categories) + [category]))
                            existing_excerpt_ids = split_string_by_multi_markers(existing_node["excerpt_id"], [KG_SEP])
                            excerpt_ids = KG_SEP.join(set(list(existing_excerpt_ids) + [excerpt_id]))
                            # Todo: summarise descriptions with LLM query if they get too long
                            self.graph.add_node(
                                name,
                                category=categories,
                                description=descriptions,
                                excerpt_id=excerpt_ids
                            )
                        else:
                            self.graph.add_node(name, category=category, description=description, excerpt_id=excerpt_id)
                        total_entities += 1
                        entity_id = make_hash(name, prefix="ent-")
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
                        # Todo: summarise descriptions with LLM query if they get too long
                        existing_edge = self.graph.get_edge((source, target))
                        weight = float(weight) if is_float_regex(weight) else 1.0
                        if existing_edge:
                            existing_descriptions = split_string_by_multi_markers(existing_edge["description"],
                                                                                  [KG_SEP])
                            # Use set to remove duplicates and limit to 10 most recent descriptions
                            unique_descriptions = list(set(existing_descriptions + [description]))
                            descriptions = KG_SEP.join(unique_descriptions[-10:])
                            existing_keywords = split_string_by_multi_markers(existing_edge["keywords"], [KG_SEP])
                            # Limit keywords to 20 unique entries
                            unique_keywords = list(set(existing_keywords + [keywords]))
                            keywords = KG_SEP.join(unique_keywords[-20:])
                            existing_excerpt_ids = split_string_by_multi_markers(existing_edge["excerpt_id"], [KG_SEP])
                            excerpt_ids = KG_SEP.join(set(list(existing_excerpt_ids) + [excerpt_id]))
                            weight = sum([weight, existing_edge["weight"]])
                            self.graph.add_edge(source, target, description=descriptions, keywords=keywords,
                                                weight=weight,
                                                excerpt_id=excerpt_ids)
                        else:
                            self.graph.add_edge(source, target, description=description, keywords=keywords,
                                                weight=weight,
                                                excerpt_id=excerpt_id)
                        total_relationships += 1

                        relationship_id = make_hash(f"{source}_{target}", prefix="ent-")
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
                        self.graph.set_field('content_keywords', fields[1])

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

            await asyncio.gather(self.entities_db.upsert(entities_to_upsert), self.relationships_db.upsert(relationships_to_upsert))

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
        excerpts = truncate_list_by_token_size(excerpts, get_text_for_row=lambda x: x["excerpt"], max_token_size=4000)
        return excerpts

    async def hybrid_kg_query(self, text):
        prompt = get_high_low_level_keywords_prompt(text)
        result = await self.rate_limited_get_completion(prompt)
        keyword_data = extract_json_from_text(result)
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
        keyword_data = extract_json_from_text(result)
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
        keyword_data = extract_json_from_text(result)
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
        keyword_data = extract_json_from_text(result)
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
        hl_keywords = keyword_data.get("high_level_keywords", [])
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
            hl_dataset.append({"src_tgt": (k["__source__"], k["__target__"]), "rank": d, **n, })
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
        ll_keywords = keyword_data.get("low_level_keywords", [])
        logger.info(f"Found {len(ll_keywords)} low-level keywords.")
        ll_results = []
        if len(ll_keywords):
            ll_embedding = await self.rate_limited_get_embedding(ll_keywords)
            ll_embedding_array = np.array(ll_embedding)
            ll_results = await self.entities_db.query(query=ll_embedding_array, top_k=25, better_than_threshold=0.02)
        ll_data = [self.graph.get_node(r["__entity_name__"]) for r in ll_results]
        ll_degrees = [self.graph.degree(r["__entity_name__"]) for r in ll_results]
        ll_dataset = [
            {**n, "entity_name": k["__entity_name__"], "rank": d}
            for k, n, d in zip(ll_results, ll_data, ll_degrees)
        ]
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
            for k, n, d in zip(entity_names, data, degrees) if 'description' in n
        ]

        # Todo: figure out how we hit a bug here with missing description
        data = truncate_list_by_token_size(
            data,
            get_text_for_row=lambda x: x["description"],
            max_token_size=4000,
        )
        logger.info(f"Extracted {len(data)} entities from relationships.")
        return data


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
