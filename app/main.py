import inspect
import os
import time

import networkx as nx
import numpy as np

from nano_vectordb import NanoVectorDB

from app.llm import get_embedding, get_completion
from app.logger import logger, set_logger

from app.definitions import INPUT_DOCS_DIR, SOURCE_TO_DOC_ID_MAP, DOC_ID_TO_SOURCE_MAP, EMBEDDINGS_DB, EXCERPT_DB, \
    DOC_ID_TO_EXCERPT_IDS, KG_DB, ENTITIES_DB, RELATIONSHIPS_DB, KG_SEP, TUPLE_SEP, REC_SEP, COMPLETE_TAG, QUERY_CACHE, \
    EMBEDDING_CACHE, DATA_DIR, LOG_DIR
from app.prompts import get_query_system_prompt, excerpt_summary_prompt, get_extract_entities_prompt, \
    get_high_low_level_keywords_prompt, get_kg_query_system_prompt, get_mix_system_prompt

from app.utilities import get_json, remove_from_json, read_file, get_docs, make_hash, add_to_json, \
    split_string_by_multi_markers, clean_str, extract_json_from_text, is_float_regex, truncate_list_by_token_size, \
    list_of_list_to_csv, get_encoded_tokens, create_file_if_not_exists, delete_all_files

dim = 1536
embeddings_db = NanoVectorDB(dim, storage_file=EMBEDDINGS_DB)
entities_db = NanoVectorDB(dim, storage_file=ENTITIES_DB)
relationships_db = NanoVectorDB(dim, storage_file=RELATIONSHIPS_DB)


def remove_document_by_id(doc_id):
    doc_id_to_excerpt_ids = get_json(DOC_ID_TO_EXCERPT_IDS)
    doc_id_to_source_map = get_json(DOC_ID_TO_SOURCE_MAP)
    if doc_id in doc_id_to_source_map:
        source = doc_id_to_source_map[doc_id]
        remove_from_json(DOC_ID_TO_SOURCE_MAP, doc_id)
        remove_from_json(SOURCE_TO_DOC_ID_MAP, source)
    if doc_id in doc_id_to_excerpt_ids:
        excerpt_ids = doc_id_to_excerpt_ids[doc_id]
        for excerpt_id in excerpt_ids:
            remove_from_json(EXCERPT_DB, excerpt_id)
        remove_from_json(DOC_ID_TO_EXCERPT_IDS, doc_id)
        embeddings_db.delete(excerpt_ids)
        embeddings_db.save()


def import_documents():
    sources = get_docs(INPUT_DOCS_DIR)
    for source in sources:
        content = read_file(source)
        doc_id = make_hash(content, "doc_")

        source_to_doc_id_map = get_json(SOURCE_TO_DOC_ID_MAP)

        if source not in source_to_doc_id_map:
            logger.info(f"importing new document {source} with id {doc_id}")
            add_document_maps(source, content)
            embed_document(content, doc_id)
            extract_entities(content, doc_id)
        elif source_to_doc_id_map[source] != doc_id:
            logger.info(f"updating existing document {source} with id {doc_id}")
            old_doc_id = source_to_doc_id_map[source]
            remove_document_by_id(old_doc_id)
            add_document_maps(source, content)
            embed_document(content, doc_id)
            extract_entities(content, doc_id)
        else:
            logger.info(f"no changes, skipping document {source} with id {doc_id}")


def add_document_maps(source, content):
    doc_id = make_hash(content, "doc_")
    add_to_json(SOURCE_TO_DOC_ID_MAP, source, doc_id)
    add_to_json(DOC_ID_TO_SOURCE_MAP, doc_id, source)


def embed_document(content, doc_id):
    excerpts = get_excerpts(content)
    excerpt_ids = []
    for i, excerpt in enumerate(excerpts):
        excerpt_id = make_hash(excerpt, "excerpt_id_")
        excerpt_ids.append(excerpt_id)
        summary = get_excerpt_summary(content, excerpt)
        embedding_content = f"{excerpt}\n\n{summary}"
        embedding_result = get_embedding(embedding_content)
        vector = np.array(embedding_result, dtype=np.float32)
        embeddings_db.upsert([
            {"__id__": excerpt_id, "__vector__": vector, "__doc_id__": doc_id, "__inserted_at__": time.time()}
        ])
        add_to_json(EXCERPT_DB, excerpt_id, {
            "doc_id": doc_id,
            "doc_order_index": i,
            "excerpt": excerpt,
            "summary": summary,
            "indexed_at": time.time()
        })
        logger.info(f"created embedding for {excerpt_id} — {embedding_result}")

    embeddings_db.save()
    add_to_json(DOC_ID_TO_EXCERPT_IDS, doc_id, excerpt_ids)

# Todo: write smart chunking function to extract excerpts without chopping words
def get_excerpts(content, n=2000, overlap=200):
    excerpts = []
    step = n - overlap
    for i in range(0, len(content), step):
        excerpts.append(content[i:i + n])
    return excerpts


def get_excerpt_summary(full_doc, excerpt):
    prompt = excerpt_summary_prompt(full_doc, excerpt)
    summary = get_completion(prompt)

    logger.info(f"Excerpt:\n{excerpt}\n\nSummary:\n{summary}")

    return summary


def extract_entities(content, doc_id):
    excerpts = get_excerpts(content)
    if os.path.exists(KG_DB):
        try:
            graph = nx.read_graphml(KG_DB)
            logger.info(f"Loaded existing graph from {KG_DB}")
        except Exception as e:
            logger.error(f"Error loading graph from {KG_DB}: {e}")
            graph = nx.Graph()
    else:
        graph = nx.Graph()
        logger.info("No existing graph found. Creating a new graph.")

    for excerpt in excerpts:
        excerpt_id = make_hash(excerpt, "excerpt_id_")
        result = get_completion(get_extract_entities_prompt(excerpt))
        logger.info(result)

        data_str = result.replace(COMPLETE_TAG, '').strip()

        records = split_string_by_multi_markers(data_str, [REC_SEP])

        clean_records = []
        for record in records:
            if record.startswith('(') and record.endswith(')'):
                record = record[1:-1]
            clean_records.append(clean_str(record))
        records = clean_records

        for record in records:
            fields = split_string_by_multi_markers(record, [TUPLE_SEP])
            logger.info("FIELDS: " + ",".join(fields))
            if not fields:
                continue
            fields = [field[1:-1] if field.startswith('"') and field.endswith('"') else field for field in fields]
            logger.info("FIELDS TWO: " + ",".join(fields))
            record_type = fields[0].lower()
            logger.info(f"{record_type} {len(fields)}")
            if record_type == 'entity':
                if len(fields) >= 4:
                    _, name, category, description = fields[:4]
                    logger.info(f"Entity - Name: {name}, Category: {category}, Description: {description}")
                    existing_node = graph.nodes.get(name)
                    if existing_node:
                        logger.info("NODE:" + str(existing_node))
                        existing_descriptions = split_string_by_multi_markers(existing_node["description"], KG_SEP)
                        descriptions = KG_SEP.join(set(list(existing_descriptions) + [description]))
                        existing_categories = split_string_by_multi_markers(existing_node["category"], KG_SEP)
                        categories = KG_SEP.join(set(list(existing_categories) + [category]))
                        existing_excerpt_ids = split_string_by_multi_markers(existing_node["excerpt_id"], KG_SEP)
                        excerpt_ids = KG_SEP.join(set(list(existing_excerpt_ids) + [excerpt_id]))
                        # Todo: summarise descriptions with LLM query if they get too long
                        graph.add_node(
                            name,
                            category=categories,
                            description=descriptions,
                            excerpt_id=excerpt_ids
                        )
                    else:
                        graph.add_node(name, category=category, description=description, excerpt_id=excerpt_id)
                    entity_id = make_hash(name, prefix="ent-")
                    embedding_content = f"{name} {description}"
                    embedding_result = get_embedding(embedding_content)
                    vector = np.array(embedding_result, dtype=np.float32)
                    entities_db.upsert([
                        {
                            "__id__": entity_id,
                            "__vector__": vector,
                            "__entity_name__": name,
                            "__inserted_at__": time.time(),
                        }
                    ])
            elif record_type == 'relationship':
                if len(fields) >= 6:
                    _, source, target, description, keywords, weight = fields[:6]
                    source, target = sorted([source, target])
                    logger.info(
                        f"Relationship - Source: {source}, Target: {target}, Description: {description}, Keywords: {keywords}, Weight: {weight}, Excerpt ID: {excerpt_id}")
                    # Todo: summarise descriptions with LLM query if they get too long
                    existing_edge = graph.edges.get((source, target))
                    weight = float(weight) if is_float_regex(weight) else 1.0
                    if existing_edge:
                        existing_descriptions = split_string_by_multi_markers(existing_edge["description"], KG_SEP)
                        descriptions = KG_SEP.join(set(list(existing_descriptions) + [description]))
                        existing_keywords = split_string_by_multi_markers(existing_edge["keywords"], KG_SEP)
                        keywords = KG_SEP.join(set(list(existing_keywords) + [keywords]))
                        existing_excerpt_ids = split_string_by_multi_markers(existing_edge["excerpt_id"], KG_SEP)
                        excerpt_ids = KG_SEP.join(set(list(existing_excerpt_ids) + [excerpt_id]))
                        weight = sum([weight, existing_edge["weight"]])
                        graph.add_edge(source, target, description=descriptions, keywords=keywords, weight=weight,
                                       excerpt_id=excerpt_ids)
                    else:
                        graph.add_edge(source, target, description=description, keywords=keywords, weight=weight,
                                       excerpt_id=excerpt_id)
                    relationship_id = make_hash(f"{source}_{target}", prefix="ent-")
                    embedding_content = f"{keywords} {source} {target} {description}"
                    embedding_result = get_embedding(embedding_content)
                    vector = np.array(embedding_result, dtype=np.float32)
                    relationships_db.upsert([
                        {
                            "__id__": relationship_id,
                            "__vector__": vector,
                            "__source__": source,
                            "__target__": target,
                            "__inserted_at__": time.time(),
                        }
                    ])
            elif record_type == 'content_keywords':
                if len(fields) >= 2:
                    logger.info(f"Content Keywords: {fields[1]}")
                    graph.graph['content_keywords'] = fields[1]

    entities_db.save()
    relationships_db.save()

    nx.write_graphml(graph, KG_DB)


def query(text):
    logger.info(f"Received Query:\n{text}")
    excerpts = get_query_excerpts(text)
    excerpt_context = get_excerpt_context(excerpts)
    system_prompt = get_query_system_prompt(excerpt_context)

    return get_completion(text, context=system_prompt.strip(), use_cache=False)


def get_excerpt_context(excerpts):
    context = ""
    for i, excerpt in enumerate(excerpts):
        context += inspect.cleandoc(f"""
            ## Excerpt

            {excerpt["excerpt"]}

            ## Summary

            {excerpt["summary"]} 
        """)
        context += "\n\n"

    return context


def get_query_excerpts(text):
    embedding = get_embedding(text)
    embedding_array = np.array(embedding)
    results = embeddings_db.query(query=embedding_array, top_k=5, better_than_threshold=0.02)
    excerpt_db = get_json(EXCERPT_DB)
    excerpts = [excerpt_db[result["__id__"]] for result in results]
    excerpts = truncate_list_by_token_size(excerpts, get_text_for_row=lambda x: x["excerpt"], max_token_size=4000)
    return excerpts


def hybrid_kg_query(text):
    prompt = get_high_low_level_keywords_prompt(text)
    result = get_completion(prompt)
    keyword_data = extract_json_from_text(result)
    logger.info(f'keyword_data: {keyword_data}')

    graph = nx.read_graphml(KG_DB)

    ll_dataset, ll_entity_excerpts, ll_relations = get_low_level_dataset(graph, keyword_data)
    hl_dataset, hl_entities, hl_entity_excerpts = get_high_level_dataset(graph, keyword_data)

    entities = ll_dataset + hl_entities
    relations = ll_relations + hl_dataset
    excerpts = ll_entity_excerpts + hl_entity_excerpts

    context = get_kg_query_context(entities, excerpts, relations)

    system_prompt = get_kg_query_system_prompt(context)

    return get_completion(text, context=system_prompt.strip(), use_cache=False)


def local_kg_query(text):
    prompt = get_high_low_level_keywords_prompt(text)
    result = get_completion(prompt)
    keyword_data = extract_json_from_text(result)
    logger.info(f'keyword_data: {keyword_data}')

    graph = nx.read_graphml(KG_DB)

    ll_dataset, ll_entity_excerpts, ll_relations = get_low_level_dataset(graph, keyword_data)

    entities = ll_dataset
    relations = ll_relations
    excerpts = ll_entity_excerpts

    context = get_kg_query_context(entities, excerpts, relations)
    system_prompt = get_kg_query_system_prompt(context)

    return get_completion(text, context=system_prompt.strip(), use_cache=False)


def global_kg_query(text):
    prompt = get_high_low_level_keywords_prompt(text)
    result = get_completion(prompt)
    keyword_data = extract_json_from_text(result)
    logger.info(f'keyword_data: {keyword_data}')

    graph = nx.read_graphml(KG_DB)

    hl_dataset, hl_entities, hl_entity_excerpts = get_high_level_dataset(graph, keyword_data)

    entities = hl_entities
    relations = hl_dataset
    excerpts = hl_entity_excerpts

    context = get_kg_query_context(entities, excerpts, relations)
    system_prompt = get_kg_query_system_prompt(context)

    return get_completion(text, context=system_prompt.strip(), use_cache=False)


def mix_query(text):
    prompt = get_high_low_level_keywords_prompt(text)
    result = get_completion(prompt)
    keyword_data = extract_json_from_text(result)
    logger.info(f'keyword_data: {keyword_data}')

    graph = nx.read_graphml(KG_DB)

    ll_dataset, ll_entity_excerpts, ll_relations = get_low_level_dataset(graph, keyword_data)
    hl_dataset, hl_entities, hl_entity_excerpts = get_high_level_dataset(graph, keyword_data)

    kg_entities = ll_dataset + hl_entities
    kg_relations = ll_relations + hl_dataset
    kg_excerpts = ll_entity_excerpts + hl_entity_excerpts

    query_excerpts = get_query_excerpts(text)

    kg_context = get_kg_query_context(kg_entities, kg_excerpts, kg_relations)
    excerpt_context = get_excerpt_context(query_excerpts)

    system_prompt = get_mix_system_prompt(excerpt_context, kg_context)

    return get_completion(text, context=system_prompt.strip(), use_cache=False)


def get_kg_query_context(entities, excerpts, relations):
    entity_csv = [["entity", "type", "description", "rank"]]
    for entity in entities:
        entity_csv.append(
            [
                entity["entity_name"],
                entity.get("category", "UNKNOWN"),
                entity.get("description", "UNKNOWN"),
                entity["rank"],
            ]
        )
    entity_context = list_of_list_to_csv(entity_csv)
    relation_csv = [["source", "target", "description", "keywords", "weight", "rank"]]
    for relation in relations:
        relation_csv.append(
            [
                relation["src_tgt"][0],
                relation["src_tgt"][1],
                relation["description"],
                relation["keywords"],
                relation["weight"],
                relation["rank"],
            ]
        )
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
       excerpts
        {relations_context}
        ```
        -----Excerpts-----
        ```csv
        {excerpt_context}
        ```
    """)

    return context


def get_high_level_dataset(graph, keyword_data):
    hl_keywords = keyword_data.get("high_level_keywords", [])
    logger.info(f'hl_keywords: {hl_keywords}')
    if len(hl_keywords):
        hl_embedding = get_embedding(hl_keywords)
        hl_embedding_array = np.array(hl_embedding)
        hl_results = relationships_db.query(query=hl_embedding_array, top_k=25, better_than_threshold=0.02)
        logger.info(f'hl_results: {hl_results}')
    hl_data = [graph.edges.get((r["__source__"], r["__target__"])) for r in hl_results]
    hl_degrees = [graph.degree(r["__source__"]) + graph.degree(r["__target__"]) for r in hl_results]
    logger.info(f'hl_data: {hl_data}')
    logger.info(f'hl_degrees: {hl_degrees}')
    hl_dataset = []
    for k, n, d in zip(hl_results, hl_data, hl_degrees):
        hl_dataset.append({"src_tgt": (k["__source__"], k["__target__"]), "rank": d, **n, })
    logger.info(hl_dataset)
    hl_dataset = sorted(hl_dataset, key=lambda x: (x["rank"], x["weight"]), reverse=True)
    hl_dataset = truncate_list_by_token_size(
        hl_dataset,
        get_text_for_row=lambda x: x["description"],
        max_token_size=4000,
    )
    logger.info(f'hl_dataset: {hl_dataset}')
    hl_entity_excerpts = get_excerpts_for_relationships(hl_dataset)
    logger.info(f'hl_entity_excerpts {hl_entity_excerpts}')
    hl_entities = get_entities_from_relationships(graph, hl_dataset)
    logger.info(f'hl_relation_excerpts {hl_entities}')
    return hl_dataset, hl_entities, hl_entity_excerpts


def get_low_level_dataset(graph, keyword_data):
    ll_keywords = keyword_data.get("low_level_keywords", [])
    logger.info(f'll_keywords: {ll_keywords}')
    if len(ll_keywords):
        ll_embedding = get_embedding(ll_keywords)
        ll_embedding_array = np.array(ll_embedding)
        ll_results = entities_db.query(query=ll_embedding_array, top_k=25, better_than_threshold=0.02)
        logger.info(f'll_results: {ll_results}')
    ll_data = [graph.nodes.get(r["__entity_name__"]) for r in ll_results]
    ll_degrees = [graph.degree(r["__entity_name__"]) for r in ll_results]
    logger.info(f'll_degrees: {ll_degrees}')
    logger.info(f'll_data: {ll_data}')
    ll_dataset = [
        {**n, "entity_name": k["__entity_name__"], "rank": d}
        for k, n, d in zip(ll_results, ll_data, ll_degrees)
    ]
    logger.info(f'll_dataset: {ll_dataset}')
    ll_entity_excerpts = get_excerpts_for_entities(graph, ll_dataset)
    logger.info(f'll_entity_excerpts {ll_entity_excerpts}')
    ll_relations = get_relationships_from_entities(graph, ll_dataset)
    logger.info(f'll_relations {ll_relations}')
    return ll_dataset, ll_entity_excerpts, ll_relations


def get_excerpts_for_entities(graph, kg_dataset):
    excerpt_db = get_json(EXCERPT_DB)

    excerpt_ids = [split_string_by_multi_markers(row["excerpt_id"], KG_SEP) for row in kg_dataset]
    logger.info(f'excerpt_ids: {excerpt_ids}')
    all_edges = [graph.edges(row["entity_name"]) for row in kg_dataset]
    logger.info(f'edges: {all_edges}')
    sibling_names = set()
    for edge in all_edges:
        if not edge:
            continue
        sibling_names.update([e[1] for e in edge])
    logger.info(f'sibling_names: {sibling_names}')
    sibling_nodes = [graph.nodes.get(name) for name in list(sibling_names)]
    logger.info(f'sibling_nodes: {sibling_nodes}')
    sibling_excerpt_lookup = {
        k: set(split_string_by_multi_markers(v["excerpt_id"], [KG_SEP]))
        for k, v in zip(sibling_names, sibling_nodes)
        if v is not None and "excerpt_id" in v
    }
    logger.info(f'sibling_excerpt_lookup: {sibling_excerpt_lookup}')
    all_excerpt_data_lookup = {}
    for index, (excerpt_ids, edges) in enumerate(zip(excerpt_ids, all_edges)):
        logger.info(f'excerpt_ids: {excerpt_ids}, edges: {edges}')
        for excerpt_id in excerpt_ids:
            if excerpt_id in all_excerpt_data_lookup:
                continue
            relation_counts = 0
            if edges:
                for edge in edges:
                    sibling_name = edge[1]
                    if sibling_name in sibling_excerpt_lookup and excerpt_id in sibling_excerpt_lookup[sibling_name]:
                        relation_counts += 1
            logger.info(f'excerpt_id: {excerpt_id}')
            excerpt_data = excerpt_db.get(excerpt_id)
            logger.info(f'excerpt: {excerpt_data}')
            if excerpt_data is not None and "excerpt" in excerpt_data:
                all_excerpt_data_lookup[excerpt_id] = {
                    "data": excerpt_data,
                    "order": index,
                    "relation_counts": relation_counts,
                }
    logger.info(f'all_excerpts_lookup: {all_excerpt_data_lookup}')

    all_excerpts = [
        {"id": k, **v}
        for k, v in all_excerpt_data_lookup.items()
        if v is not None and v.get("data") is not None and "excerpt" in v["data"]
    ]

    logger.info(f'all_excerpts: {all_excerpts}')
    logger.info(f'len_all_excerpts: {len(all_excerpts)}')

    if not all_excerpts:
        logger.warning("No valid excerpts found")
        return []

    all_excerpts = sorted(all_excerpts, key=lambda x: (x["order"], -x["relation_counts"]))
    all_excerpts = [t["data"] for t in all_excerpts]

    all_excerpts = truncate_list_by_token_size(
        all_excerpts,
        get_text_for_row=lambda x: x["excerpt"],
        max_token_size=4000,
    )

    logger.info(f'len_all_excerpts: {len(all_excerpts)}')
    logger.info(f'final_all_excerpts: {all_excerpts}')

    return all_excerpts


def get_relationships_from_entities(graph, kg_dataset):
    node_edges_list = [graph.edges(row["entity_name"]) for row in kg_dataset]
    logger.info(f'node_edges_list: {node_edges_list}')

    edges = []
    seen = set()

    for node_edges in node_edges_list:
        for edge in node_edges:
            sorted_edge = tuple(sorted(edge))
            if sorted_edge not in seen:
                seen.add(sorted_edge)
                edges.append(sorted_edge)

    logger.info(f'edges: {edges}')

    edges_pack = [graph.edges.get((e[0], e[1])) for e in edges]
    logger.info(f'edges_pack: {edges_pack}')

    edges_degree = [graph.degree(e[0]) + graph.degree(e[1]) for e in edges]
    logger.info(f'edges_degree: {edges_degree}')

    edges_data = [
        {"src_tgt": k, "rank": d, **v}
        for k, v, d in zip(edges, edges_pack, edges_degree)
        if v is not None
    ]
    logger.info(f'edges_data: {edges_data}')
    edges_data = sorted(edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True)
    logger.info(f'sorted_edges_data: {edges_data}')
    edges_data = truncate_list_by_token_size(
        edges_data,
        get_text_for_row=lambda x: x["description"],
        max_token_size=1000,
    )
    logger.info(f'final_edges_data: {edges_data}')

    return edges_data


def get_excerpts_for_relationships(kg_dataset):
    excerpt_db = get_json(EXCERPT_DB)

    excerpt_ids = [
        split_string_by_multi_markers(dp["excerpt_id"], [KG_SEP])
        for dp in kg_dataset
    ]

    all_excerpts_lookup = {}

    for index, excerpt_ids in enumerate(excerpt_ids):
        for excerpt_id in excerpt_ids:
            if excerpt_id not in all_excerpts_lookup:
                all_excerpts_lookup[excerpt_id] = {
                    "data": excerpt_db.get(excerpt_id),
                    "order": index,
                }

    if any([v is None for v in all_excerpts_lookup.values()]):
        logger.warning("Text chunks are missing, maybe the storage is damaged")
    all_excerpts = [
        {"id": k, **v} for k, v in all_excerpts_lookup.items() if v is not None
    ]
    all_excerpts = sorted(all_excerpts, key=lambda x: x["order"])
    all_excerpts = [t["data"] for t in all_excerpts]

    all_excerpts = truncate_list_by_token_size(
        all_excerpts,
        get_text_for_row=lambda x: x["excerpt"],
        max_token_size=4000,
    )

    return all_excerpts


def get_entities_from_relationships(graph, kg_dataset):
    entity_names = []
    seen = set()


    for e in kg_dataset:
        if e["src_tgt"][0] not in seen:
            entity_names.append(e["src_tgt"][0])
            seen.add(e["src_tgt"][0])
        if e["src_tgt"][1] not in seen:
            entity_names.append(e["src_tgt"][1])
            seen.add(e["src_tgt"][1])

    data = [graph.nodes.get(entity_name) for entity_name in entity_names]
    degrees = [graph.degree(entity_name) for entity_name in entity_names]

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

    return data


if __name__ == '__main__':
    set_logger("main.log")

    delete_all_files(DATA_DIR)
    delete_all_files(LOG_DIR)

    create_file_if_not_exists(SOURCE_TO_DOC_ID_MAP, "{}")
    create_file_if_not_exists(DOC_ID_TO_SOURCE_MAP, "{}")
    create_file_if_not_exists(DOC_ID_TO_EXCERPT_IDS, "{}")
    create_file_if_not_exists(EXCERPT_DB, "{}")
    create_file_if_not_exists(QUERY_CACHE, "{}")
    create_file_if_not_exists(EMBEDDING_CACHE, "{}")

    import_documents()

    # print(query("what do rabbits eat?"))  # Should answer
    # print(query("what do cats eat?"))  # Should reject
    # print(query("What's the subject matter we can discuss?"))

    # print(kg_query("what do rabbits eat?"))  # Should answer
    # print(kg_query("what do cows eat?"))  # Should reject
    # print(kg_query("What's the subject matter we can discuss?"))

    # print(local_kg_query("what do rabbits eat?"))  # Should answer
    # print(local_kg_query("what do cows eat?"))  # Should reject
    # print(local_kg_query("What's the subject matter we can discuss?"))

    # print(global_kg_query("what do rabbits eat?"))  # Should answer
    # print(global_kg_query("what do cows eat?"))  # Should reject
    # print(global_kg_query("What's the subject matter we can discuss?"))

    print(mix_query("what do rabbits eat?"))  # Should answer
    # print(mix_query("what do cows eat?"))  # Should reject
    # print(mix_query("What's the subject matter we can discuss?"))

    # remove_document_by_id("doc_4c3f8100da0b90c1a44c94e6b4ffa041")

    # Todo: Use Jaal graph visualisation to inspect knowledge graph
