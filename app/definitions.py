import os

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
COMPLETION_MODEL = os.getenv('COMPLETION_MODEL', 'gpt-4o-mini')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
TEST_SET_DIR = os.path.join(ROOT_DIR, "evaluation/test_sets")
CACHE_DIR = os.path.join(ROOT_DIR, "cache")
LOG_DIR = os.path.join(ROOT_DIR, "logs")
INPUT_DOCS_DIR = os.path.join(ROOT_DIR, "input_docs")

SOURCE_TO_DOC_ID_KV_PATH = os.path.join(DATA_DIR, "source_to_doc_id_map.json")
DOC_ID_TO_SOURCE_KV_PATH = os.path.join(DATA_DIR, "doc_id_to_source_map.json")
DOC_ID_TO_EXCERPT_KV_PATH = os.path.join(DATA_DIR, "doc_id_to_excerpt_ids.json")

EXCERPT_KV_PATH = os.path.join(DATA_DIR, "excerpt_db.json")
EMBEDDINGS_DB = os.path.join(DATA_DIR, "embeddings_db.json")
ENTITIES_DB = os.path.join(DATA_DIR, "entities_db.json")
RELATIONSHIPS_DB = os.path.join(DATA_DIR, "relationships_db.json")

KG_DB = os.path.join(DATA_DIR, "kg_db.graphml")

EVALUATION_DATA_SET = os.path.join(TEST_SET_DIR, "evaluation_data_set.json")

QUERY_CACHE_KV_PATH = os.path.join(CACHE_DIR, "query_cache.json")
EMBEDDING_CACHE_KV_PATH = os.path.join(CACHE_DIR, "embedding_cache.json")

KG_SEP = ":|:"
TUPLE_SEP = "<|>"
REC_SEP = "+|+"
COMPLETE_TAG = "<|COMPLETE|>"
