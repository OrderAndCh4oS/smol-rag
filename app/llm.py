import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

from app.logger import logger
from app.definitions import QUERY_CACHE, EMBEDDING_CACHE, COMPLETION_MODEL, EMBEDDING_MODEL
from app.utilities import add_to_json, get_json, make_hash

load_dotenv()

# logging.basicConfig(level=logging.DEBUG)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

query_cache = {}
embedding_cache = {}

def get_completion(query, model=COMPLETION_MODEL, context="", use_cache=True):
    # Todo write to and read from query cache

    global query_cache
    if not query_cache:
        query_cache = get_json(QUERY_CACHE)

    query_hash = make_hash(query, 'qry-')
    if use_cache and query_hash in query_cache:
        logger.info(f"query cache hit")
        data = query_cache[query_hash]
        result = data["result"]
    else:
        logger.info(f"new query")
        system_message = [{"role": "developer", "content": context}] if context else []
        messages = [{"role": "user", "content": query}]

        response = client.chat.completions.create(
            model=model,
            store=True,
            messages=system_message + messages
        )
        result = response.choices[0].message.content

        add_to_json(QUERY_CACHE, query_hash, {"query": query, "result": result})

    return result


def get_chat_completion(query, model=COMPLETION_MODEL, context="", chat_history=[]):
    # Todo: add cache for chat
    system_message = [{"role": "developer", "content": context}] if context else []
    messages = chat_history + [{"role": "user", "content": query}]

    response = client.chat.completions.create(
        model=model,
        store=True,
        messages=system_message + messages
    )

    return messages + [{"role": "assistant", "content": response.choices[0].message.content}]


def get_embedding(content, model=EMBEDDING_MODEL):
    global embedding_cache
    if not embedding_cache:
        embedding_cache = get_json(EMBEDDING_CACHE)

    content_hash = make_hash(str(content), 'emb-')

    if content_hash in embedding_cache:
        logger.info(f"Embedding cache hit")

        embedding = embedding_cache[content_hash]
    else:
        logger.info(f"New embedding")
        response = client.embeddings.create(
            model=model,
            input=content,
        )
        embedding = response.data[0].embedding
        add_to_json(EMBEDDING_CACHE, content_hash, embedding)


    return embedding
