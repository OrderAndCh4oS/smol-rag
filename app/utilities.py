import csv
import html
import io
import json
import os
import pathlib
import re
from hashlib import md5
from typing import List

import tiktoken

from app.definitions import COMPLETION_MODEL

tiktoken_encoders = {}

def create_file_if_not_exists(file_path, default_content=""):
    if not os.path.exists(file_path):
        write_file(file_path, default_content)


def write_file(file_path, content):
    with open(file_path, "w") as f:
        f.write(content)


def read_file(file_path):
    f = open(file_path, "r")
    return f.read()


def delete_all_files(directory):
    p = pathlib.Path(directory)
    if not p.exists():
        print(f"Directory '{directory}' does not exist.")
        return
    for item in p.iterdir():
        if item.is_file() and not item.is_symlink():
            item.unlink()


def get_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def write_json(file_path, data):
    with open(file_path, "w") as f:
        return json.dump(data, f, indent=4)


def get_docs(root_dir):
    text_files = []
    for path, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(('.txt', '.md', '.mdx', '.yaml', '.yml', '.tex', '.rst')):
                text_files.append(os.path.join(path, filename))
    return text_files


def make_hash(text, prefix=""):
    return prefix + md5(text.encode()).hexdigest()


def add_to_json(file_path, key, value):
    with open(file_path, "r+") as f:
        data = json.load(f)
        data[key] = value
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()


def remove_from_json(file_path, key):
    with open(file_path, "r+") as f:
        data = json.load(f)
        if key not in data:
            return
        del data[key]
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()


# Refer the utils functions of the official GraphRAG implementation:
# https://github.com/microsoft/graphrag
def clean_str(text: str) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    assert isinstance(text, str)

    result = html.unescape(text.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers."""
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


def extract_json_from_text(content: str):
    json_str = re.search(r"{.*}", content, re.DOTALL)
    if json_str is not None:
        json_str = json_str.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    return None

def truncate_list_by_token_size(data_list, get_text_for_row, max_token_size=4000, model=COMPLETION_MODEL):
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(data_list):
        tokens += len(get_encoded_tokens(get_text_for_row(data), model))
        if tokens >= max_token_size:
            return data_list[:i]

    return data_list

def get_encoded_tokens(text, model=COMPLETION_MODEL):
    global tiktoken_encoders
    if not model in tiktoken_encoders:
        tiktoken_encoders[model] = tiktoken.encoding_for_model(model)

    return tiktoken_encoders[model].encode(text)

def is_float_regex(value):
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))

def list_of_list_to_csv(data: List[List[str]]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerows(data)
    return output.getvalue()