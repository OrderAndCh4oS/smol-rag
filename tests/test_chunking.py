import pytest

pytest.importorskip("nltk")

from app.chunking import preserve_markdown_code_excerpts


@pytest.mark.parametrize(
    "content,n,expected",
    [
        ("This is a short paragraph.", 50, ["This is a short paragraph."]),
        (
            """```python\nprint('hi')\n```""",
            50,
            ["```python\nprint('hi')\n```"],
        ),
        (
            "First para.\n\n```js\nconsole.log('hi');\n```",
            80,
            ["First para.\n\n```js\nconsole.log('hi');\n```"],
        ),
    ],
)
def test_basic_cases(content, n, expected):
    assert preserve_markdown_code_excerpts(content, n=n) == expected


def test_paragraph_split():
    content = " ".join(["word"] * 60)  # ~300 chars
    excerpts = preserve_markdown_code_excerpts(content, n=100)
    assert len(excerpts) == 3  # should split roughly every 100 chars
    for ex in excerpts:
        assert len(ex) <= 100


def test_code_block_split():
    long_code = "```\n" + "a = 1\n" * 100 + "```"  # > 400 chars
    excerpts = preserve_markdown_code_excerpts(long_code, n=120)
    # code fences preserved in every chunk
    for ex in excerpts:
        assert ex.startswith("```") and ex.endswith("```")
        assert len(ex) <= 120


def test_overlap():
    text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 10
    ex_no_overlap = preserve_markdown_code_excerpts(text, n=120)
    ex_overlap = preserve_markdown_code_excerpts(text, n=120, overlap=10)
    assert len(ex_no_overlap) == len(ex_overlap)
    # each overlapped excerpt (except the first) should start with the last 10 chars of previous
    for prev, curr in zip(ex_overlap, ex_overlap[1:]):
        assert prev[-10:] == curr[:10]


def test_extremely_long_sentence():
    sentence = "a" * 300
    excerpts = preserve_markdown_code_excerpts(sentence, n=100)
    # function should hard-split the sentence
    assert all(len(ex) <= 100 for ex in excerpts)


def test_buffer_flush_on_blank_lines():
    content = "para1\n\n\n\npara2"
    excerpts = preserve_markdown_code_excerpts(content, n=20)
    assert excerpts == ["para1", "para2"]
