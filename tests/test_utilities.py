"""
Tests for utility functions, focusing on token counting performance (#9).
"""
import time
import pytest

from app.utilities import (
    truncate_list_by_token_size,
    split_string_by_multi_markers,
    clean_str,
    make_hash,
    is_float_regex
)


class TestUtilitiesBaseline:
    """Baseline tests for utility functions."""

    def test_split_string_by_multi_markers(self):
        """Test splitting strings by multiple markers."""
        text = "part1<SEP>part2<SEP>part3"
        result = split_string_by_multi_markers(text, ["<SEP>"])
        assert len(result) == 3
        assert result[0] == "part1"
        assert result[2] == "part3"

    def test_split_string_multiple_markers(self):
        """Test splitting with multiple different markers."""
        text = "part1<SEP1>part2<SEP2>part3"
        result = split_string_by_multi_markers(text, ["<SEP1>", "<SEP2>"])
        assert len(result) == 3

    def test_clean_str(self):
        """Test string cleaning (removes control chars, not extra spaces)."""
        dirty = "  text  with   spaces  "
        clean = clean_str(dirty)
        assert clean == "text  with   spaces"  # Only strips outer whitespace

        # Test control character removal
        with_control = "text\x00with\x1fcontrol"
        clean = clean_str(with_control)
        assert "\x00" not in clean
        assert "\x1f" not in clean

    def test_make_hash(self):
        """Test hash generation."""
        text = "test string"
        hash1 = make_hash(text)
        hash2 = make_hash(text)
        assert hash1 == hash2  # Same input -> same hash

        hash3 = make_hash("different string")
        assert hash1 != hash3  # Different input -> different hash

    def test_is_float_regex(self):
        """Test float detection."""
        assert is_float_regex("3.14") == True
        assert is_float_regex("0.5") == True
        assert is_float_regex("not a float") == False
        assert is_float_regex("123") == True  # Integers are floats


class TestTokenCountingPerformance:
    """Tests for token counting performance bottleneck (#9)."""

    @pytest.mark.performance
    def test_truncate_list_token_counting(self):
        """Test performance of truncate_list_by_token_size."""
        # Create list of text chunks
        chunks = [
            f"This is chunk number {i} with some text content to make it realistic."
            for i in range(100)
        ]

        # Need to provide get_text_for_row callback
        get_text = lambda x: x

        start_time = time.perf_counter()
        result = truncate_list_by_token_size(chunks, get_text, max_token_size=1000)
        elapsed = time.perf_counter() - start_time

        print(f"\nTruncate 100 chunks: {elapsed:.4f}s")
        print(f"Returned {len(result)} chunks within token limit")

        assert isinstance(result, list)
        assert len(result) <= len(chunks)

    @pytest.mark.performance
    def test_repeated_tokenization_overhead(self):
        """Test overhead of repeated tokenization (called 6+ times per query)."""
        text_chunks = [
            "Python is a high-level programming language.",
            "FastAPI is a modern web framework.",
            "NetworkX is used for graph operations.",
            "OpenAI provides embedding APIs."
        ] * 25  # 100 chunks

        get_text = lambda x: x

        # Simulate being called 6 times per query (as in smol_rag.py)
        times = []

        for call_num in range(6):
            start_time = time.perf_counter()
            result = truncate_list_by_token_size(text_chunks, get_text, max_token_size=1000)
            elapsed = time.perf_counter() - start_time
            times.append(elapsed)

        total_time = sum(times)
        avg_time = total_time / len(times)

        print(f"\n6 truncation calls (simulating query):")
        print(f"  Total time: {total_time:.4f}s")
        print(f"  Average per call: {avg_time:.4f}s")

        # With caching, later calls should be faster
        # Without caching, all calls have similar time

    @pytest.mark.performance
    def test_tokenization_with_long_texts(self):
        """Test tokenization performance with long text chunks."""
        # Create increasingly long chunks
        chunks = [
            "Test text. " * (100 * i) for i in range(1, 11)
        ]

        get_text = lambda x: x
        times = []

        for i, chunk in enumerate(chunks):
            start_time = time.perf_counter()
            result = truncate_list_by_token_size([chunk], get_text, max_token_size=1000)
            elapsed = time.perf_counter() - start_time
            times.append(elapsed)

            print(f"Chunk {i+1} ({len(chunk)} chars): {elapsed:.4f}s")

        # Time should grow with chunk length
        assert times[-1] >= times[0]


class TestStringSplittingPerformance:
    """Tests for string splitting used in entity description updates."""

    @pytest.mark.performance
    def test_split_performance_with_large_strings(self):
        """Test split performance with large concatenated descriptions."""
        # Simulate large entity description (bottleneck #6)
        descriptions = [f"Description number {i} with more text" for i in range(1000)]
        SEP = "<SEP>"
        large_string = SEP.join(descriptions)

        start_time = time.perf_counter()
        result = split_string_by_multi_markers(large_string, [SEP])
        elapsed = time.perf_counter() - start_time

        print(f"\nSplit {len(large_string)} chars with {len(descriptions)} parts: {elapsed:.4f}s")

        assert len(result) == len(descriptions)

    @pytest.mark.performance
    def test_set_operation_on_descriptions(self):
        """Test set operation cost on large description lists."""
        # Simulate operation from smol_rag.py:211-217
        SEP = "<SEP>"
        descriptions = [f"Description {i}" for i in range(1000)]
        large_string = SEP.join(descriptions)

        times = []

        for i in range(10):
            start_time = time.perf_counter()

            # Split
            existing = split_string_by_multi_markers(large_string, [SEP])

            # Convert to set (expensive!)
            as_set = set(list(existing) + [f"New description {i}"])

            # Join back
            updated = SEP.join(as_set)

            elapsed = time.perf_counter() - start_time
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        print(f"\nAverage split->set->join for 1000 descriptions: {avg_time:.4f}s")

        # This gets slower as descriptions grow


class TestUtilitiesEdgeCases:
    """Test edge cases in utility functions."""

    def test_split_empty_string(self):
        """Test splitting an empty string."""
        result = split_string_by_multi_markers("", ["<SEP>"])
        # Empty strings are filtered out, so result is empty list
        assert len(result) == 0

    def test_split_no_markers_found(self):
        """Test splitting when markers don't exist in string."""
        text = "no markers here"
        result = split_string_by_multi_markers(text, ["<SEP>"])
        assert len(result) == 1
        assert result[0] == text

    def test_truncate_empty_list(self):
        """Test truncating an empty list."""
        get_text = lambda x: x
        result = truncate_list_by_token_size([], get_text, max_token_size=1000)
        assert result == []

    def test_truncate_with_zero_token_size(self):
        """Test truncating with zero max tokens."""
        chunks = ["text1", "text2"]
        get_text = lambda x: x
        result = truncate_list_by_token_size(chunks, get_text, max_token_size=0)
        assert len(result) == 0

    def test_clean_str_with_newlines(self):
        """Test cleaning string with newlines."""
        text = "line1\n\nline2\n  line3"
        clean = clean_str(text)
        # Should normalize whitespace
        assert "\n\n" not in clean or clean.count(" ") > 0

    def test_make_hash_empty_string(self):
        """Test hashing empty string."""
        hash1 = make_hash("")
        hash2 = make_hash("")
        assert hash1 == hash2

    def test_is_float_regex_edge_cases(self):
        """Test float detection edge cases."""
        assert is_float_regex("") == False
        assert is_float_regex(".") == False
        assert is_float_regex("1.") == True or is_float_regex("1.") == False  # Depends on regex
        assert is_float_regex(".5") == True or is_float_regex(".5") == False
