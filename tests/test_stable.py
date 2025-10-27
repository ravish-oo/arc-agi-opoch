"""
Self-tests for stable.py

Verifies:
- Hash determinism across runs
- Dict order invariance
- row_major_string exact format
"""

import pytest
from stable import stable_hash64, sorted_items, row_major_string


class TestStableHash64:
    """Tests for stable_hash64 determinism and canonicalization."""

    def test_same_object_same_hash(self):
        """Same object hashed twice produces identical values."""
        obj = {"a": 1, "b": [2, 3], "c": {"nested": True}}
        hash1 = stable_hash64(obj)
        hash2 = stable_hash64(obj)
        assert hash1 == hash2

    def test_dict_order_invariance(self):
        """Dict insertion order doesn't affect hash."""
        dict1 = {"a": 1, "b": 2, "c": 3}
        dict2 = {"c": 3, "a": 1, "b": 2}
        dict3 = {"b": 2, "c": 3, "a": 1}

        hash1 = stable_hash64(dict1)
        hash2 = stable_hash64(dict2)
        hash3 = stable_hash64(dict3)

        assert hash1 == hash2 == hash3

    def test_tuple_list_equivalence(self):
        """Tuples and lists with same content hash identically."""
        hash_list = stable_hash64([1, 2, 3])
        hash_tuple = stable_hash64((1, 2, 3))
        assert hash_list == hash_tuple

    def test_set_normalization(self):
        """Sets are normalized to sorted lists before hashing."""
        # Sets are unordered, but should hash consistently
        hash1 = stable_hash64({3, 1, 2})
        hash2 = stable_hash64({1, 2, 3})
        hash3 = stable_hash64({2, 3, 1})
        assert hash1 == hash2 == hash3

    def test_nested_dict_order_invariance(self):
        """Nested dict order doesn't affect hash."""
        obj1 = {"outer": {"b": 2, "a": 1}, "list": [3, 4]}
        obj2 = {"list": [3, 4], "outer": {"a": 1, "b": 2}}
        assert stable_hash64(obj1) == stable_hash64(obj2)

    def test_returns_64bit_int(self):
        """Hash returns a valid 64-bit integer."""
        h = stable_hash64({"test": 123})
        assert isinstance(h, int)
        assert 0 <= h < 2**64


class TestSortedItems:
    """Tests for sorted_items key ordering."""

    def test_basic_sorting(self):
        """Returns items sorted by key."""
        d = {"z": 1, "a": 2, "m": 3}
        result = sorted_items(d)
        assert result == [("a", 2), ("m", 3), ("z", 1)]

    def test_numeric_keys(self):
        """Works with numeric keys."""
        d = {3: "three", 1: "one", 2: "two"}
        result = sorted_items(d)
        assert result == [(1, "one"), (2, "two"), (3, "three")]

    def test_empty_dict(self):
        """Handles empty dict."""
        assert sorted_items({}) == []


class TestRowMajorString:
    """Tests for row_major_string exact format."""

    def test_locked_format_3x3_zeros(self):
        """3x3 grid of zeros produces exact format."""
        grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        result = row_major_string(grid)
        assert result == "000\n000\n000"

    def test_locked_format_2x3(self):
        """2x3 grid produces exact format."""
        grid = [[0, 1, 0], [2, 1, 2]]
        result = row_major_string(grid)
        assert result == "010\n212"

    def test_no_trailing_newline(self):
        """Result has no trailing newline."""
        grid = [[1, 2], [3, 4]]
        result = row_major_string(grid)
        assert not result.endswith('\n')
        assert result == "12\n34"

    def test_single_row(self):
        """Single row grid."""
        grid = [[1, 2, 3, 4]]
        result = row_major_string(grid)
        assert result == "1234"

    def test_single_col(self):
        """Single column grid."""
        grid = [[1], [2], [3]]
        result = row_major_string(grid)
        assert result == "1\n2\n3"

    def test_duck_typing_with_tuples(self):
        """Works with tuple of tuples."""
        grid = ((0, 1), (2, 3))
        result = row_major_string(grid)
        assert result == "01\n23"

    def test_empty_grid(self):
        """Handles empty grid."""
        assert row_major_string([]) == ""
        assert row_major_string([[]]) == ""

    def test_larger_digits(self):
        """Works with digits > 9 (for OFA usage later)."""
        grid = [[10, 11], [12, 13]]
        result = row_major_string(grid)
        assert result == "1011\n1213"
