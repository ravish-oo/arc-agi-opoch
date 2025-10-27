"""
Stable primitives for deterministic hashing and ordering.

All functions are platform-independent and produce identical results
across runs. No use of Python's hash(), random, or datetime.
"""

import hashlib
import json
from typing import Any


def _normalize_for_json(obj: Any) -> Any:
    """
    Recursively normalize Python objects for canonical JSON serialization.

    - tuple → list
    - set → sorted list
    - dict → dict (keys will be sorted by json.dumps)
    - recursively apply to nested structures
    """
    if isinstance(obj, dict):
        return {k: _normalize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_normalize_for_json(item) for item in obj]
    elif isinstance(obj, set):
        return sorted(_normalize_for_json(item) for item in obj)
    else:
        return obj


def stable_hash64(obj: Any) -> int:
    """
    Compute a deterministic 64-bit hash of any JSON-serializable object.

    Uses SHA-256 with canonical JSON serialization (sorted keys, no whitespace).
    Returns an integer derived from the first 16 hex digits of the hash.

    Args:
        obj: Any JSON-serializable Python object (dict, list, tuple, set, etc.)

    Returns:
        64-bit integer hash (0 to 2^64-1)

    Examples:
        >>> stable_hash64({"a": 1, "b": 2}) == stable_hash64({"b": 2, "a": 1})
        True
        >>> stable_hash64([1, 2, 3]) == stable_hash64((1, 2, 3))
        True
    """
    normalized = _normalize_for_json(obj)
    canonical_json = json.dumps(normalized, sort_keys=True, separators=(',', ':'))
    hash_bytes = hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()
    return int(hash_bytes[:16], 16)


def sorted_items(d: dict) -> list[tuple]:
    """
    Return dictionary items as a list of (key, value) tuples, sorted by key.

    Args:
        d: Dictionary with sortable keys

    Returns:
        List of (key, value) tuples in key-sorted order

    Examples:
        >>> sorted_items({"z": 1, "a": 2, "m": 3})
        [('a', 2), ('m', 3), ('z', 1)]
    """
    return sorted(d.items())


def row_major_string(grid_like) -> str:
    """
    Convert a 2D grid to a canonical string representation.

    Rows are concatenated with newlines, digits only (no spaces).
    No trailing newline.

    Args:
        grid_like: Any object supporting len(G), len(G[0]), and G[r][c]
                   Typically list[list[int]] or tuple[tuple[int]]

    Returns:
        String with format "000\\n000\\n000" for a 3x3 grid of zeros

    Examples:
        >>> row_major_string([[0, 1, 0], [2, 1, 2]])
        '010\\n212'
        >>> row_major_string([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        '000\\n000\\n000'
    """
    if not grid_like or not grid_like[0]:
        return ""

    rows = []
    for r in range(len(grid_like)):
        row_str = ''.join(str(grid_like[r][c]) for c in range(len(grid_like[0])))
        rows.append(row_str)

    return '\n'.join(rows)
