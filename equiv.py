"""
Partitions and refinement checking.

Partitions represent equivalence relations over grid positions.
A partition is a dict mapping (r, c) -> block_id, where positions
with the same block_id are in the same equivalence class.
"""

from typing import Iterator
from stable import stable_hash64


def new_partition_from_equiv(pairs: list[tuple[tuple[int, int], tuple[int, int]]]) -> dict[tuple[int, int], int]:
    """
    Build a partition from equivalence pairs using union-find.

    Args:
        pairs: List of ((r1, c1), (r2, c2)) pairs indicating equivalences

    Returns:
        Partition dict mapping (r, c) -> block_id

    Examples:
        >>> p = new_partition_from_equiv([((0,0), (0,1)), ((0,1), (0,2))])
        >>> p[(0,0)] == p[(0,1)] == p[(0,2)]
        True
        >>> p[(1,0)] != p[(0,0)]
        True
    """
    parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(pos: tuple[int, int]) -> tuple[int, int]:
        """Find root with path compression."""
        if pos not in parent:
            parent[pos] = pos
            return pos

        if parent[pos] != pos:
            parent[pos] = find(parent[pos])

        return parent[pos]

    def union(p1: tuple[int, int], p2: tuple[int, int]) -> None:
        """Union two positions."""
        root1 = find(p1)
        root2 = find(p2)

        if root1 != root2:
            # Use lexicographic order for determinism
            if root1 < root2:
                parent[root2] = root1
            else:
                parent[root1] = root2

    # Process all equivalence pairs
    for p1, p2 in pairs:
        union(p1, p2)

    # Build partition: map each position to its root
    partition: dict[tuple[int, int], int] = {}
    root_to_id: dict[tuple[int, int], int] = {}
    next_id = 0

    # Collect all positions
    positions = set()
    for p1, p2 in pairs:
        positions.add(p1)
        positions.add(p2)

    # Assign block IDs deterministically
    for pos in sorted(positions):
        root = find(pos)
        if root not in root_to_id:
            root_to_id[root] = next_id
            next_id += 1
        partition[pos] = root_to_id[root]

    return partition


def relabel_stable(P: dict[tuple[int, int], int]) -> dict[tuple[int, int], int]:
    """
    Relabel partition blocks with stable 0..k-1 IDs.

    Block IDs are assigned in deterministic order based on the
    lexicographically smallest position in each block.

    Args:
        P: Input partition

    Returns:
        Partition with block IDs relabeled to 0..k-1

    Examples:
        >>> p = {(0,0): 5, (0,1): 5, (1,0): 3}
        >>> relabeled = relabel_stable(p)
        >>> relabeled[(0,0)] == relabeled[(0,1)]
        True
        >>> relabeled[(0,0)] != relabeled[(1,0)]
        True
        >>> min(relabeled.values())
        0
    """
    if not P:
        return {}

    # Group positions by block
    blocks: dict[int, list[tuple[int, int]]] = {}
    for pos, block_id in P.items():
        if block_id not in blocks:
            blocks[block_id] = []
        blocks[block_id].append(pos)

    # Sort blocks by their lexicographically smallest position
    # This ensures deterministic ordering
    block_representatives: list[tuple[tuple[int, int], int]] = []
    for block_id, positions in blocks.items():
        min_pos = min(positions)
        block_representatives.append((min_pos, block_id))

    block_representatives.sort()

    # Build mapping from old block_id to new stable block_id
    old_to_new: dict[int, int] = {}
    for new_id, (_, old_id) in enumerate(block_representatives):
        old_to_new[old_id] = new_id

    # Relabel
    return {pos: old_to_new[block_id] for pos, block_id in P.items()}


def is_refinement(P_input: dict[tuple[int, int], int],
                  P_label: dict[tuple[int, int], int]) -> bool:
    """
    Check if P_input refines P_label.

    P_input refines P_label if every block of P_label is a union of
    blocks of P_input. Equivalently: if two positions are in the same
    block in P_input, they must be in the same block in P_label.

    This is the CPRQ law: label-required partitions must be expressible
    by the input-only present.

    Args:
        P_input: Partition from input-only relations (finer)
        P_label: Partition required by labels (coarser or equal)

    Returns:
        True if P_input refines P_label (i.e., P_input is finer/equal)

    Examples:
        >>> # P_input = {{0,1}, {2,3}}, P_label = {{0,1,2,3}}
        >>> p_in = {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): 1}
        >>> p_lab = {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 0}
        >>> is_refinement(p_in, p_lab)
        True

        >>> # P_input = {{0,1,2,3}}, P_label = {{0,1}, {2,3}}
        >>> p_in = {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 0}
        >>> p_lab = {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): 1}
        >>> is_refinement(p_in, p_lab)
        False
    """
    # Domains must match
    if set(P_input.keys()) != set(P_label.keys()):
        return False

    # For each block in P_input, check all positions map to same block in P_label
    # Build blocks of P_input
    input_blocks: dict[int, list[tuple[int, int]]] = {}
    for pos, block_id in P_input.items():
        if block_id not in input_blocks:
            input_blocks[block_id] = []
        input_blocks[block_id].append(pos)

    # Check each block of P_input
    for positions in input_blocks.values():
        if not positions:
            continue

        # All positions in this P_input block must have same P_label block
        first_label = P_label[positions[0]]
        for pos in positions[1:]:
            if P_label[pos] != first_label:
                return False

    return True


def get_blocks(P: dict[tuple[int, int], int]) -> list[set[tuple[int, int]]]:
    """
    Convert partition to list of blocks (sets of positions).

    Args:
        P: Partition dict

    Returns:
        List of sets, each containing positions in the same block

    Examples:
        >>> p = {(0,0): 0, (0,1): 0, (1,0): 1}
        >>> blocks = get_blocks(p)
        >>> len(blocks)
        2
        >>> {(0,0), (0,1)} in blocks
        True
    """
    blocks_dict: dict[int, set[tuple[int, int]]] = {}
    for pos, block_id in P.items():
        if block_id not in blocks_dict:
            blocks_dict[block_id] = set()
        blocks_dict[block_id].add(pos)

    return list(blocks_dict.values())
