"""
Self-tests for equiv.py

Verifies:
- Partition construction from equivalence pairs
- Stable relabeling
- Refinement checking (positive and negative cases)
"""

import pytest
from equiv import new_partition_from_equiv, relabel_stable, is_refinement, get_blocks


class TestNewPartitionFromEquiv:
    """Tests for new_partition_from_equiv."""

    def test_empty_pairs(self):
        """Empty pairs produces empty partition."""
        p = new_partition_from_equiv([])
        assert p == {}

    def test_single_pair(self):
        """Single equivalence pair."""
        p = new_partition_from_equiv([((0, 0), (0, 1))])
        assert p[(0, 0)] == p[(0, 1)]

    def test_transitive_closure(self):
        """Transitive closure: if a~b and b~c, then a~c."""
        pairs = [((0, 0), (0, 1)), ((0, 1), (0, 2))]
        p = new_partition_from_equiv(pairs)

        # All three should be in same block
        assert p[(0, 0)] == p[(0, 1)] == p[(0, 2)]

    def test_multiple_blocks(self):
        """Multiple disconnected blocks."""
        pairs = [
            ((0, 0), (0, 1)),  # Block 1
            ((1, 0), (1, 1)),  # Block 2
        ]
        p = new_partition_from_equiv(pairs)

        # Within blocks
        assert p[(0, 0)] == p[(0, 1)]
        assert p[(1, 0)] == p[(1, 1)]

        # Between blocks
        assert p[(0, 0)] != p[(1, 0)]

    def test_deterministic_block_ids(self):
        """Block IDs are assigned deterministically."""
        pairs1 = [((0, 0), (0, 1)), ((1, 0), (1, 1))]
        pairs2 = [((1, 0), (1, 1)), ((0, 0), (0, 1))]

        p1 = new_partition_from_equiv(pairs1)
        p2 = new_partition_from_equiv(pairs2)

        # Same partition structure
        assert (p1[(0, 0)] == p1[(0, 1)]) == (p2[(0, 0)] == p2[(0, 1)])
        assert (p1[(1, 0)] == p1[(1, 1)]) == (p2[(1, 0)] == p2[(1, 1)])

    def test_self_equivalence(self):
        """Position equivalent to itself."""
        p = new_partition_from_equiv([((0, 0), (0, 0))])
        assert (0, 0) in p

    def test_complex_chain(self):
        """Long chain of equivalences."""
        pairs = [((0, i), (0, i + 1)) for i in range(5)]
        p = new_partition_from_equiv(pairs)

        # All positions in same block
        block_id = p[(0, 0)]
        for i in range(6):
            assert p[(0, i)] == block_id


class TestRelabelStable:
    """Tests for relabel_stable."""

    def test_empty_partition(self):
        """Empty partition stays empty."""
        assert relabel_stable({}) == {}

    def test_single_block(self):
        """Single block gets ID 0."""
        p = {(0, 0): 5, (0, 1): 5, (1, 0): 5}
        relabeled = relabel_stable(p)

        # All should have same ID
        block_id = relabeled[(0, 0)]
        assert relabeled[(0, 1)] == block_id
        assert relabeled[(1, 0)] == block_id

        # Should be 0 for single block
        assert block_id == 0

    def test_multiple_blocks_ordered(self):
        """Multiple blocks get IDs 0..k-1 in deterministic order."""
        p = {
            (0, 0): 10,
            (0, 1): 10,
            (1, 0): 20,
            (1, 1): 20,
            (2, 0): 30,
        }
        relabeled = relabel_stable(p)

        # Check IDs are in range 0..2
        ids = set(relabeled.values())
        assert ids == {0, 1, 2}

        # Check block structure preserved
        assert relabeled[(0, 0)] == relabeled[(0, 1)]
        assert relabeled[(1, 0)] == relabeled[(1, 1)]
        assert relabeled[(0, 0)] != relabeled[(1, 0)]
        assert relabeled[(0, 0)] != relabeled[(2, 0)]

    def test_deterministic_across_runs(self):
        """Same partition produces same relabeling."""
        p = {(2, 3): 100, (0, 0): 200, (1, 1): 100}

        r1 = relabel_stable(p)
        r2 = relabel_stable(p)

        assert r1 == r2

    def test_lexicographic_ordering(self):
        """Blocks ordered by lexicographically smallest position."""
        p = {
            (2, 0): 1,  # Block at (2, 0)
            (0, 5): 2,  # Block at (0, 5)
            (0, 3): 3,  # Block at (0, 3)
        }
        relabeled = relabel_stable(p)

        # (0, 3) < (0, 5) < (2, 0) lexicographically
        # So (0, 3) should get ID 0, (0, 5) ID 1, (2, 0) ID 2
        assert relabeled[(0, 3)] == 0
        assert relabeled[(0, 5)] == 1
        assert relabeled[(2, 0)] == 2

    def test_relabeling_preserves_structure(self):
        """Relabeling doesn't change partition structure."""
        pairs = [((0, 0), (0, 1)), ((1, 0), (1, 1)), ((2, 0), (2, 1))]
        p = new_partition_from_equiv(pairs)

        original_structure = {
            (p[(0, 0)] == p[(0, 1)]),
            (p[(1, 0)] == p[(1, 1)]),
            (p[(0, 0)] != p[(1, 0)]),
        }

        relabeled = relabel_stable(p)

        new_structure = {
            (relabeled[(0, 0)] == relabeled[(0, 1)]),
            (relabeled[(1, 0)] == relabeled[(1, 1)]),
            (relabeled[(0, 0)] != relabeled[(1, 0)]),
        }

        assert original_structure == new_structure


class TestIsRefinement:
    """Tests for is_refinement (CPRQ law)."""

    def test_identity_refinement(self):
        """Partition refines itself."""
        p = {(0, 0): 0, (0, 1): 1, (1, 0): 2}
        assert is_refinement(p, p)

    def test_finer_refines_coarser(self):
        """Finer partition refines coarser partition."""
        # P_input: {{(0,0)}, {(0,1)}, {(1,0)}, {(1,1)}}
        p_input = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}

        # P_label: {{(0,0), (0,1)}, {(1,0), (1,1)}}
        p_label = {(0, 0): 0, (0, 1): 0, (1, 0): 1, (1, 1): 1}

        assert is_refinement(p_input, p_label)

    def test_coarser_does_not_refine_finer(self):
        """Coarser partition does NOT refine finer partition."""
        # P_input: {{(0,0), (0,1)}, {(1,0), (1,1)}}
        p_input = {(0, 0): 0, (0, 1): 0, (1, 0): 1, (1, 1): 1}

        # P_label: {{(0,0)}, {(0,1)}, {(1,0), (1,1)}}
        p_label = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 2}

        assert not is_refinement(p_input, p_label)

    def test_any_refines_all_same_block(self):
        """Any partition refines single-block partition (coarsest)."""
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

        # P_input: any partition
        p_input = {(0, 0): 0, (0, 1): 0, (1, 0): 1, (1, 1): 1}

        # P_label: all in one block (coarsest)
        p_label = {pos: 0 for pos in positions}

        assert is_refinement(p_input, p_label)

    def test_refinement_positive_example(self):
        """Example: P_input = {{0,1}, {2,3}}, P_label = {{0,1,2,3}}."""
        p_input = {(0, 0): 0, (0, 1): 0, (1, 0): 1, (1, 1): 1}
        p_label = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}

        assert is_refinement(p_input, p_label)

    def test_refinement_negative_example(self):
        """Example: P_input = {{0,1,2,3}}, P_label = {{0,1}, {2,3}}."""
        p_input = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
        p_label = {(0, 0): 0, (0, 1): 0, (1, 0): 1, (1, 1): 1}

        assert not is_refinement(p_input, p_label)

    def test_incomparable_partitions(self):
        """Incomparable partitions don't refine each other."""
        # P1: {{(0,0), (1,0)}, {(0,1), (1,1)}}
        p1 = {(0, 0): 0, (1, 0): 0, (0, 1): 1, (1, 1): 1}

        # P2: {{(0,0), (0,1)}, {(1,0), (1,1)}}
        p2 = {(0, 0): 0, (0, 1): 0, (1, 0): 1, (1, 1): 1}

        assert not is_refinement(p1, p2)
        assert not is_refinement(p2, p1)

    def test_different_domains_not_refinement(self):
        """Partitions on different domains can't refine each other."""
        p1 = {(0, 0): 0, (0, 1): 0}
        p2 = {(0, 0): 0, (1, 0): 0}

        assert not is_refinement(p1, p2)

    def test_empty_partitions(self):
        """Empty partitions refine each other."""
        assert is_refinement({}, {})


class TestGetBlocks:
    """Tests for get_blocks helper."""

    def test_empty_partition(self):
        """Empty partition has no blocks."""
        assert get_blocks({}) == []

    def test_single_block(self):
        """Single block partition."""
        p = {(0, 0): 0, (0, 1): 0, (1, 0): 0}
        blocks = get_blocks(p)

        assert len(blocks) == 1
        assert {(0, 0), (0, 1), (1, 0)} in blocks

    def test_multiple_blocks(self):
        """Multiple blocks partition."""
        p = {(0, 0): 0, (0, 1): 0, (1, 0): 1, (1, 1): 1}
        blocks = get_blocks(p)

        assert len(blocks) == 2
        assert {(0, 0), (0, 1)} in blocks
        assert {(1, 0), (1, 1)} in blocks

    def test_singleton_blocks(self):
        """Each position in its own block."""
        p = {(0, 0): 0, (0, 1): 1, (1, 0): 2}
        blocks = get_blocks(p)

        assert len(blocks) == 3
        assert {(0, 0)} in blocks
        assert {(0, 1)} in blocks
        assert {(1, 0)} in blocks


class TestDeterminism:
    """Test determinism across multiple runs."""

    def test_partition_determinism(self):
        """Same pairs produce identical partition."""
        pairs = [((0, i), (0, i + 1)) for i in range(10)]

        p1 = new_partition_from_equiv(pairs)
        p2 = new_partition_from_equiv(pairs)

        assert p1 == p2

    def test_relabel_determinism(self):
        """Relabeling is deterministic."""
        p = {(i, j): (i * 3 + j) % 5 for i in range(3) for j in range(3)}

        r1 = relabel_stable(p)
        r2 = relabel_stable(p)

        assert r1 == r2

    def test_refinement_determinism(self):
        """is_refinement returns same result."""
        p_input = {(0, 0): 0, (0, 1): 0, (1, 0): 1}
        p_label = {(0, 0): 0, (0, 1): 1, (1, 0): 2}

        r1 = is_refinement(p_input, p_label)
        r2 = is_refinement(p_input, p_label)

        assert r1 == r2
