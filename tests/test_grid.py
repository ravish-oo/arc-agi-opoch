"""
Self-tests for grid.py

Verifies:
- Palette guards (0..9)
- Bounds guards (1..30 for H and W)
- Set/get round-trip
- positions() row-major iteration
"""

import pytest
from grid import Grid


class TestGridConstruction:
    """Tests for Grid construction and validation."""

    def test_valid_grid_2x2(self):
        """Valid 2x2 grid constructs successfully."""
        g = Grid([[0, 1], [2, 3]])
        assert g.H == 2
        assert g.W == 2

    def test_valid_grid_1x1_min(self):
        """Minimum 1x1 grid is valid."""
        g = Grid([[5]])
        assert g.H == 1
        assert g.W == 1

    def test_valid_grid_30x30_max(self):
        """Maximum 30x30 grid is valid."""
        data = [[0] * 30 for _ in range(30)]
        g = Grid(data)
        assert g.H == 30
        assert g.W == 30

    def test_palette_all_values(self):
        """All palette values 0..9 are valid."""
        data = [[i] for i in range(10)]
        g = Grid(data)
        assert g.H == 10
        assert g.W == 1

    def test_empty_grid_rejected(self):
        """Empty grid raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            Grid([])

    def test_empty_rows_rejected(self):
        """Grid with empty rows raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            Grid([[]])

    def test_height_too_large(self):
        """Height > 30 raises ValueError."""
        data = [[0]] * 31
        with pytest.raises(ValueError, match="Height 31 out of bounds"):
            Grid(data)

    def test_width_too_large(self):
        """Width > 30 raises ValueError."""
        data = [[0] * 31]
        with pytest.raises(ValueError, match="Width 31 out of bounds"):
            Grid(data)

    def test_non_rectangular_rejected(self):
        """Non-rectangular data raises ValueError."""
        data = [[1, 2], [3, 4, 5]]
        with pytest.raises(ValueError, match="expected 2"):
            Grid(data)

    def test_palette_negative_rejected(self):
        """Negative values rejected."""
        with pytest.raises(ValueError, match="not in palette"):
            Grid([[-1, 0]])

    def test_palette_too_large_rejected(self):
        """Values > 9 rejected."""
        with pytest.raises(ValueError, match="not in palette"):
            Grid([[0, 10]])

    def test_palette_float_rejected(self):
        """Float values rejected."""
        with pytest.raises(ValueError, match="not in palette"):
            Grid([[0, 1.5]])


class TestGridAccess:
    """Tests for __getitem__ and __setitem__."""

    def test_getitem_basic(self):
        """Basic __getitem__ access works."""
        g = Grid([[1, 2], [3, 4]])
        assert g[0][0] == 1
        assert g[0][1] == 2
        assert g[1][0] == 3
        assert g[1][1] == 4

    def test_setitem_basic(self):
        """Basic __setitem__ access works."""
        g = Grid([[0, 0], [0, 0]])
        g[0][0] = 5
        g[1][1] = 9
        assert g[0][0] == 5
        assert g[1][1] == 9

    def test_set_get_round_trip(self):
        """Values can be set and retrieved."""
        g = Grid([[0] * 3 for _ in range(3)])

        # Set all values
        for r in range(3):
            for c in range(3):
                g[r][c] = r * 3 + c

        # Verify all values
        for r in range(3):
            for c in range(3):
                expected = r * 3 + c
                assert g[r][c] == expected

    def test_row_out_of_bounds_negative(self):
        """Negative row index raises IndexError."""
        g = Grid([[1, 2]])
        with pytest.raises(IndexError, match="Row index -1"):
            _ = g[-1]

    def test_row_out_of_bounds_too_large(self):
        """Row index >= H raises IndexError."""
        g = Grid([[1, 2]])
        with pytest.raises(IndexError, match="Row index 1"):
            _ = g[1]

    def test_col_out_of_bounds_negative(self):
        """Negative column index raises IndexError."""
        g = Grid([[1, 2]])
        with pytest.raises(IndexError, match="Column index -1"):
            _ = g[0][-1]

    def test_col_out_of_bounds_too_large(self):
        """Column index >= W raises IndexError."""
        g = Grid([[1, 2]])
        with pytest.raises(IndexError, match="Column index 2"):
            _ = g[0][2]

    def test_setitem_palette_guard(self):
        """Setting value outside palette raises ValueError."""
        g = Grid([[0, 0]])

        with pytest.raises(ValueError, match="not in palette"):
            g[0][0] = 10

        with pytest.raises(ValueError, match="not in palette"):
            g[0][0] = -1

    def test_setitem_bounds_guard(self):
        """Setting out of bounds raises IndexError."""
        g = Grid([[0, 0]])

        with pytest.raises(IndexError):
            g[1][0] = 5

        with pytest.raises(IndexError):
            g[0][2] = 5


class TestPositions:
    """Tests for positions() iterator."""

    def test_positions_2x2(self):
        """positions() yields row-major order for 2x2."""
        g = Grid([[1, 2], [3, 4]])
        positions = list(g.positions())
        expected = [(0, 0), (0, 1), (1, 0), (1, 1)]
        assert positions == expected

    def test_positions_1x1(self):
        """positions() works for 1x1 grid."""
        g = Grid([[5]])
        positions = list(g.positions())
        assert positions == [(0, 0)]

    def test_positions_3x4(self):
        """positions() row-major for 3x4 grid."""
        g = Grid([[0] * 4 for _ in range(3)])
        positions = list(g.positions())

        # Verify count
        assert len(positions) == 12

        # Verify row-major order
        expected = []
        for r in range(3):
            for c in range(4):
                expected.append((r, c))

        assert positions == expected

    def test_positions_values_match(self):
        """positions() can be used to access all values."""
        g = Grid([[1, 2, 3], [4, 5, 6]])

        values = [g[r][c] for r, c in g.positions()]
        assert values == [1, 2, 3, 4, 5, 6]


class TestGridEquality:
    """Tests for Grid equality."""

    def test_equal_grids(self):
        """Grids with same data are equal."""
        g1 = Grid([[1, 2], [3, 4]])
        g2 = Grid([[1, 2], [3, 4]])
        assert g1 == g2

    def test_different_grids(self):
        """Grids with different data are not equal."""
        g1 = Grid([[1, 2]])
        g2 = Grid([[3, 4]])
        assert g1 != g2

    def test_different_shapes(self):
        """Grids with different shapes are not equal."""
        g1 = Grid([[1]])
        g2 = Grid([[1, 2]])
        assert g1 != g2


class TestDuckTyping:
    """Tests for duck-typing compatibility with stable.row_major_string."""

    def test_duck_typing_len(self):
        """Grid supports len(grid)."""
        g = Grid([[1, 2], [3, 4]])
        # len(g) should give height
        # This is for compatibility with row_major_string which uses len(G)
        assert g.H == 2

    def test_duck_typing_indexing(self):
        """Grid supports grid[r][c] indexing."""
        g = Grid([[1, 2], [3, 4]])
        assert g[0][0] == 1
        assert g[1][1] == 4

    def test_duck_typing_row_length(self):
        """Grid row supports len(grid[0])."""
        g = Grid([[1, 2], [3, 4]])
        assert len(g[0]) == 2

    def test_with_row_major_string(self):
        """Grid works with row_major_string from stable.py."""
        from stable import row_major_string

        g = Grid([[0, 1, 0], [2, 1, 2]])
        result = row_major_string(g)
        assert result == "010\n212"
