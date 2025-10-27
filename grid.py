"""
Grid: Immutable view with safe setters and strict guards.

All grids enforce:
- Palette: values in 0..9
- Bounds: 1 ≤ H, W ≤ 30
"""

from typing import Iterator


class Grid:
    """
    Immutable grid with bounds and palette guards.

    Grids are rectangular arrays of integers in the range 0..9.
    Shape is restricted to 1 ≤ H, W ≤ 30.

    Examples:
        >>> g = Grid([[0, 1], [2, 3]])
        >>> g.H, g.W
        (2, 2)
        >>> g[0][1]
        1
        >>> list(g.positions())
        [(0, 0), (0, 1), (1, 0), (1, 1)]
    """

    def __init__(self, data: list[list[int]]):
        """
        Create a grid from a 2D list.

        Args:
            data: 2D list of integers, must be rectangular

        Raises:
            ValueError: If shape is out of bounds (not 1..30)
            ValueError: If data is not rectangular
            ValueError: If any value is not in palette 0..9
        """
        if not data or not data[0]:
            raise ValueError("Grid must be non-empty")

        self._H = len(data)
        self._W = len(data[0])

        # Validate bounds: 1 ≤ H, W ≤ 30
        if not (1 <= self._H <= 30):
            raise ValueError(f"Height {self._H} out of bounds [1, 30]")
        if not (1 <= self._W <= 30):
            raise ValueError(f"Width {self._W} out of bounds [1, 30]")

        # Validate rectangular and palette
        self._data: list[list[int]] = []
        for r, row in enumerate(data):
            if len(row) != self._W:
                raise ValueError(f"Row {r} has length {len(row)}, expected {self._W}")

            validated_row = []
            for c, val in enumerate(row):
                if not isinstance(val, int) or not (0 <= val <= 9):
                    raise ValueError(f"Value {val} at ({r}, {c}) not in palette 0..9")
                validated_row.append(val)

            self._data.append(validated_row)

    @property
    def H(self) -> int:
        """Height (number of rows)."""
        return self._H

    @property
    def W(self) -> int:
        """Width (number of columns)."""
        return self._W

    def __getitem__(self, r: int) -> 'GridRow':
        """
        Get a row for reading or writing.

        Args:
            r: Row index (0-based)

        Returns:
            GridRow object supporting [c] indexing

        Raises:
            IndexError: If row index out of bounds
        """
        if not (0 <= r < self._H):
            raise IndexError(f"Row index {r} out of bounds [0, {self._H})")
        return GridRow(self, r)

    def __len__(self) -> int:
        """
        Length of grid (number of rows).

        Returns:
            Height (same as H property)

        Note:
            Enables duck-typing with row_major_string(grid)
        """
        return self._H

    def positions(self) -> Iterator[tuple[int, int]]:
        """
        Iterate over all positions in row-major order.

        Yields:
            (r, c) tuples in row-major order (top-to-bottom, left-to-right)

        Examples:
            >>> g = Grid([[1, 2], [3, 4]])
            >>> list(g.positions())
            [(0, 0), (0, 1), (1, 0), (1, 1)]
        """
        for r in range(self._H):
            for c in range(self._W):
                yield (r, c)

    def _get_value(self, r: int, c: int) -> int:
        """Internal getter for GridRow."""
        return self._data[r][c]

    def _set_value(self, r: int, c: int, value: int) -> None:
        """
        Internal setter for GridRow with palette guard.

        Args:
            r: Row index
            c: Column index
            value: New value (must be in 0..9)

        Raises:
            ValueError: If value not in palette 0..9
        """
        if not isinstance(value, int) or not (0 <= value <= 9):
            raise ValueError(f"Value {value} not in palette 0..9")
        self._data[r][c] = value

    def __repr__(self) -> str:
        """String representation."""
        return f"Grid({self._data!r})"

    def __eq__(self, other) -> bool:
        """Equality comparison."""
        if not isinstance(other, Grid):
            return False
        return self._data == other._data


class GridRow:
    """
    Row proxy for safe __getitem__ and __setitem__ access.

    Supports grid[r][c] syntax while enforcing bounds and palette guards.
    """

    def __init__(self, grid: Grid, row: int):
        """
        Create a row proxy.

        Args:
            grid: Parent Grid instance
            row: Row index
        """
        self._grid = grid
        self._row = row

    def __getitem__(self, c: int) -> int:
        """
        Get value at column c.

        Args:
            c: Column index (0-based)

        Returns:
            Value at (row, c)

        Raises:
            IndexError: If column index out of bounds
        """
        if not (0 <= c < self._grid.W):
            raise IndexError(f"Column index {c} out of bounds [0, {self._grid.W})")
        return self._grid._get_value(self._row, c)

    def __setitem__(self, c: int, value: int) -> None:
        """
        Set value at column c with palette guard.

        Args:
            c: Column index (0-based)
            value: New value (must be in 0..9)

        Raises:
            IndexError: If column index out of bounds
            ValueError: If value not in palette 0..9
        """
        if not (0 <= c < self._grid.W):
            raise IndexError(f"Column index {c} out of bounds [0, {self._grid.W})")
        self._grid._set_value(self._row, c, value)

    def __len__(self) -> int:
        """Length of row (width of grid)."""
        return self._grid.W
