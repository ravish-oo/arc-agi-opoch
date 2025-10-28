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


# Π Canonicalization: D8 + Transpose transforms for input-only orientation normalization

def _grid_to_canonical_string(g: Grid) -> str:
    """
    Convert grid to canonical string for lexicographic comparison.

    Row-major order: concatenate all rows into a single string.

    Args:
        g: Input grid

    Returns:
        String representation for lex-min comparison
    """
    chars = []
    for r in range(g.H):
        for c in range(g.W):
            chars.append(str(g[r][c]))
    return ''.join(chars)


def rot90(g: Grid) -> Grid:
    """Rotate 90° clockwise: new[r][c] = old[H-1-c][r]. Result is W×H."""
    H, W = g.H, g.W
    # After rotation: W rows, H columns
    data = [[g[H - 1 - c][r] for c in range(H)] for r in range(W)]
    return Grid(data)


def rot180(g: Grid) -> Grid:
    """Rotate 180°: new[H-1-r][W-1-c] = old[r][c]"""
    H, W = g.H, g.W
    data = [[g[H - 1 - r][W - 1 - c] for c in range(W)] for r in range(H)]
    return Grid(data)


def rot270(g: Grid) -> Grid:
    """Rotate 270° clockwise (90° counter-clockwise): new[r][c] = old[c][W-1-r]. Result is W×H."""
    H, W = g.H, g.W
    # After rotation: W rows, H columns
    data = [[g[c][W - 1 - r] for c in range(H)] for r in range(W)]
    return Grid(data)


def flip_h(g: Grid) -> Grid:
    """Horizontal flip: new[r][W-1-c] = old[r][c]"""
    H, W = g.H, g.W
    data = [[g[r][W - 1 - c] for c in range(W)] for r in range(H)]
    return Grid(data)


def flip_v(g: Grid) -> Grid:
    """Vertical flip: new[H-1-r][c] = old[r][c]"""
    H, W = g.H, g.W
    data = [[g[H - 1 - r][c] for c in range(W)] for r in range(H)]
    return Grid(data)


def flip_d1(g: Grid) -> Grid:
    """Diagonal flip (main diagonal): new[c][r] = old[r][c] (transpose)"""
    if g.H != g.W:
        raise ValueError(f"flip_d1 requires square grid, got {g.H}×{g.W}")
    N = g.H
    data = [[g[c][r] for c in range(N)] for r in range(N)]
    return Grid(data)


def flip_d2(g: Grid) -> Grid:
    """Diagonal flip (anti-diagonal): new[W-1-c][H-1-r] = old[r][c]"""
    if g.H != g.W:
        raise ValueError(f"flip_d2 requires square grid, got {g.H}×{g.W}")
    N = g.H
    data = [[g[N - 1 - c][N - 1 - r] for c in range(N)] for r in range(N)]
    return Grid(data)


def transpose(g: Grid) -> Grid:
    """Transpose: new[r][c] = old[c][r]. Result is W×H."""
    H, W = g.H, g.W
    # After transpose: W rows, H columns
    data = [[g[c][r] for c in range(H)] for r in range(W)]
    return Grid(data)


# Inverse transform mappings
_INVERSE_TRANSFORM = {
    'identity': 'identity',
    'rot90': 'rot270',
    'rot180': 'rot180',
    'rot270': 'rot90',
    'flip_h': 'flip_h',
    'flip_v': 'flip_v',
    'flip_d1': 'flip_d1',
    'flip_d2': 'flip_d2',
    'transpose': 'transpose',
}


def get_inverse_transform(tag: str) -> str:
    """
    Get the inverse of a transform tag.

    Args:
        tag: Transform name

    Returns:
        Inverse transform name

    Examples:
        >>> get_inverse_transform('rot90')
        'rot270'
        >>> get_inverse_transform('flip_h')
        'flip_h'
    """
    if tag not in _INVERSE_TRANSFORM:
        raise ValueError(f"Unknown transform tag: {tag}")
    return _INVERSE_TRANSFORM[tag]


def apply_transform(g: Grid, tag: str) -> Grid:
    """
    Apply a transform by tag name.

    Args:
        g: Input grid
        tag: Transform name (e.g., 'rot90', 'flip_h', 'identity')

    Returns:
        Transformed grid
    """
    if tag == 'identity':
        return g
    elif tag == 'rot90':
        return rot90(g)
    elif tag == 'rot180':
        return rot180(g)
    elif tag == 'rot270':
        return rot270(g)
    elif tag == 'flip_h':
        return flip_h(g)
    elif tag == 'flip_v':
        return flip_v(g)
    elif tag == 'flip_d1':
        return flip_d1(g)
    elif tag == 'flip_d2':
        return flip_d2(g)
    elif tag == 'transpose':
        return transpose(g)
    else:
        raise ValueError(f"Unknown transform tag: {tag}")


def pi_canon(X: Grid) -> tuple[Grid, str, str]:
    """
    Π canonicalization: choose lexicographically smallest orientation.

    Per math_spec.md: "Orientation (Π): dihedral (rot/flip) + transpose
    (only if shape-compatible); choose lex-min image on X; apply same
    transform to Y on train; on test, apply to X*, remember inverse."

    Tries all D8 transforms (8 dihedral symmetries) plus transpose (if square).
    Returns the transform that produces the lexicographically smallest
    row-major string representation.

    Properties:
        - Idempotent: pi_canon(pi_canon(X)[0])[0] == pi_canon(X)[0]
        - Deterministic: Same input always produces same canonical form

    Args:
        X: Input grid

    Returns:
        (X_canonical, pi_tag, pi_inverse):
            - X_canonical: Transformed grid (lex-min)
            - pi_tag: Name of transform applied
            - pi_inverse: Name of inverse transform

    Examples:
        >>> g = Grid([[3, 1], [2, 0]])
        >>> gc, tag, inv = pi_canon(g)
        >>> # gc is lexicographically smallest orientation
        >>> tag in ['identity', 'rot90', 'rot180', 'rot270', 'flip_h', 'flip_v', 'flip_d1', 'flip_d2', 'transpose']
        True
    """
    # D8 transforms (always applicable)
    transforms = [
        ('identity', X),
        ('rot90', rot90(X)),
        ('rot180', rot180(X)),
        ('rot270', rot270(X)),
        ('flip_h', flip_h(X)),
        ('flip_v', flip_v(X)),
    ]

    # Square-only transforms
    if X.H == X.W:
        transforms.append(('flip_d1', flip_d1(X)))
        transforms.append(('flip_d2', flip_d2(X)))
        transforms.append(('transpose', transpose(X)))

    # Find lex-min
    best_tag = 'identity'
    best_grid = X
    best_str = _grid_to_canonical_string(X)

    for tag, grid in transforms:
        grid_str = _grid_to_canonical_string(grid)
        if grid_str < best_str:
            best_str = grid_str
            best_grid = grid
            best_tag = tag

    # Look up inverse
    inv_tag = _INVERSE_TRANSFORM[best_tag]

    return (best_grid, best_tag, inv_tag)
