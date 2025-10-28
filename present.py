"""
Present: Input-only relations for CPRQ.

All relations are built ONLY from input grid X, never from labels Y or deltas.
AST linter enforces this ban at test time.

Always-on relations:
- E4, sameRow, sameCol, sameComp8, bandRow, bandCol

Optional relations (via opts):
- E8, CBC1, CBC2

CBC (Canonical Block Code):
- Extract r×r neighborhood
- OFA: offset to 0-base, fit to bounding box
- D8: compute all 8 dihedral transformations, pick lexicographically smallest
- Hash with stable_hash64
"""

from typing import Dict, List, Tuple, Set, Optional, Any
from grid import Grid
from equiv import new_partition_from_equiv, relabel_stable
from stable import stable_hash64, row_major_string


# Type aliases
Present = Dict[str, Any]
Position = Tuple[int, int]
Partition = Dict[Position, int]


def build_present(X: Grid, opts: Dict[str, bool]) -> Present:
    """
    Build input-only present from grid X.

    Always includes: E4, sameRow, sameCol, sameComp8, bandRow, bandCol
    Optional (via opts): E8, CBC1, CBC2

    Args:
        X: Input grid (never uses labels)
        opts: Dict with optional relation flags (E8, CBC1, CBC2)

    Returns:
        Present dict with relations and tokens

    Examples:
        >>> g = Grid([[1, 2], [3, 4]])
        >>> p = build_present(g, {})
        >>> 'E4' in p and 'sameRow' in p
        True
    """
    present: Present = {}

    # Always-on relations
    present['E4'] = _build_E4(X)
    present['sameRow'] = _build_sameRow(X)
    present['sameCol'] = _build_sameCol(X)
    present['sameComp8'] = sameComp8(X)

    row_bands, col_bands = detect_bands(X)
    present['bandRow'] = row_bands
    present['bandCol'] = col_bands

    # Optional relations
    if opts.get('E8', False):
        present['E8'] = _build_E8(X)

    if opts.get('CBC1', False):
        present['CBC1'] = cbc_r(X, 1)

    if opts.get('CBC2', False):
        present['CBC2'] = cbc_r(X, 2)

    return present


def _build_E4(X: Grid) -> List[Tuple[Position, Position]]:
    """Build 4-neighborhood pairs (up, down, left, right)."""
    pairs = []
    for r, c in X.positions():
        # Right neighbor
        if c + 1 < X.W:
            pairs.append(((r, c), (r, c + 1)))
        # Down neighbor
        if r + 1 < X.H:
            pairs.append(((r, c), (r + 1, c)))
    return pairs


def _build_E8(X: Grid) -> List[Tuple[Position, Position]]:
    """Build 8-neighborhood pairs (E4 + diagonals)."""
    pairs = _build_E4(X)

    # Add diagonals
    for r, c in X.positions():
        # Down-right diagonal
        if r + 1 < X.H and c + 1 < X.W:
            pairs.append(((r, c), (r + 1, c + 1)))
        # Down-left diagonal
        if r + 1 < X.H and c - 1 >= 0:
            pairs.append(((r, c), (r + 1, c - 1)))

    return pairs


def _build_sameRow(X: Grid) -> Partition:
    """Build sameRow equivalence: positions in same row."""
    partition = {}
    for r, c in X.positions():
        partition[(r, c)] = r
    return partition


def _build_sameCol(X: Grid) -> Partition:
    """Build sameCol equivalence: positions in same column."""
    partition = {}
    for r, c in X.positions():
        partition[(r, c)] = c
    return partition


def sameComp8(X: Grid) -> Partition:
    """
    Build 8-connected component equivalence.

    Two positions are in the same component if:
    1. They have the same color
    2. They are connected via 8-neighbors

    Args:
        X: Input grid

    Returns:
        Partition mapping positions to component IDs

    Examples:
        >>> g = Grid([[1, 1], [2, 2]])
        >>> comp = sameComp8(g)
        >>> comp[(0,0)] == comp[(0,1)]
        True
        >>> comp[(0,0)] != comp[(1,0)]
        True
    """
    visited = set()
    components = {}
    comp_id = 0

    def flood_fill(start: Position) -> Set[Position]:
        """Flood fill from start position using 8-neighbors."""
        stack = [start]
        component = set()
        color = X[start[0]][start[1]]

        while stack:
            r, c = stack.pop()
            if (r, c) in visited:
                continue

            visited.add((r, c))
            component.add((r, c))

            # Check 8 neighbors
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue

                    nr, nc = r + dr, c + dc
                    if 0 <= nr < X.H and 0 <= nc < X.W:
                        if (nr, nc) not in visited and X[nr][nc] == color:
                            stack.append((nr, nc))

        return component

    # Process all positions in deterministic order
    for r, c in X.positions():
        if (r, c) not in visited:
            component = flood_fill((r, c))
            for pos in component:
                components[pos] = comp_id
            comp_id += 1

    return relabel_stable(components)


def detect_bands(X: Grid) -> Tuple[Partition, Partition]:
    """
    Detect row and column bands via change edges.

    A change edge separates positions where value changes.
    Positions in the same band have no change edges between them.

    Args:
        X: Input grid

    Returns:
        (row_bands, col_bands) - two partitions

    Examples:
        >>> g = Grid([[1, 1], [2, 2]])
        >>> row_bands, col_bands = detect_bands(g)
        >>> row_bands[(0,0)] == row_bands[(0,1)]
        True
    """
    # Row bands: separate rows where vertical change occurs
    row_equiv_pairs = []
    for r in range(X.H):
        for c in range(X.W):
            # Same row positions are in same band by default
            if c + 1 < X.W:
                row_equiv_pairs.append(((r, c), (r, c + 1)))

            # Check if there's a vertical change edge
            if r + 1 < X.H:
                if X[r][c] == X[r + 1][c]:
                    # No change: same band
                    row_equiv_pairs.append(((r, c), (r + 1, c)))

    # Column bands: separate columns where horizontal change occurs
    col_equiv_pairs = []
    for r in range(X.H):
        for c in range(X.W):
            # Same col positions are in same band by default
            if r + 1 < X.H:
                col_equiv_pairs.append(((r, c), (r + 1, c)))

            # Check if there's a horizontal change edge
            if c + 1 < X.W:
                if X[r][c] == X[r][c + 1]:
                    # No change: same band
                    col_equiv_pairs.append(((r, c), (r, c + 1)))

    # Build partitions, ensuring all positions are included
    row_bands = new_partition_from_equiv(row_equiv_pairs) if row_equiv_pairs else {}
    col_bands = new_partition_from_equiv(col_equiv_pairs) if col_equiv_pairs else {}

    # For positions not in any equivalence pairs, add them as singletons
    all_positions = list(X.positions())

    # Row bands
    next_row_id = max(row_bands.values()) + 1 if row_bands else 0
    for pos in all_positions:
        if pos not in row_bands:
            row_bands[pos] = next_row_id
            next_row_id += 1

    # Col bands
    next_col_id = max(col_bands.values()) + 1 if col_bands else 0
    for pos in all_positions:
        if pos not in col_bands:
            col_bands[pos] = next_col_id
            next_col_id += 1

    return relabel_stable(row_bands), relabel_stable(col_bands)


def cbc_r(X: Grid, r: int) -> Dict[Position, int]:
    """
    Canonical Block Code with radius r.

    For each position, extract r×r neighborhood, apply OFA → D8 → hash.

    OFA (Offset-Fit-Align):
    - Extract values in r×r window around position
    - Offset: subtract minimum to make 0-based
    - Fit: crop to bounding box of non-background

    D8 (Dihedral group):
    - Compute all 8 transformations (4 rotations × 2 reflections)
    - Pick lexicographically smallest using row_major_string

    Hash:
    - Use stable_hash64 for deterministic token

    Args:
        X: Input grid
        r: Radius (1 for 3×3, 2 for 5×5, etc.)

    Returns:
        Dict mapping positions to CBC tokens

    Examples:
        >>> g = Grid([[1, 2], [3, 4]])
        >>> tokens = cbc_r(g, 1)
        >>> len(tokens) == 4
        True
    """
    tokens = {}

    for pos in X.positions():
        # Extract neighborhood
        patch = _extract_patch(X, pos, r)

        # OFA: offset and fit
        ofa_patch = _apply_ofs(patch)

        # D8: find canonical form
        canonical = _apply_d8_canonical(ofa_patch)

        # Hash
        token = stable_hash64(canonical)
        tokens[pos] = token

    return tokens


def _extract_patch(X: Grid, center: Position, r: int) -> List[List[int]]:
    """
    Extract (2r+1)×(2r+1) patch around center position.

    Out-of-bounds positions are filled with -1 (background marker).
    """
    cr, cc = center
    patch = []

    for dr in range(-r, r + 1):
        row = []
        for dc in range(-r, r + 1):
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < X.H and 0 <= nc < X.W:
                row.append(X[nr][nc])
            else:
                row.append(-1)
        patch.append(row)

    return patch


def _apply_ofs(patch: List[List[int]]) -> List[List[int]]:
    """
    Apply Offset-Fit-Align to patch.

    1. Offset: subtract minimum non-background value
    2. Keep background (-1) as -1
    3. Fit: no cropping (keep full patch for D8 invariance)
    """
    # Find minimum non-background value
    non_bg = [val for row in patch for val in row if val >= 0]

    if not non_bg:
        # All background
        return patch

    min_val = min(non_bg)

    # Offset
    offset_patch = []
    for row in patch:
        offset_row = []
        for val in row:
            if val >= 0:
                offset_row.append(val - min_val)
            else:
                offset_row.append(-1)
        offset_patch.append(offset_row)

    return offset_patch


def _apply_d8_canonical(patch: List[List[int]]) -> str:
    """
    Apply all 8 D8 transformations and return lexicographically smallest.

    D8 = {identity, rot90, rot180, rot270, flip_h, flip_v, flip_d1, flip_d2}

    Returns:
        Canonical form as row_major_string
    """
    candidates = []

    # Original
    candidates.append(row_major_string(patch))

    # 90° rotation
    candidates.append(row_major_string(_rotate_90(patch)))

    # 180° rotation
    candidates.append(row_major_string(_rotate_90(_rotate_90(patch))))

    # 270° rotation
    candidates.append(row_major_string(_rotate_90(_rotate_90(_rotate_90(patch)))))

    # Horizontal flip
    candidates.append(row_major_string(_flip_horizontal(patch)))

    # Vertical flip
    candidates.append(row_major_string(_flip_vertical(patch)))

    # Diagonal flip (main diagonal)
    candidates.append(row_major_string(_transpose(patch)))

    # Anti-diagonal flip
    candidates.append(row_major_string(_flip_horizontal(_transpose(patch))))

    # Return lexicographically smallest
    return min(candidates)


def _rotate_90(patch: List[List[int]]) -> List[List[int]]:
    """Rotate patch 90° clockwise."""
    n = len(patch)
    rotated = [[0] * n for _ in range(n)]

    for r in range(n):
        for c in range(n):
            rotated[c][n - 1 - r] = patch[r][c]

    return rotated


def _flip_horizontal(patch: List[List[int]]) -> List[List[int]]:
    """Flip patch horizontally (left-right)."""
    return [row[::-1] for row in patch]


def _flip_vertical(patch: List[List[int]]) -> List[List[int]]:
    """Flip patch vertically (top-bottom)."""
    return patch[::-1]


def _transpose(patch: List[List[int]]) -> List[List[int]]:
    """Transpose patch (reflect over main diagonal)."""
    n = len(patch)
    transposed = [[0] * n for _ in range(n)]

    for r in range(n):
        for c in range(n):
            transposed[c][r] = patch[r][c]

    return transposed
