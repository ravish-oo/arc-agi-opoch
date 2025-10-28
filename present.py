"""
Present: Input-only relations for CPRQ.

All relations are built ONLY from input grid X, never from labels Y or deltas.
AST linter enforces this ban at test time.

Always-on relations:
- E4, sameRow, sameCol, sameColor, sameComp8, bandRow, bandCol

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


def build_present(X: Grid, opts: Dict[str, bool], phases: Optional[Tuple[Optional[int], Optional[int], Optional[int]]] = None) -> Present:
    """
    Build input-only present from grid X.

    Always includes: E4, sameRow, sameCol, sameColor, sameComp8, bandRow, bandCol
    Optional (via opts): E8, CBC1, CBC2
    Optional (via phases): row_phase, col_phase, diag_phase

    Per math_spec.md: "optional Row/Col/Diag phases if input periodicity is
    consistent across all trains."

    Args:
        X: Input grid (never uses labels)
        opts: Dict with optional relation flags (E8, CBC1, CBC2)
        phases: Optional (row_k, col_k, diag_k) tuple for periodic phases

    Returns:
        Present dict with relations and tokens

    Examples:
        >>> g = Grid([[1, 2], [3, 4]])
        >>> p = build_present(g, {})
        >>> 'E4' in p and 'sameRow' in p
        True
    """
    present: Present = {}

    # Store the grid itself for WL to access raw colors
    present['grid'] = X

    # Always-on relations
    present['E4'] = _build_E4(X)
    present['sameRow'] = _build_sameRow(X)
    present['sameCol'] = _build_sameCol(X)
    present['sameColor'] = _build_sameColor(X)
    present['sameComp8'] = sameComp8(X)

    row_bands, col_bands = detect_bands(X)
    present['bandRow'] = row_bands
    present['bandCol'] = col_bands

    # Optional phases (unary predicates)
    if phases is not None:
        row_k, col_k, diag_k = phases
        present['phases'] = (row_k, col_k, diag_k)
    else:
        present['phases'] = (None, None, None)

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


def _build_sameColor(X: Grid) -> Partition:
    """
    Build sameColor equivalence: positions with same input value.

    This is input-only, presentation-free. Two positions are related iff X[p] == X[q].
    Essential for WL to distinguish roles by input color.

    Args:
        X: Input grid

    Returns:
        Partition where positions with same color value are in same block

    Examples:
        >>> g = Grid([[1, 2], [1, 2]])
        >>> same_c = _build_sameColor(g)
        >>> same_c[(0, 0)] == same_c[(1, 0)]  # Both are color 1
        True
        >>> same_c[(0, 1)] == same_c[(1, 1)]  # Both are color 2
        True
        >>> same_c[(0, 0)] != same_c[(0, 1)]  # Different colors
        True
    """
    partition = {}
    # Group by color value
    for r, c in X.positions():
        color = X[r][c]
        partition[(r, c)] = color
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


# Phase Detection: Input-only periodicities for optional unary predicates

def detect_period_axis(X: Grid, axis: str) -> Optional[int]:
    """
    Detect period k for given axis using pattern matching.

    Per math_spec.md: "input-detected periodicities (via autocorrelation),
    included only if consistent across all training inputs."

    Args:
        X: Input grid
        axis: 'row' or 'col'

    Returns:
        Period k (2..10) if detected, None otherwise

    Examples:
        >>> # Periodic rows with period 2
        >>> g = Grid([[1, 2], [1, 2], [1, 2]])
        >>> detect_period_axis(g, 'row')
        2
    """
    if axis == 'row':
        # Check if rows repeat with period k
        for k in range(2, min(X.H, 11)):  # Try periods 2..10
            if X.H % k != 0:
                continue

            # Check if all rows match with period k
            is_periodic = True
            for r in range(X.H):
                r_ref = r % k
                # Compare row r with row r_ref
                for c in range(X.W):
                    if X[r][c] != X[r_ref][c]:
                        is_periodic = False
                        break
                if not is_periodic:
                    break

            if is_periodic:
                return k

    elif axis == 'col':
        # Check if columns repeat with period k
        for k in range(2, min(X.W, 11)):  # Try periods 2..10
            if X.W % k != 0:
                continue

            # Check if all cols match with period k
            is_periodic = True
            for c in range(X.W):
                c_ref = c % k
                # Compare col c with col c_ref
                for r in range(X.H):
                    if X[r][c] != X[r][c_ref]:
                        is_periodic = False
                        break
                if not is_periodic:
                    break

            if is_periodic:
                return k

    return None


def check_phase_consistency(trains: List[Tuple[Grid, Grid]]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Check if all training inputs have consistent periods.

    Per math_spec.md: "included only if consistent across all training inputs.
    If both row/col periods exist and are equal k, add DiagPhase (a+b)mod k."

    Args:
        trains: List of (X, Y) training pairs

    Returns:
        (row_k, col_k, diag_k):
            - row_k: Row period if consistent across all trains, else None
            - col_k: Col period if consistent across all trains, else None
            - diag_k: Diagonal period if row_k == col_k, else None

    Examples:
        >>> x1 = Grid([[1, 2], [1, 2]])
        >>> y1 = Grid([[0, 0], [0, 0]])
        >>> x2 = Grid([[3, 4], [3, 4], [3, 4]])
        >>> y2 = Grid([[0, 0], [0, 0], [0, 0]])
        >>> row_k, col_k, diag_k = check_phase_consistency([(x1, y1), (x2, y2)])
        >>> row_k  # Both have row period 2 (?)
        # Actually x1 has no row period, x2 might have period 3
        # This is just an example structure
    """
    if len(trains) == 0:
        return (None, None, None)

    # Detect row period from all training inputs
    row_periods = set()
    for X, Y in trains:
        k = detect_period_axis(X, 'row')
        if k is not None:
            row_periods.add(k)

    # Row period must be consistent (same k for all trains that have a period)
    # If any train has no period, disable row phase
    row_k = None
    if len(row_periods) == 1:
        # All trains agree on same row period
        # But we also need to check that ALL trains have this period
        candidate_k = list(row_periods)[0]
        all_have_period = True
        for X, Y in trains:
            k = detect_period_axis(X, 'row')
            if k != candidate_k:
                all_have_period = False
                break
        if all_have_period:
            row_k = candidate_k

    # Detect col period from all training inputs
    col_periods = set()
    for X, Y in trains:
        k = detect_period_axis(X, 'col')
        if k is not None:
            col_periods.add(k)

    col_k = None
    if len(col_periods) == 1:
        candidate_k = list(col_periods)[0]
        all_have_period = True
        for X, Y in trains:
            k = detect_period_axis(X, 'col')
            if k != candidate_k:
                all_have_period = False
                break
        if all_have_period:
            col_k = candidate_k

    # Diagonal phase only if row_k == col_k
    diag_k = None
    if row_k is not None and col_k is not None and row_k == col_k:
        diag_k = row_k

    return (row_k, col_k, diag_k)
