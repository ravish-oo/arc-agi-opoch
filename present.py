"""
Present: Input-only relational structure for grids.

Per math_spec.md: "present congruence of X under free moves 𝒢 (coarsest
input-only WL fixed point)."

Relations (always-on):
- E4 (4-adjacency)
- sameRow, sameCol (equivalences by row/col index)
- sameColor (equivalence by raw color)
- sameComp8 (equivalence by 8-connected component with same color)
- bandRow, bandCol (bands from change-edges)
- E8, CBC1, CBC2

CBC (Canonical Block Code):
- Extract r×r neighborhood → OFA (offset-fit-align) → D8 canonicalization → hash
- CBC1 (r=1, 3×3), CBC2 (r=2, 5×5)
- Color-blind structure token

Per math_spec.md line 36: "CBC: 'color-blind canonical patch' (OFA inside patch + D8)"
"""

from typing import List, Tuple, Dict, Set, Optional
from grid import Grid
from stable import stable_hash64

# Type aliases
Position = Tuple[int, int]
Partition = Dict[Position, int]  # Maps positions to class IDs
Present = Dict[str, any]


def build_present(X: Grid, opts: Dict[str, bool], phases: Optional[Tuple[Optional[int], Optional[int], Optional[int]]] = None) -> Present:
    """
    Build input-only present from grid X.

    Always includes: E4, sameRow, sameCol, sameColor, sameComp8, bandRow, bandCol, CBC1, CBC2, CBC3
    Optional (via opts): E8
    Optional (via phases): row_phase, col_phase, diag_phase

    Per math_spec_addon_a_bit_more.md: "CBC at r=1,2,3 as unary (always-on)"
    Per math_spec.md: "optional Row/Col/Diag phases if input periodicity is
    consistent across all trains."

    Args:
        X: Input grid (never uses labels)
        opts: Dict with optional relation flags (E8)
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

    # Change-edge bands (legacy)
    row_bands, col_bands = detect_bands(X)
    present['bandRow'] = row_bands
    present['bandCol'] = col_bands

    # 1D WL bands (per math_spec_addon_a_bit_more.md)
    # These refine positions by 1D WL structure along rows/cols
    present['rowWL'] = compute_1d_wl_rows(X)
    present['colWL'] = compute_1d_wl_cols(X)

    # Optional phases (unary predicates)
    if phases is not None:
        row_k, col_k, diag_k = phases
        present['phases'] = (row_k, col_k, diag_k)
    else:
        present['phases'] = (None, None, None)

    # Always-on CBC at all radii (per math_spec_addon_a_bit_more.md)
    present['CBC1'] = cbc_r(X, 1)
    present['CBC2'] = cbc_r(X, 2)
    present['CBC3'] = cbc_r(X, 3)

    # Optional relations
    if opts.get('E8', False):
        present['E8'] = _build_E8(X)

    return present


def _build_E4(X: Grid) -> List[Tuple[Position, Position]]:
    """Build 4-neighborhood pairs (up, down, left, right)."""
    edges = []
    for r in range(X.H):
        for c in range(X.W):
            # Right neighbor
            if c + 1 < X.W:
                edges.append(((r, c), (r, c + 1)))
            # Down neighbor
            if r + 1 < X.H:
                edges.append(((r, c), (r + 1, c)))
    return edges


def _build_E8(X: Grid) -> List[Tuple[Position, Position]]:
    """Build 8-neighborhood pairs (all adjacent including diagonals)."""
    edges = []
    for r in range(X.H):
        for c in range(X.W):
            # All 8 neighbors
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < X.H and 0 <= nc < X.W:
                        edges.append(((r, c), (nr, nc)))
    return edges


def _build_sameRow(X: Grid) -> Partition:
    """Build sameRow equivalence: positions in same row get same class."""
    partition = {}
    for r in range(X.H):
        for c in range(X.W):
            partition[(r, c)] = r  # Row index as class
    return partition


def _build_sameCol(X: Grid) -> Partition:
    """Build sameCol equivalence: positions in same column get same class."""
    partition = {}
    for r in range(X.H):
        for c in range(X.W):
            partition[(r, c)] = c  # Col index as class
    return partition


def _build_sameColor(X: Grid) -> Partition:
    """Build sameColor equivalence: positions with same color get same class."""
    partition = {}
    for r in range(X.H):
        for c in range(X.W):
            partition[(r, c)] = X[r][c]
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

            # Check all 8 neighbors
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < X.H and 0 <= nc < X.W:
                        if (nr, nc) not in visited and X[nr][nc] == color:
                            stack.append((nr, nc))

        return component

    # Build components
    for r in range(X.H):
        for c in range(X.W):
            pos = (r, c)
            if pos not in visited:
                component = flood_fill(pos)
                for p in component:
                    components[p] = comp_id
                comp_id += 1

    return components


def detect_bands(X: Grid) -> Tuple[Partition, Partition]:
    """
    Detect row and column bands from change-edges.

    A band is a contiguous set of rows (or cols) where all rows (or cols)
    in the band have the same "signature" (e.g., no internal changes).

    For now, use simple change-edge detection: row band changes when
    row differs from previous row.

    Args:
        X: Input grid

    Returns:
        (row_bands, col_bands) partitions

    Examples:
        >>> g = Grid([[1, 2], [1, 2], [3, 4]])
        >>> row_bands, col_bands = detect_bands(g)
        >>> row_bands[(0,0)] == row_bands[(1,0)]
        True
        >>> row_bands[(2,0)] != row_bands[(1,0)]
        True
    """
    # Row bands: consecutive rows with same content
    row_bands = {}
    row_band_id = 0
    prev_row_content = None

    for r in range(X.H):
        row_content = tuple(X[r][c] for c in range(X.W))
        if row_content != prev_row_content:
            row_band_id += 1
        for c in range(X.W):
            row_bands[(r, c)] = row_band_id
        prev_row_content = row_content

    # Col bands: consecutive cols with same content
    col_bands = {}
    col_band_id = 0
    prev_col_content = None

    for c in range(X.W):
        col_content = tuple(X[r][c] for r in range(X.H))
        if col_content != prev_col_content:
            col_band_id += 1
        for r in range(X.H):
            col_bands[(r, c)] = col_band_id
        prev_col_content = col_content

    return row_bands, col_bands


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
    window_size = 2 * r + 1

    for r_pos in range(X.H):
        for c_pos in range(X.W):
            # Extract window centered at (r_pos, c_pos)
            patch = []
            for dr in range(-r, r + 1):
                row = []
                for dc in range(-r, r + 1):
                    nr, nc = r_pos + dr, c_pos + dc
                    if 0 <= nr < X.H and 0 <= nc < X.W:
                        row.append(X[nr][nc])
                    else:
                        row.append(-1)  # Out of bounds marker
                patch.append(row)

            # OFA: Offset-Fit-Align
            patch_ofa = _ofa(patch)

            # D8: Apply all 8 transformations and pick lex-min
            canonical = _d8_canonical(patch_ofa)

            # Hash to get token
            token = stable_hash64(canonical)
            tokens[(r_pos, c_pos)] = token

    return tokens


def _ofa(patch: List[List[int]]) -> Tuple:
    """
    Offset-Fit-Align: normalize patch.

    1. Offset: subtract minimum to make 0-based
    2. Fit: crop to bounding box of non-(-1) values
    3. Return as tuple for hashing

    Args:
        patch: 2D list of values

    Returns:
        Tuple representation of normalized patch
    """
    # Offset: subtract min
    flat = [v for row in patch for v in row if v != -1]
    if not flat:
        return ()

    min_val = min(flat)
    patch_offset = [[v - min_val if v != -1 else -1 for v in row] for row in patch]

    # Fit: crop to bounding box
    min_r, max_r = None, None
    min_c, max_c = None, None

    n = len(patch_offset)
    for r in range(n):
        for c in range(n):
            if patch_offset[r][c] != -1:
                if min_r is None or r < min_r:
                    min_r = r
                if max_r is None or r > max_r:
                    max_r = r
                if min_c is None or c < min_c:
                    min_c = c
                if max_c is None or c > max_c:
                    max_c = c

    if min_r is None:
        return ()

    # Extract bounding box
    cropped = []
    for r in range(min_r, max_r + 1):
        row = []
        for c in range(min_c, max_c + 1):
            row.append(patch_offset[r][c])
        cropped.append(tuple(row))

    return tuple(cropped)


def _d8_canonical(patch: Tuple) -> Tuple:
    """
    Apply all D8 transformations and return lex-min.

    D8 = {identity, rot90, rot180, rot270, flip_h, flip_v, flip_d1, flip_d2}

    Args:
        patch: Tuple representation of patch

    Returns:
        Lex-min transformation
    """
    if not patch:
        return ()

    # Convert to list for transformations
    patch_list = [list(row) for row in patch]

    transformations = [
        patch_list,
        _rotate_90(patch_list),
        _rotate_180(patch_list),
        _rotate_270(patch_list),
        _flip_horizontal(patch_list),
        _flip_vertical(patch_list),
        _flip_diagonal_main(patch_list),
        _flip_diagonal_anti(patch_list),
    ]

    # Convert to tuples and pick lex-min
    as_tuples = [tuple(tuple(row) for row in t) for t in transformations]
    return min(as_tuples)


def _rotate_90(patch: List[List[int]]) -> List[List[int]]:
    """Rotate patch 90 degrees clockwise."""
    n = len(patch)
    m = len(patch[0]) if n > 0 else 0
    rotated = [[0 for _ in range(n)] for _ in range(m)]

    for r in range(n):
        for c in range(m):
            rotated[c][n - 1 - r] = patch[r][c]

    return rotated


def _rotate_180(patch: List[List[int]]) -> List[List[int]]:
    """Rotate patch 180 degrees."""
    return _rotate_90(_rotate_90(patch))


def _rotate_270(patch: List[List[int]]) -> List[List[int]]:
    """Rotate patch 270 degrees clockwise."""
    n = len(patch)
    m = len(patch[0]) if n > 0 else 0
    rotated = [[0 for _ in range(n)] for _ in range(m)]

    for r in range(n):
        for c in range(m):
            rotated[m - 1 - c][r] = patch[r][c]

    return rotated


def _flip_horizontal(patch: List[List[int]]) -> List[List[int]]:
    """Flip patch horizontally (left-right)."""
    return [row[::-1] for row in patch]


def _flip_vertical(patch: List[List[int]]) -> List[List[int]]:
    """Flip patch vertically (up-down)."""
    return patch[::-1]


def _flip_diagonal_main(patch: List[List[int]]) -> List[List[int]]:
    """Flip along main diagonal (top-left to bottom-right)."""
    n = len(patch)
    m = len(patch[0]) if n > 0 else 0
    flipped = [[0 for _ in range(n)] for _ in range(m)]

    for r in range(n):
        for c in range(m):
            flipped[c][r] = patch[r][c]

    return flipped


def _flip_diagonal_anti(patch: List[List[int]]) -> List[List[int]]:
    """Flip along anti-diagonal (top-right to bottom-left)."""
    return _rotate_90(_flip_horizontal(patch))


def detect_period_axis(X: Grid, axis: str) -> Optional[int]:
    """
    Detect period k along axis ('row' or 'col').

    If grid has period k along axis, return k, else None.

    Args:
        X: Input grid
        axis: 'row' or 'col'

    Returns:
        Period k if found, else None

    Examples:
        >>> g = Grid([[1, 2], [1, 2], [1, 2]])
        >>> detect_period_axis(g, 'row')
        1
    """
    if axis == 'row':
        # Check if rows repeat with period k
        for k in range(1, X.H):
            if X.H % k != 0:
                continue
            periodic = True
            for r in range(k, X.H):
                for c in range(X.W):
                    if X[r][c] != X[r % k][c]:
                        periodic = False
                        break
                if not periodic:
                    break
            if periodic:
                return k
    else:  # col
        # Check if cols repeat with period k
        for k in range(1, X.W):
            if X.W % k != 0:
                continue
            periodic = True
            for r in range(X.H):
                for c in range(k, X.W):
                    if X[r][c] != X[r][c % k]:
                        periodic = False
                        break
                if not periodic:
                    break
            if periodic:
                return k

    return None


def check_phase_consistency(trains: List[Tuple[Grid, Grid]]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Check if all training inputs have consistent periodicity.

    Per math_spec.md line 37: "Optional phases: input-detected periodicities
    (via autocorrelation), included only if consistent across all training inputs."

    Args:
        trains: List of (input, output) pairs

    Returns:
        (row_k, col_k, diag_k) tuple with periods if consistent, else (None, None, None)

    Examples:
        >>> x1 = Grid([[1, 2], [1, 2]])
        >>> y1 = Grid([[0, 0], [0, 0]])
        >>> x2 = Grid([[3, 4], [3, 4]])
        >>> y2 = Grid([[1, 1], [1, 1]])
        >>> check_phase_consistency([(x1, y1), (x2, y2)])
        (1, None, None)
    """
    if not trains:
        return (None, None, None)

    # Check row periodicity
    row_k = None
    first_X = trains[0][0]
    candidate_k = detect_period_axis(first_X, 'row')
    if candidate_k is not None:
        all_have_period = True
        for X, Y in trains:
            k = detect_period_axis(X, 'row')
            if k != candidate_k:
                all_have_period = False
                break
        if all_have_period:
            row_k = candidate_k

    # Check col periodicity
    col_k = None
    candidate_k = detect_period_axis(first_X, 'col')
    if candidate_k is not None:
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


def compute_1d_wl_rows(X: Grid, max_iters: int = 10) -> Partition:
    """
    Compute 1D WL partition along rows.

    Per math_spec_addon_a_bit_more.md: "1D row WL bands: close rows under
    1D WL (color + E1 adjacency along that axis)."

    Each row is treated as a 1D sequence. WL refines positions within each row
    based on (color, left_neighbor_color, right_neighbor_color).

    Args:
        X: Input grid
        max_iters: Maximum WL iterations

    Returns:
        Partition mapping positions to row-WL class IDs

    Examples:
        >>> g = Grid([[1, 2, 1], [1, 2, 1], [3, 4, 3]])
        >>> p = compute_1d_wl_rows(g)
        >>> p[(0,0)] == p[(1,0)]  # Same row structure
        True
        >>> p[(0,0)] != p[(2,0)]  # Different row structure
        True
    """
    # Initialize: each position gets hash of (row_idx, color)
    coloring: Dict[Position, int] = {}
    for r in range(X.H):
        for c in range(X.W):
            color = X[r][c]
            # Initial hash: (row_idx, position_in_row, color)
            atom = (r, c, color)
            coloring[(r, c)] = stable_hash64(atom)

    # Iterate WL refinement along rows
    for iteration in range(max_iters):
        new_coloring: Dict[Position, int] = {}
        changed = False

        for r in range(X.H):
            for c in range(X.W):
                current_color = coloring[(r, c)]

                # Get E1 neighbors (left, right) within same row
                left_color = coloring[(r, c-1)] if c > 0 else None
                right_color = coloring[(r, c+1)] if c < X.W - 1 else None

                # New signature
                signature = (current_color, left_color, right_color)
                new_hash = stable_hash64(signature)

                if new_hash != current_color:
                    changed = True

                new_coloring[(r, c)] = new_hash

        coloring = new_coloring

        if not changed:
            break

    return coloring


def compute_1d_wl_cols(X: Grid, max_iters: int = 10) -> Partition:
    """
    Compute 1D WL partition along columns.

    Per math_spec_addon_a_bit_more.md: "1D col WL bands: close columns under
    1D WL (color + E1 adjacency along that axis)."

    Each column is treated as a 1D sequence. WL refines positions within each column
    based on (color, top_neighbor_color, bottom_neighbor_color).

    Args:
        X: Input grid
        max_iters: Maximum WL iterations

    Returns:
        Partition mapping positions to col-WL class IDs

    Examples:
        >>> g = Grid([[1, 2, 3], [2, 2, 4], [1, 2, 3]])
        >>> p = compute_1d_wl_cols(g)
        >>> p[(0,0)] == p[(2,0)]  # Same col structure
        True
        >>> p[(0,0)] != p[(0,1)]  # Different col structure
        True
    """
    # Initialize: each position gets hash of (col_idx, color)
    coloring: Dict[Position, int] = {}
    for r in range(X.H):
        for c in range(X.W):
            color = X[r][c]
            # Initial hash: (col_idx, position_in_col, color)
            atom = (c, r, color)
            coloring[(r, c)] = stable_hash64(atom)

    # Iterate WL refinement along columns
    for iteration in range(max_iters):
        new_coloring: Dict[Position, int] = {}
        changed = False

        for r in range(X.H):
            for c in range(X.W):
                current_color = coloring[(r, c)]

                # Get E1 neighbors (up, down) within same column
                up_color = coloring[(r-1, c)] if r > 0 else None
                down_color = coloring[(r+1, c)] if r < X.H - 1 else None

                # New signature
                signature = (current_color, up_color, down_color)
                new_hash = stable_hash64(signature)

                if new_hash != current_color:
                    changed = True

                new_coloring[(r, c)] = new_hash

        coloring = new_coloring

        if not changed:
            break

    return coloring
