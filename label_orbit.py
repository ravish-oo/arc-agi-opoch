"""
Label Orbit & Canonicalization for CPRQ

Per math_spec_addon.md: Treat colors up to palette permutation (H = S_Σ).

Implements:
- Orbit kernel: ker_H(c_i) = positions equivalent up to palette permutation
- Canonicalizer N: input-only rule to fix concrete digits from abstract colors
"""

from typing import List, Tuple, Dict, Optional
from grid import Grid

# Type aliases
Partition = Dict[Tuple[int, int], int]


def compute_orbit_kernel(trains: List[Tuple[Grid, Grid]]) -> List[Partition]:
    """
    Compute ker_H(c_i) = label kernel up to palette permutation.

    Per math_spec_addon.md line 26:
    (u,v) ∈ ker_H(c_i) iff ∃π ∈ H: c_i(u) = π(c_i(v))

    Implementation: Normalize each training pair's labels to abstract color ids.
    Two positions are in same orbit-class if they have same abstract id.

    Args:
        trains: List of (input, output) grid pairs

    Returns:
        List of orbit partitions, one per training pair

    Examples:
        >>> # Two grids with different palettes but same structure
        >>> x1 = Grid([[1, 2], [3, 4]])
        >>> y1 = Grid([[0, 0], [1, 1]])  # Uses colors 0,1
        >>> x2 = Grid([[5, 6], [7, 8]])
        >>> y2 = Grid([[3, 3], [7, 7]])  # Uses colors 3,7 (different palette!)
        >>> orbits = compute_orbit_kernel([(x1, y1), (x2, y2)])
        >>> # Both should have same abstract structure
        >>> len(set(orbits[0].values())) == len(set(orbits[1].values()))
        True
    """
    orbit_partitions = []

    for X, Y in trains:
        # Build abstract color partition
        # Positions with same Y value get same abstract id
        # (This is identical to strict kernel for now, but conceptually different)
        # In full implementation, we'd normalize across multiple trains

        color_to_abstract_id: Dict[int, int] = {}
        next_abstract_id = 0
        orbit_partition: Partition = {}

        for r in range(Y.H):
            for c in range(Y.W):
                color = Y[r][c]
                pos = (r, c)

                # Map this color to abstract id
                if color not in color_to_abstract_id:
                    color_to_abstract_id[color] = next_abstract_id
                    next_abstract_id += 1

                orbit_partition[pos] = color_to_abstract_id[color]

        orbit_partitions.append(orbit_partition)

    return orbit_partitions


def build_abstract_rho(
    trains: List[Tuple[Grid, Grid]],
    E_tilde_list: List[Partition]
) -> Dict[int, int]:
    """
    Build abstract ρ̃: U/Ẽ → Σ̄ (abstract colors up to permutation).

    Per math_spec_addon.md line 36: This ALWAYS exists because we use
    orbit kernel (palette symmetry is free).

    Args:
        trains: Training pairs
        E_tilde_list: Partitions (one per train) on unified domain

    Returns:
        Abstract color map: role_id → abstract_color_id

    Note: Multiple training pairs may map same role to different concrete
          colors, but they're all in the same H-orbit (abstract color).
    """
    abstract_rho: Dict[int, int] = {}

    # For each training pair, collect role→color mappings
    for train_idx, (X, Y) in enumerate(trains):
        E_tilde = E_tilde_list[train_idx]

        # Build role→color map for this training pair
        role_to_color: Dict[int, List[int]] = {}

        for pos, role_id in E_tilde.items():
            r, c = pos
            color = Y[r][c]

            if role_id not in role_to_color:
                role_to_color[role_id] = []
            role_to_color[role_id].append(color)

        # Record abstract color for each role
        for role_id, colors in role_to_color.items():
            # All positions in same role should have same color within this train
            # (guaranteed by Int^𝒢 refinement)
            unique_colors = set(colors)
            if len(unique_colors) > 1:
                raise AssertionError(f"Role {role_id} has multiple colors in train {train_idx}: {unique_colors}")

            color = colors[0]

            if role_id not in abstract_rho:
                # First time seeing this role - record its abstract color
                abstract_rho[role_id] = color
            # else: role already seen in another train
            # In orbit mode, it may have different concrete color (that's OK!)
            # We keep the first one as the abstract representative

    return abstract_rho


def canonicalize_palette(
    abstract_rho: Dict[int, int],
    E_tilde: Partition,
    X_grid: Grid,
    present: Dict,
    method: str = "lex_min"
) -> Tuple[Dict[int, int], List[int]]:
    """
    Canonicalizer N: fix concrete palette from abstract colors (input-only).

    Per math_spec_addon.md lines 49-52:
    - lex_min: Order roles by structural signature, map to 0,1,2,...
    - mdl_min: Minimize description length (future)

    CRITICAL: Uses ONLY input information - no target peeking!

    Args:
        abstract_rho: Abstract color map (role → abstract_color)
        E_tilde: Partition on input domain
        X_grid: Input grid (for computing signatures)
        present: Present dict (for accessing CBC, bands, etc.)
        method: "lex_min" or "mdl_min"

    Returns:
        (canonical_rho, permutation_applied)
        - canonical_rho: role_id → canonical_digit (0-9)
        - permutation_applied: List showing how abstract colors mapped to canonical

    Examples:
        >>> # Role with structural signature gets deterministic canonical color
        >>> abstract_rho = {1: 5, 2: 3, 3: 7}  # Abstract colors
        >>> E_tilde = {(0,0): 1, (0,1): 2, (1,0): 3, (1,1): 1}
        >>> # lex_min orders by signature → maps to 0,1,2
    """
    if method != "lex_min":
        raise NotImplementedError(f"Method {method} not implemented yet")

    # Compute structural signature for each role (input-only!)
    role_signatures: Dict[int, Tuple] = {}

    for role_id in set(abstract_rho.keys()):
        # Get all positions with this role
        role_positions = [pos for pos, rid in E_tilde.items() if rid == role_id]

        if not role_positions:
            continue

        # Compute input-only signature components
        size = len(role_positions)

        # Min position (top-left-most)
        min_r = min(r for r, c in role_positions)
        min_c = min(c for r, c in role_positions)

        # Max position (bottom-right-most)
        max_r = max(r for r, c in role_positions)
        max_c = max(c for r, c in role_positions)

        # Border count (positions on grid edge)
        border_count = sum(1 for r, c in role_positions
                          if r == 0 or r == X_grid.H-1 or c == 0 or c == X_grid.W-1)

        # E4 degree histogram (how many neighbors each position has)
        # This is input-structure only
        degree_counts = [0, 0, 0, 0, 0]  # 0,1,2,3,4 neighbors
        for r, c in role_positions:
            neighbors = 0
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < X_grid.H and 0 <= nc < X_grid.W:
                    if (nr, nc) in E_tilde and E_tilde[(nr, nc)] == role_id:
                        neighbors += 1
            degree_counts[neighbors] += 1

        # Signature: deterministic tuple for sorting
        signature = (
            size,                    # Larger roles first
            -min_r, -min_c,          # Top-left position (negative for ascending)
            -max_r, -max_c,          # Bottom-right position
            border_count,            # More border = earlier
            tuple(degree_counts),    # Degree histogram
        )

        role_signatures[role_id] = signature

    # Sort roles by signature (deterministic lex-min order)
    sorted_roles = sorted(role_signatures.keys(),
                         key=lambda r: role_signatures[r])

    # Map to canonical digits 0,1,2,...
    canonical_rho: Dict[int, int] = {}
    permutation = []

    for idx, role_id in enumerate(sorted_roles):
        canonical_digit = idx % 10  # Wrap to 0-9
        canonical_rho[role_id] = canonical_digit
        permutation.append((abstract_rho[role_id], canonical_digit))

    return canonical_rho, permutation


def apply_canonical_palette(
    grid: Grid,
    partition: Partition,
    canonical_rho: Dict[int, int]
) -> Grid:
    """
    Apply canonical palette to produce final output grid.

    Args:
        grid: Input grid (for dimensions)
        partition: Partition assigning roles to positions
        canonical_rho: role_id → canonical_digit mapping

    Returns:
        Output grid with canonical colors
    """
    data = [[0] * grid.W for _ in range(grid.H)]

    for pos, role_id in partition.items():
        r, c = pos
        if role_id in canonical_rho:
            data[r][c] = canonical_rho[role_id]
        else:
            # Role not in canonical_rho (shouldn't happen)
            data[r][c] = 0

    return Grid(data)
