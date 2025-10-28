"""
1-WL (Weisfeiler-Leman) refinement on disjoint union.

Runs WL on the disjoint union of all training inputs to produce
aligned role IDs across examples. This is the key to CPRQ:
positions with same role across examples get same ID.

Atom seed: (sameColor_tag, sameComp8_tag, bandRow_tag, bandCol_tag, CBC_token_or_0, is_border)

Algorithm:
1. Build initial coloring from atom seed
2. Iterate: hash each position's (color, neighbor_colors)
3. Stop at fixed point or 50 iterations
"""

from typing import List, Dict, Tuple, Any, Set
from grid import Grid
from present import Present
from stable import stable_hash64, sorted_items
from equiv import relabel_stable


Position = Tuple[int, int]
Partition = Dict[Position, int]
GlobalPosition = Tuple[int, Position]  # (grid_index, (r, c))


def wl_disjoint_union(presents: List[Present], debug_positions: List[GlobalPosition] = None) -> List[Partition]:
    """
    Run 1-WL on disjoint union of all grids.

    The disjoint union treats each grid as separate, but the WL refinement
    process assigns consistent role IDs to positions with similar neighborhoods
    across all grids.

    Args:
        presents: List of Present dicts, one per training input
        debug_positions: Optional list of global positions to trace

    Returns:
        List of partitions, one per grid, with aligned role IDs

    Examples:
        >>> # Two grids with similar structure get aligned IDs
        >>> g1 = Grid([[1, 2], [3, 4]])
        >>> g2 = Grid([[5, 6], [7, 8]])
        >>> p1 = build_present(g1, {})
        >>> p2 = build_present(g2, {})
        >>> parts = wl_disjoint_union([p1, p2])
        >>> len(parts) == 2
        True
    """
    if not presents:
        return []

    # Build initial coloring from atom seed
    coloring = _build_initial_coloring(presents)

    # Build edge relations (E4 from all grids)
    edges = _build_edges(presents)

    # Iterate WL refinement to fixed point
    coloring = _refine_to_fixed_point(coloring, edges, debug_positions=debug_positions)

    # Split back into per-grid partitions
    partitions = _split_into_grids(coloring, presents)

    return partitions


def _build_initial_coloring(presents: List[Present]) -> Dict[GlobalPosition, int]:
    """
    Build initial coloring from atom seed.

    Atom seed: (sameColor_tag, sameComp8_tag, bandRow_tag, bandCol_tag, CBC_token_or_0, is_border)

    Args:
        presents: List of Present dicts

    Returns:
        Dict mapping global positions to initial color IDs
    """
    # Collect all atoms across all grids
    atoms: List[Tuple[Any, GlobalPosition]] = []

    for grid_idx, present in enumerate(presents):
        grid = present['grid']  # Get the actual grid for raw color access
        same_comp8 = present['sameComp8']
        band_row = present['bandRow']
        band_col = present['bandCol']

        # Get CBC token if present (CBC1 or CBC2, preferring CBC2)
        cbc_tokens = None
        if 'CBC2' in present:
            cbc_tokens = present['CBC2']
        elif 'CBC1' in present:
            cbc_tokens = present['CBC1']

        # Get grid dimensions from grid
        H = grid.H
        W = grid.W
        positions = list(same_comp8.keys())

        for pos in positions:
            r, c = pos
            is_border = (r == 0 or r == H - 1 or c == 0 or c == W - 1)

            # Build atom with RAW color value (0-9) for global stability
            color_tag = grid[r][c]  # Raw palette value, globally stable across disjoint union
            comp8_tag = same_comp8[pos]
            row_tag = band_row[pos]
            col_tag = band_col[pos]
            cbc_token = cbc_tokens[pos] if cbc_tokens else 0

            atom = (color_tag, comp8_tag, row_tag, col_tag, cbc_token, is_border)

            global_pos = (grid_idx, pos)
            atoms.append((atom, global_pos))

    # Hash atoms to get initial colors
    # Group by atom, assign IDs
    atom_to_positions: Dict[Any, List[GlobalPosition]] = {}
    for atom, gpos in atoms:
        # Make atom hashable
        atom_key = stable_hash64(atom)
        if atom_key not in atom_to_positions:
            atom_to_positions[atom_key] = []
        atom_to_positions[atom_key].append(gpos)

    # Assign color IDs deterministically
    coloring: Dict[GlobalPosition, int] = {}
    color_id = 0

    for atom_key in sorted(atom_to_positions.keys()):
        for gpos in sorted(atom_to_positions[atom_key]):
            coloring[gpos] = color_id
        color_id += 1

    return coloring


def _build_edges(presents: List[Present]) -> Set[Tuple[GlobalPosition, GlobalPosition]]:
    """
    Build edge set from E4 relations across all grids.

    Args:
        presents: List of Present dicts

    Returns:
        Set of (global_pos1, global_pos2) edges
    """
    edges: Set[Tuple[GlobalPosition, GlobalPosition]] = []

    for grid_idx, present in enumerate(presents):
        e4_pairs = present['E4']

        for pos1, pos2 in e4_pairs:
            gpos1 = (grid_idx, pos1)
            gpos2 = (grid_idx, pos2)

            # Add both directions for undirected graph
            edges.append((gpos1, gpos2))
            edges.append((gpos2, gpos1))

    return set(edges)


def _refine_to_fixed_point(
    coloring: Dict[GlobalPosition, int],
    edges: Set[Tuple[GlobalPosition, GlobalPosition]],
    max_iters: int = 50,
    debug_positions: List[GlobalPosition] = None
) -> Dict[GlobalPosition, int]:
    """
    Refine coloring using WL until fixed point.

    At each iteration, each position's new color is:
    hash(current_color, sorted([neighbor_colors]))

    Args:
        coloring: Initial coloring
        edges: Edge set
        max_iters: Maximum iterations (default 50)
        debug_positions: Optional list of positions to trace

    Returns:
        Final coloring after fixed point
    """
    # Build adjacency list for faster lookup
    neighbors: Dict[GlobalPosition, List[GlobalPosition]] = {}
    for pos in coloring.keys():
        neighbors[pos] = []

    for pos1, pos2 in edges:
        if pos1 in neighbors:
            neighbors[pos1].append(pos2)

    # Ensure deterministic neighbor order
    for pos in neighbors:
        neighbors[pos] = sorted(neighbors[pos])

    # Debug tracking
    if debug_positions:
        print("\n=== WL Refinement Trace ===")
        print(f"Tracking positions: {debug_positions}\n")
        for pos in debug_positions:
            if pos in coloring:
                print(f"Initial: {pos} → color {coloring[pos]}")
        print()

    # Iterate refinement
    for iteration in range(max_iters):
        new_coloring: Dict[GlobalPosition, int] = {}
        color_signatures: Dict[Any, List[GlobalPosition]] = {}

        # Compute new signature for each position
        for pos in sorted(coloring.keys()):
            current_color = coloring[pos]

            # Get neighbor colors (in sorted order)
            neighbor_colors = [coloring[nbr] for nbr in neighbors.get(pos, [])]
            neighbor_colors.sort()

            # Signature: (current_color, neighbor_colors)
            signature = (current_color, tuple(neighbor_colors))
            sig_hash = stable_hash64(signature)

            if sig_hash not in color_signatures:
                color_signatures[sig_hash] = []
            color_signatures[sig_hash].append(pos)

            # Debug trace
            if debug_positions and pos in debug_positions:
                print(f"Iter {iteration}: {pos}")
                print(f"  Current color: {current_color}")
                print(f"  Neighbor colors: {neighbor_colors}")
                print(f"  Signature hash: {sig_hash}")

        # Assign new colors based on signatures
        new_color_id = 0
        for sig_hash in sorted(color_signatures.keys()):
            for pos in sorted(color_signatures[sig_hash]):
                new_coloring[pos] = new_color_id
            new_color_id += 1

        # Debug trace new colors
        if debug_positions:
            print(f"\nAfter iteration {iteration}:")
            for pos in debug_positions:
                if pos in new_coloring:
                    old_c = coloring.get(pos, '?')
                    new_c = new_coloring[pos]
                    changed = "←" if old_c != new_c else ""
                    print(f"  {pos}: {old_c} → {new_c} {changed}")

            # Check if debug positions have merged
            debug_colors = [new_coloring[pos] for pos in debug_positions if pos in new_coloring]
            if len(set(debug_colors)) == 1:
                print(f"\n⚠️  MERGE DETECTED at iteration {iteration}! All debug positions now have color {debug_colors[0]}")
            print()

        # Check for fixed point
        if new_coloring == coloring:
            # Fixed point reached
            if debug_positions:
                print(f"Fixed point reached at iteration {iteration}")
            break

        coloring = new_coloring

    return coloring


def _split_into_grids(
    coloring: Dict[GlobalPosition, int],
    presents: List[Present]
) -> List[Partition]:
    """
    Split global coloring back into per-grid partitions.

    Uses GLOBAL relabeling to preserve WL alignment across grids.
    Positions with the same WL color get the same role ID regardless of grid.

    Args:
        coloring: Global coloring
        presents: List of Present dicts (to get positions)

    Returns:
        List of partitions, one per grid
    """
    # Collect all unique WL colors across all grids
    all_wl_colors = sorted(set(coloring.values()))

    # Create global mapping: WL color -> role ID
    # This ensures positions with same WL color get same role ID
    wl_color_to_role = {wl_color: role_id for role_id, wl_color in enumerate(all_wl_colors)}

    # Apply global mapping to each grid
    partitions: List[Partition] = []
    for grid_idx, present in enumerate(presents):
        # Get all positions for this grid
        positions = list(present['sameComp8'].keys())

        # Extract colors for this grid using GLOBAL mapping
        partition: Partition = {}
        for pos in positions:
            gpos = (grid_idx, pos)
            wl_color = coloring[gpos]
            partition[pos] = wl_color_to_role[wl_color]

        partitions.append(partition)

    return partitions


def get_role_count(partition: Partition) -> int:
    """
    Count the number of distinct roles in a partition.

    Args:
        partition: Position -> role_id mapping

    Returns:
        Number of distinct roles

    Examples:
        >>> p = {(0,0): 0, (0,1): 0, (1,0): 1}
        >>> get_role_count(p)
        2
    """
    return len(set(partition.values()))


def wl_stats(partitions: List[Partition]) -> Dict[str, Any]:
    """
    Compute statistics about WL partitions.

    Args:
        partitions: List of partitions

    Returns:
        Dict with stats (role_counts, total_roles, etc.)
    """
    role_counts = [get_role_count(p) for p in partitions]

    return {
        'num_grids': len(partitions),
        'role_counts': role_counts,
        'min_roles': min(role_counts) if role_counts else 0,
        'max_roles': max(role_counts) if role_counts else 0,
        'total_positions': sum(len(p) for p in partitions),
    }
