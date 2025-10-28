"""
1-WL (Weisfeiler-Leman) refinement on disjoint union.

Runs WL on the disjoint union of all training inputs to produce
aligned role IDs across examples. This is the key to CPRQ:
positions with same role across examples get same ID.

Atom seed (per math_spec line 46): (X[p], CBC_token_or_0, is_border)

Algorithm:
1. Build initial coloring from minimal atom (NO positional tags)
2. Iterate: hash(current_color, adj_bag, comp_bag, bandRow_bag, bandCol_bag)
   - Adjacency (E4/E8): local neighborhoods
   - Equivalences (sameComp8, bandRow, bandCol): global class bags
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


def wl_disjoint_union(
    presents: List[Present],
    depth: int = 1,
    debug_positions: List[GlobalPosition] = None
) -> Tuple[List[Partition], int]:
    """
    Run WL on disjoint union of all grids.

    The disjoint union treats each grid as separate, but the WL refinement
    process assigns consistent role IDs to positions with similar neighborhoods
    across all grids.

    Properties:
        - Permutation invariance: Reordering grids produces equivalent partitions
          (roles may be renumbered but structure preserved)
        - Alignment: Positions across grids get same role ID iff WL-indistinguishable

    Args:
        presents: List of Present dicts, one per training input
        depth: WL depth (1 or 2). depth=1 is standard WL, depth=2 is 2-WL
        debug_positions: Optional list of global positions to trace

    Returns:
        (List of partitions, one per grid, with aligned role IDs, iteration count)

    Examples:
        >>> # Two grids with similar structure get aligned IDs
        >>> g1 = Grid([[1, 2], [3, 4]])
        >>> g2 = Grid([[5, 6], [7, 8]])
        >>> p1 = build_present(g1, {})
        >>> p2 = build_present(g2, {})
        >>> parts, iter_count = wl_disjoint_union([p1, p2], depth=1)
        >>> len(parts) == 2
        True
    """
    if not presents:
        return [], 0

    if depth == 2:
        # 2-WL: refine on pairs instead of nodes
        return _wl_2_disjoint_union(presents, debug_positions)

    # depth=1: standard WL (current behavior)
    # Build initial coloring from atom seed (minimal: X, CBC, border)
    coloring = _build_initial_coloring(presents)

    # Build relation data
    # Per math_spec: adjacency (E4/E8) as edges, equivalences as class membership
    adjacency_edges, equivalence_classes = _build_relation_data(presents)

    # Iterate WL refinement to fixed point
    coloring, iter_count = _refine_to_fixed_point(
        coloring,
        adjacency_edges,
        equivalence_classes,
        debug_positions=debug_positions
    )

    # Split back into per-grid partitions
    partitions = _split_into_grids(coloring, presents)

    return partitions, iter_count


def _build_initial_coloring(presents: List[Present]) -> Dict[GlobalPosition, int]:
    """
    Build initial coloring from atom seed.

    Per WO-MK-05.2a: Minimal atom = (X[p], CBC_token_or_0, is_border)

    Bands and components are NOT in the atom - they influence WL only via
    neighbor multisets during iteration (as relations, not unary tags).

    Args:
        presents: List of Present dicts

    Returns:
        Dict mapping global positions to initial color IDs
    """
    # Collect all atoms across all grids
    atoms: List[Tuple[Any, GlobalPosition]] = []

    for grid_idx, present in enumerate(presents):
        grid = present['grid']  # Get the actual grid for raw color access

        # Get CBC token if present (CBC1, CBC2, or CBC3, preferring highest radius)
        # Per math_spec_addon_a_bit_more.md: CBC at r=1,2,3 always included
        cbc_tokens = None
        if 'CBC3' in present:
            cbc_tokens = present['CBC3']
        elif 'CBC2' in present:
            cbc_tokens = present['CBC2']
        elif 'CBC1' in present:
            cbc_tokens = present['CBC1']

        # Get grid dimensions from grid
        H = grid.H
        W = grid.W

        # Get all positions from any present relation
        positions = list(present['sameComp8'].keys())

        for pos in positions:
            r, c = pos
            is_border = (r == 0 or r == H - 1 or c == 0 or c == W - 1)

            # Minimal atom: raw color, CBC token, and border flag
            # Per math_spec.md line 46: hash(CBC, raw, phases?)
            # Raw color needed for colormap tasks
            # NO band tags, NO component tags
            color_tag = grid[r][c]  # Raw palette value 0-9
            cbc_token = cbc_tokens[pos] if cbc_tokens else 0

            atom = (color_tag, cbc_token, is_border)

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

    # Assign colors deterministically using atom hash directly (stable)
    coloring: Dict[GlobalPosition, int] = {}

    for atom_key in sorted(atom_to_positions.keys()):
        for gpos in sorted(atom_to_positions[atom_key]):
            # Use atom hash directly as color (stable across unions)
            coloring[gpos] = atom_key

    return coloring


def _build_relation_data(presents: List[Present]) -> Tuple[Dict[str, Set], Dict[str, Dict[GlobalPosition, List[GlobalPosition]]]]:
    """
    Build relation data for WL iteration.

    Per math_spec lines 46-51:
    - Adjacency relations (E4, E8): edge sets for local neighborhoods
    - Equivalence relations (sameComp8, bandRow, bandCol): class membership maps
      for global bags

    Args:
        presents: List of Present dicts

    Returns:
        Tuple of (adjacency_edges, equivalence_classes)
        - adjacency_edges: Dict mapping relation name to edge set
        - equivalence_classes: Dict mapping relation name to
          {global_pos: [all_class_members]} for computing bags
    """
    adjacency_edges: Dict[str, Set] = {
        'E4': set(),
        'E8': set(),
    }

    equivalence_classes: Dict[str, Dict[GlobalPosition, List[GlobalPosition]]] = {
        'sameComp8': {},
        'bandRow': {},
        'bandCol': {},
    }

    for grid_idx, present in enumerate(presents):
        # E4 adjacency (always present)
        if 'E4' in present:
            for pos1, pos2 in present['E4']:
                gpos1 = (grid_idx, pos1)
                gpos2 = (grid_idx, pos2)
                adjacency_edges['E4'].add((gpos1, gpos2))
                adjacency_edges['E4'].add((gpos2, gpos1))

        # E8 adjacency (optional escalation)
        if 'E8' in present:
            for pos1, pos2 in present['E8']:
                gpos1 = (grid_idx, pos1)
                gpos2 = (grid_idx, pos2)
                adjacency_edges['E8'].add((gpos1, gpos2))
                adjacency_edges['E8'].add((gpos2, gpos1))

        # Equivalence relations: map each position to ALL members of its class
        # Per math_spec line 70-71: "as bags of current WL colors"

        # sameComp8
        if 'sameComp8' in present:
            comp_partition = present['sameComp8']
            # Group by equivalence class
            classes: Dict[int, List[Position]] = {}
            for pos, cls_id in comp_partition.items():
                if cls_id not in classes:
                    classes[cls_id] = []
                classes[cls_id].append(pos)
            # Map each position to all members of its class
            for cls_id, positions in classes.items():
                for pos in positions:
                    gpos = (grid_idx, pos)
                    # Store all class members (including self) for bag computation
                    equivalence_classes['sameComp8'][gpos] = [
                        (grid_idx, p) for p in positions
                    ]

        # bandRow
        if 'bandRow' in present:
            band_partition = present['bandRow']
            classes: Dict[int, List[Position]] = {}
            for pos, cls_id in band_partition.items():
                if cls_id not in classes:
                    classes[cls_id] = []
                classes[cls_id].append(pos)
            for cls_id, positions in classes.items():
                for pos in positions:
                    gpos = (grid_idx, pos)
                    equivalence_classes['bandRow'][gpos] = [
                        (grid_idx, p) for p in positions
                    ]

        # bandCol
        if 'bandCol' in present:
            band_partition = present['bandCol']
            classes: Dict[int, List[Position]] = {}
            for pos, cls_id in band_partition.items():
                if cls_id not in classes:
                    classes[cls_id] = []
                classes[cls_id].append(pos)
            for cls_id, positions in classes.items():
                for pos in positions:
                    gpos = (grid_idx, pos)
                    equivalence_classes['bandCol'][gpos] = [
                        (grid_idx, p) for p in positions
                    ]

    return adjacency_edges, equivalence_classes


def _refine_to_fixed_point(
    coloring: Dict[GlobalPosition, int],
    adjacency_edges: Dict[str, Set[Tuple[GlobalPosition, GlobalPosition]]],
    equivalence_classes: Dict[str, Dict[GlobalPosition, List[GlobalPosition]]],
    max_iters: int = 50,
    debug_positions: List[GlobalPosition] = None
) -> Tuple[Dict[GlobalPosition, int], int]:
    """
    Refine coloring using WL until fixed point.

    Per math_spec lines 46-51:
    - Adjacency (E4/E8): small local neighborhoods → AdjBag
    - Equivalences (sameComp8, bandRow, bandCol): global class summaries → bags

    At each iteration, signature is:
    hash(current_color, adj_bag, comp_bag, bandRow_bag, bandCol_bag)

    Args:
        coloring: Initial coloring
        adjacency_edges: Edge sets for E4/E8
        equivalence_classes: Class membership maps for sameComp8, bandRow, bandCol
        max_iters: Maximum iterations (default 50)
        debug_positions: Optional list of positions to trace

    Returns:
        (Final coloring after fixed point, iteration count)
    """
    # Build adjacency neighbor lists for E4/E8
    adjacency_neighbors: Dict[str, Dict[GlobalPosition, List[GlobalPosition]]] = {}

    for relation_name, edges in adjacency_edges.items():
        neighbors: Dict[GlobalPosition, List[GlobalPosition]] = {}
        for pos in coloring.keys():
            neighbors[pos] = []

        for pos1, pos2 in edges:
            if pos1 in neighbors:
                neighbors[pos1].append(pos2)

        # Ensure deterministic neighbor order
        for pos in neighbors:
            neighbors[pos] = sorted(neighbors[pos])

        adjacency_neighbors[relation_name] = neighbors

    # Debug tracking
    if debug_positions:
        print("\n=== WL Refinement Trace ===")
        print(f"Tracking positions: {debug_positions}\n")
        for pos in debug_positions:
            if pos in coloring:
                print(f"Initial: {pos} → color {coloring[pos]}")
        print()

    # Iterate refinement
    final_iteration = 0
    for iteration in range(max_iters):
        final_iteration = iteration
        new_coloring: Dict[GlobalPosition, int] = {}
        color_signatures: Dict[Any, List[GlobalPosition]] = {}

        # Compute new signature for each position
        for pos in sorted(coloring.keys()):
            current_color = coloring[pos]

            # Per math_spec line 50: hash(CBC, phases?, AdjBag, CompBag)
            # But CBC and phases are already in current_color from init
            # So we compute: hash(current_color, adj_bag, comp_bag, bandRow_bag, bandCol_bag)

            # 1. Adjacency bags (E4/E8) - local neighborhoods
            adj_bag = []
            for rel_name in sorted(adjacency_neighbors.keys()):
                neighbors = adjacency_neighbors[rel_name].get(pos, [])
                neighbor_colors = [coloring[nbr] for nbr in neighbors]
                neighbor_colors.sort()
                if neighbor_colors:  # Only include non-empty
                    adj_bag.append((rel_name, tuple(neighbor_colors)))

            # 2. Equivalence bags (sameComp8, bandRow, bandCol) - global class summaries
            equiv_bag = []
            for rel_name in sorted(equivalence_classes.keys()):
                class_members = equivalence_classes[rel_name].get(pos, [])
                if class_members:
                    # Bag of ALL WL colors in this equivalence class
                    class_colors = [coloring[member] for member in class_members]
                    class_colors.sort()
                    equiv_bag.append((rel_name, tuple(class_colors)))

            # Signature: (current_color, adjacency_bags, equivalence_bags)
            signature = (current_color, tuple(adj_bag), tuple(equiv_bag))
            sig_hash = stable_hash64(signature)

            if sig_hash not in color_signatures:
                color_signatures[sig_hash] = []
            color_signatures[sig_hash].append(pos)

            # Debug trace
            if debug_positions and pos in debug_positions:
                print(f"Iter {iteration}: {pos}")
                print(f"  Current color: {current_color}")
                for rel_name, multiset in adj_bag:
                    print(f"  Adj-{rel_name}: {multiset[:3]}{'...' if len(multiset) > 3 else ''}")
                for rel_name, multiset in equiv_bag:
                    print(f"  Equiv-{rel_name}: {len(multiset)} members")
                print(f"  Signature hash: {sig_hash}")

        # Assign new colors based on signatures
        # Use signature hash directly as color (stable across unions)
        for sig_hash in sorted(color_signatures.keys()):
            for pos in sorted(color_signatures[sig_hash]):
                new_coloring[pos] = sig_hash

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

    return coloring, final_iteration


def _split_into_grids(
    coloring: Dict[GlobalPosition, int],
    presents: List[Present]
) -> List[Partition]:
    """
    Split global coloring back into per-grid partitions.

    IMPORTANT: Does NOT relabel WL colors. Returns the stable_hash64 values
    directly as role IDs. This ensures role IDs are stable across different
    disjoint unions (e.g., [train1, train2, train3] vs [train1, train2, train3, test]).

    Args:
        coloring: Global coloring (WL color hashes)
        presents: List of Present dicts (to get positions)

    Returns:
        List of partitions, one per grid, with WL color hashes as role IDs
    """
    partitions: List[Partition] = []
    for grid_idx, present in enumerate(presents):
        # Get all positions for this grid
        positions = list(present['sameComp8'].keys())

        # Extract WL colors directly (no relabeling)
        partition: Partition = {}
        for pos in positions:
            gpos = (grid_idx, pos)
            wl_color_hash = coloring[gpos]
            # Use WL color hash directly as role ID (stable across unions)
            partition[pos] = wl_color_hash

        partitions.append(partition)

    return partitions


def _wl_2_disjoint_union(
    presents: List[Present],
    debug_positions: List[GlobalPosition] = None
) -> Tuple[List[Partition], int]:
    """
    Run 2-WL on disjoint union.

    2-WL maintains colors for ordered pairs (u, v) instead of individual nodes.
    This provides more discriminative power than 1-WL.

    Algorithm:
    1. Initial coloring for pairs based on individual node atoms
    2. Refine by hashing pair-color + multisets from neighboring pairs
    3. Derive node coloring from stabilized pair coloring

    Args:
        presents: List of Present dicts
        debug_positions: Optional debug positions

    Returns:
        (List of partitions with 2-WL refined colors, iteration count)
    """
    if not presents:
        return [], 0

    # Build 1-WL initial coloring for nodes
    node_coloring = _build_initial_coloring(presents)
    adjacency_edges, equivalence_classes = _build_relation_data(presents)

    # Get all positions
    all_positions = list(node_coloring.keys())

    # Build initial pair coloring: hash(color[u], color[v])
    pair_coloring: Dict[Tuple[GlobalPosition, GlobalPosition], int] = {}

    for u in all_positions:
        for v in all_positions:
            # Only pairs within same grid
            if u[0] == v[0]:  # same grid_idx
                pair_color = stable_hash64((node_coloring[u], node_coloring[v]))
                pair_coloring[(u, v)] = pair_color

    # Refine pairs to fixed point
    final_iteration = 0
    for iteration in range(50):
        final_iteration = iteration
        new_pair_coloring = {}
        changed = False

        for (u, v), current_pair_color in sorted(pair_coloring.items()):
            # Collect multiset of neighboring pair colors
            # Neighbors of (u,v): pairs sharing u or v
            neighbor_pairs_multiset = []

            # Pairs (u, w) for all w != v in same grid
            for w in all_positions:
                if w[0] == u[0] and w != v:  # same grid as u, not v
                    if (u, w) in pair_coloring:
                        neighbor_pairs_multiset.append(pair_coloring[(u, w)])

            # Pairs (w, v) for all w != u in same grid
            for w in all_positions:
                if w[0] == v[0] and w != u:  # same grid as v, not u
                    if (w, v) in pair_coloring:
                        neighbor_pairs_multiset.append(pair_coloring[(w, v)])

            neighbor_pairs_multiset.sort()

            # New pair color
            new_color = stable_hash64((current_pair_color, tuple(neighbor_pairs_multiset)))
            new_pair_coloring[(u, v)] = new_color

            if new_color != current_pair_color:
                changed = True

        pair_coloring = new_pair_coloring

        if not changed:
            break

    # Derive node coloring from pair coloring
    # Node color = hash of multiset of its pair colors
    final_node_coloring: Dict[GlobalPosition, int] = {}

    for u in all_positions:
        # Collect all pair colors involving u
        pair_colors_for_u = []
        for v in all_positions:
            if v[0] == u[0]:  # same grid
                if (u, v) in pair_coloring:
                    pair_colors_for_u.append(pair_coloring[(u, v)])

        pair_colors_for_u.sort()
        final_node_coloring[u] = stable_hash64(tuple(pair_colors_for_u))

    # Split back into per-grid partitions
    partitions = _split_into_grids(final_node_coloring, presents)

    return partitions, final_iteration


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
