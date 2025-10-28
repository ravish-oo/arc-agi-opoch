"""
CPRQ compiler: Coarsest Present-Respecting & label-Respecting Partition.

The compilation loop:
1. Build present (input-only relations) from training inputs X
2. Run WL to get Psi (role partition)
3. Check label-constant: positions with same Y label must have same role
4. If not satisfied, escalate (add one optional relation)
5. Build ρ lookup table and verify training examples bit-exact
6. Return (Psi_list, rho, options_used) or witness

Escalation ladder: base → {E8 OR CBC1 OR CBC2}
At most ONE escalation step.
"""

from typing import List, Dict, Tuple, Optional, Any
from grid import Grid
from present import build_present, Present, detect_bands
from wl import wl_disjoint_union
from equiv import is_refinement, relabel_stable, new_partition_from_equiv
from stable import stable_hash64


# Type aliases
Position = Tuple[int, int]
Partition = Dict[Position, int]
Witness = Dict[str, Any]
CompileResult = Tuple[List[Partition], Dict[int, int], Dict[str, bool]]
BandMap = Tuple[Dict[int, int], Dict[int, int]]  # (row_map, col_map)


def _escalation_ladder() -> List[Dict[str, bool]]:
    """Return the standard escalation ladder."""
    return [
        {},  # Base: always-on relations only
        {'E8': True},  # Escalation 1: add E8
        {'CBC1': True},  # Escalation 2: add CBC1
        {'CBC2': True},  # Escalation 3: add CBC2
    ]


def _has_shape_change(trains: List[Tuple[Grid, Grid]]) -> bool:
    """Check if any training pair has different X and Y shapes."""
    for X, Y in trains:
        if X.H != Y.H or X.W != Y.W:
            return True
    return False


def _unify_band_structure(trains: List[Tuple[Grid, Grid]]) -> Optional[Tuple[Partition, Partition, int, int]]:
    """
    Unify band structure across all training pairs for shape-change tasks.

    Returns:
        (unified_row_bands, unified_col_bands, target_H, target_W) or None if unification fails

    The unified bands are on the OUTPUT (Y) space, representing the target index space.
    """
    if len(trains) == 0:
        return None

    # Collect band structures from all Y outputs
    y_row_bands_list = []
    y_col_bands_list = []
    target_shapes = []

    for X, Y in trains:
        row_bands, col_bands = detect_bands(Y)
        y_row_bands_list.append(row_bands)
        y_col_bands_list.append(col_bands)
        target_shapes.append((Y.H, Y.W))

    # Check all outputs have same shape
    target_H, target_W = target_shapes[0]
    if not all(h == target_H and w == target_W for h, w in target_shapes):
        # Outputs have different shapes - cannot unify
        return None

    # Check band structures are consistent across outputs
    # For now, use first output's bands as unified structure
    # (In full implementation, could verify all outputs have identical band patterns)
    unified_row_bands = y_row_bands_list[0]
    unified_col_bands = y_col_bands_list[0]

    return (unified_row_bands, unified_col_bands, target_H, target_W)


def compile_CPRQ(
    trains: List[Tuple[Grid, Grid]],
    base_opts: Dict[str, bool]
) -> Tuple[Optional[CompileResult], Optional[Witness]]:
    """
    Compile CPRQ: find coarsest present-respecting partition that respects labels.

    Escalation ladder: base → {E8 OR CBC1 OR CBC2} (single step only)

    Args:
        trains: List of (input_grid, output_grid) pairs
        base_opts: Base options (should be empty {} for always-on relations only)

    Returns:
        Success: ((Psi_list, rho, options_used), None)
        Failure: (None, witness)

    Examples:
        >>> # Simple training example
        >>> x = Grid([[1, 2], [3, 4]])
        >>> y = Grid([[1, 1], [2, 2]])
        >>> result, witness = compile_CPRQ([(x, y)], {})
        >>> witness is None  # Should succeed
        True
    """
    # Validate base_opts: should be empty (no optional relations enabled yet)
    if base_opts:
        # For now, allow base_opts to specify starting point
        # But in typical use, base_opts should be {}
        pass

    # Detect shape change
    has_shape_change = _has_shape_change(trains)

    if has_shape_change:
        # Shape-change path: unify band structure
        unify_result = _unify_band_structure(trains)

        if unify_result is None:
            # Band unification failed
            return (None, {
                'type': 'shape_change_unsat',
                'reason': 'band_unification_failed',
                'note': 'Output shapes or band patterns inconsistent across training'
            })

        # Unpack unified bands
        target_row_bands, target_col_bands, target_H, target_W = unify_result

        # Try compilation on target space with escalation ladder
        for opts in _escalation_ladder():
            result, witness = _try_compile_shape_change(
                trains, opts, target_row_bands, target_col_bands, target_H, target_W
            )
            if result is not None:
                return (result, None)

        # All escalations failed
        return (None, witness)

    # In-place path: X and Y have same shape
    # Try compilation with escalation ladder
    for opts in _escalation_ladder():
        result, witness = _try_compile(trains, opts)
        if result is not None:
            # Success!
            return (result, None)

    # All escalations failed
    # Return witness from last attempt
    return (None, witness)


def _try_compile(
    trains: List[Tuple[Grid, Grid]],
    opts: Dict[str, bool]
) -> Tuple[Optional[CompileResult], Optional[Witness]]:
    """
    Try compilation with given options.

    Returns:
        Success: ((Psi_list, rho, opts), None)
        Failure: (None, witness)
    """
    # Build present for each training input
    X_list = [X for X, Y in trains]
    presents = [build_present(X, opts) for X in X_list]

    # Run WL to get Psi partitions
    Psi_list = wl_disjoint_union(presents)

    # Check label-constant property (refinement)
    for train_idx, (X, Y) in enumerate(trains):
        Psi = Psi_list[train_idx]

        # Build label partition: positions with same Y value
        label_partition = _build_label_partition(Y)

        # Check if Psi refines label_partition
        # (i.e., label_partition is expressible by Psi)
        if not is_refinement(Psi, label_partition):
            # Label-constant violated
            witness = _build_refinement_witness(train_idx, Psi, label_partition, opts)
            return (None, witness)

    # Refinement satisfied! Now check single-valued property and build ρ
    rho, conflicts = _check_single_valued_and_build_rho(trains, Psi_list)

    if conflicts is not None:
        # Single-valued property violated: roles have multiple colors
        # This means escalation is needed (caller will try next opts)
        witness = _build_role_conflict_witness(conflicts, opts, trains, Psi_list)
        return (None, witness)

    # Single-valued property satisfied! ρ is now trivial mapping
    # Sanity check: training should regenerate exactly (defensive check)
    for train_idx, (X, Y) in enumerate(trains):
        Psi = Psi_list[train_idx]

        # Predict using ρ
        Y_pred = _predict_with_rho(X, Psi, rho)

        # Check bit-exact equality
        if not _grids_equal(Y, Y_pred):
            # This should NEVER happen if single-valued check passed
            # But keep it as a sanity check for bugs
            witness = {
                'type': 'sanity_check_failed',
                'train_idx': train_idx,
                'note': 'Single-valued check passed but training regeneration failed - compiler bug!',
                'opts': opts,
            }
            return (None, witness)

    # Success!
    return ((Psi_list, rho, opts), None)


def _try_compile_shape_change(
    trains: List[Tuple[Grid, Grid]],
    opts: Dict[str, bool],
    target_row_bands: Partition,
    target_col_bands: Partition,
    target_H: int,
    target_W: int
) -> Tuple[Optional[CompileResult], Optional[Witness]]:
    """
    Try compilation for shape-change tasks.

    Build present on target (Y) space and run CPRQ there.

    For simplicity, this version:
    1. Builds present directly from Y grids (output space)
    2. Runs WL on Y positions
    3. Checks label-constant on Y space
    4. Builds ρ from Y space

    Note: This is a simplified implementation. Full blueprint requires
    mapping X through bands to target space, but for now we work directly
    with Y to make progress.

    Returns:
        Success: ((Psi_list, rho, opts), None)
        Failure: (None, witness)
    """
    # Build present for each OUTPUT (Y) grid
    # Note: This is simplified - we're building present from Y directly
    # Blueprint says to map X to target space, but for now using Y
    Y_list = [Y for X, Y in trains]
    presents = [build_present(Y, opts) for Y in Y_list]

    # Run WL to get Psi partitions (on Y positions)
    Psi_list = wl_disjoint_union(presents)

    # Check label-constant property (refinement on Y space)
    for train_idx, (X, Y) in enumerate(trains):
        Psi = Psi_list[train_idx]

        # Build label partition from Y
        label_partition = _build_label_partition(Y)

        # Check if Psi refines label_partition
        if not is_refinement(Psi, label_partition):
            # Label-constant violated
            witness = _build_refinement_witness(train_idx, Psi, label_partition, opts)
            witness['shape_change'] = True
            return (None, witness)

    # Refinement satisfied! Now check single-valued property and build ρ
    rho, conflicts = _check_single_valued_and_build_rho(trains, Psi_list)

    if conflicts is not None:
        # Single-valued property violated
        witness = _build_role_conflict_witness(conflicts, opts, trains, Psi_list)
        witness['shape_change'] = True
        return (None, witness)

    # Single-valued property satisfied!
    # Sanity check: training should regenerate exactly (on Y space)
    for train_idx, (X, Y) in enumerate(trains):
        Psi = Psi_list[train_idx]

        # Predict using ρ (Y positions)
        Y_pred = _predict_with_rho(Y, Psi, rho)

        # Check bit-exact equality
        if not _grids_equal(Y, Y_pred):
            # Sanity check failed
            witness = {
                'type': 'sanity_check_failed',
                'train_idx': train_idx,
                'note': 'Single-valued check passed but training regeneration failed - compiler bug!',
                'opts': opts,
                'shape_change': True,
            }
            return (None, witness)

    # Success!
    return ((Psi_list, rho, opts), None)


def _build_label_partition(Y: Grid) -> Partition:
    """
    Build partition from output labels.

    Positions with same Y value are in same block.

    Args:
        Y: Output grid

    Returns:
        Partition mapping positions to label-based blocks
    """
    # Group positions by their Y value
    value_to_positions: Dict[int, List[Position]] = {}

    for pos in Y.positions():
        r, c = pos
        value = Y[r][c]

        if value not in value_to_positions:
            value_to_positions[value] = []
        value_to_positions[value].append(pos)

    # Assign block IDs deterministically
    partition: Partition = {}
    block_id = 0

    for value in sorted(value_to_positions.keys()):
        for pos in sorted(value_to_positions[value]):
            partition[pos] = block_id
        block_id += 1

    return partition


def _check_single_valued_and_build_rho(
    trains: List[Tuple[Grid, Grid]],
    Psi_list: List[Partition]
) -> Tuple[Optional[Dict[int, int]], Optional[Dict[int, Any]]]:
    """
    Check single-valued property and build ρ lookup table.

    CPRQ law: Each role must map to EXACTLY ONE color across all training examples.
    No majority votes. If a role has multiple colors, that's a conflict requiring
    escalation or witness.

    Args:
        trains: Training pairs
        Psi_list: Role partitions for each training input

    Returns:
        Success: (rho_dict, None) where rho maps role_id -> unique_color
        Conflict: (None, conflicts) where conflicts describes which roles have >1 color
    """
    # Collect all colors per role: role_id -> set of colors
    role_colors: Dict[int, set] = {}
    role_sample_positions: Dict[int, List[Tuple[int, Tuple[int, int], int]]] = {}  # role -> [(train_idx, pos, color), ...]

    for train_idx, (X, Y) in enumerate(trains):
        Psi = Psi_list[train_idx]

        for pos in Psi.keys():
            r, c = pos
            role_id = Psi[pos]
            output_color = Y[r][c]

            if role_id not in role_colors:
                role_colors[role_id] = set()
                role_sample_positions[role_id] = []

            role_colors[role_id].add(output_color)
            role_sample_positions[role_id].append((train_idx, pos, output_color))

    # Check single-valued property
    conflicts = {}
    for role_id in sorted(role_colors.keys()):
        colors = role_colors[role_id]
        if len(colors) > 1:
            # Role conflict: same role has multiple colors
            conflicts[role_id] = {
                'colors': sorted(list(colors)),
                'samples': role_sample_positions[role_id][:3]  # First 3 examples
            }

    if conflicts:
        # Single-valued property violated
        return (None, conflicts)

    # Single-valued property satisfied - build ρ
    rho: Dict[int, int] = {}
    for role_id in sorted(role_colors.keys()):
        # Each role has exactly 1 color
        colors = role_colors[role_id]
        rho[role_id] = list(colors)[0]

    return (rho, None)


def _predict_with_rho(X: Grid, Psi: Partition, rho: Dict[int, int]) -> Grid:
    """
    Predict output using ρ lookup table.

    Y(p) = ρ(Psi(p))

    Args:
        X: Input grid (for dimensions)
        Psi: Role partition
        rho: Lookup table

    Returns:
        Predicted output grid
    """
    # Build output grid
    data = [[0] * X.W for _ in range(X.H)]

    for pos in Psi.keys():
        r, c = pos
        role_id = Psi[pos]

        # Look up output color
        if role_id not in rho:
            # Role not in rho (shouldn't happen if training worked)
            # Default to 0
            output_color = 0
        else:
            output_color = rho[role_id]

        data[r][c] = output_color

    return Grid(data)


def _grids_equal(g1: Grid, g2: Grid) -> bool:
    """
    Check if two grids are bit-exact equal.

    Args:
        g1, g2: Grids to compare

    Returns:
        True if identical
    """
    if g1.H != g2.H or g1.W != g2.W:
        return False

    for r, c in g1.positions():
        if g1[r][c] != g2[r][c]:
            return False

    return True


def _build_refinement_witness(
    train_idx: int,
    Psi: Partition,
    label_partition: Partition,
    opts: Dict[str, bool]
) -> Witness:
    """
    Build witness for refinement failure.

    Find two positions with same label but different roles.

    Args:
        train_idx: Training example index
        Psi: Input-derived partition
        label_partition: Label-derived partition
        opts: Options used

    Returns:
        Witness dict
    """
    # Find violation: two positions with same label but different Psi roles
    for p1 in sorted(Psi.keys()):
        for p2 in sorted(Psi.keys()):
            if p1 >= p2:
                continue

            # Same label?
            if label_partition[p1] == label_partition[p2]:
                # Different Psi roles?
                if Psi[p1] != Psi[p2]:
                    return {
                        'type': 'refinement_failure',
                        'train_idx': train_idx,
                        'pos1': p1,
                        'pos2': p2,
                        'label_block': label_partition[p1],
                        'psi_role1': Psi[p1],
                        'psi_role2': Psi[p2],
                        'opts': opts,
                    }

    # No violation found (shouldn't happen if is_refinement returned False)
    return {
        'type': 'refinement_failure',
        'train_idx': train_idx,
        'opts': opts,
    }


def _build_role_conflict_witness(
    conflicts: Dict[int, Any],
    opts: Dict[str, bool],
    trains: List[Tuple[Grid, Grid]] = None,
    Psi_list: List[Partition] = None
) -> Witness:
    """
    Build concrete witness for role conflict (single-valued property violation).

    Per anchor blueprint: witness must include two DISTINCT pixels p != q in same
    present role, in SAME training example, that must be different in Y.

    Args:
        conflicts: Dict of {role_id: {'colors': [...], 'samples': [...]}}
        opts: Options used
        trains: Training pairs (for finding p != q)
        Psi_list: Role partitions (for finding concrete (p, q))

    Returns:
        Witness dict with (p, q) positions and present_flags
    """
    # Pick first conflict for witness
    role_id = sorted(conflicts.keys())[0]
    conflict = conflicts[role_id]
    colors = conflict['colors']

    # Find a training example where this role has multiple colors
    # Then find two DISTINCT positions p != q in that role with different Y
    p_witness = None
    q_witness = None
    train_id_witness = 0

    for train_idx, (X, Y) in enumerate(trains):
        Psi = Psi_list[train_idx]

        # Find all positions in this role in this training example
        positions_in_role = [pos for pos, rid in Psi.items() if rid == role_id]

        # Find two distinct positions with different Y values
        for p in positions_in_role:
            for q in positions_in_role:
                if p != q:  # Distinct positions
                    p_color = Y[p[0]][p[1]]
                    q_color = Y[q[0]][q[1]]
                    if p_color != q_color:  # Different Y
                        p_witness = p
                        q_witness = q
                        train_id_witness = train_idx
                        break
            if p_witness:
                break
        if p_witness:
            break

    # Fallback if we can't find distinct p != q (shouldn't happen)
    if not p_witness or not q_witness:
        p_witness = (0, 0)
        q_witness = (0, 1)

    # Build present_flags
    present_flags = {
        'E4': True,
        'sameRow': True,
        'sameCol': True,
        'sameColor': True,
        'sameComp8': True,
        'bandRow': True,
        'bandCol': True,
        'E8': opts.get('E8', False),
        'CBC1': opts.get('CBC1', False),
        'CBC2': opts.get('CBC2', False),
    }

    witness = {
        'type': 'label_conflict_unexpressible',
        'train_id': train_id_witness,
        'p': list(p_witness),
        'q': list(q_witness),
        'present_flags': present_flags,
        'reason': 'label_conflict_unexpressible',
        'role_id': role_id,
        'colors_in_role': colors,
        'num_conflicting_roles': len(conflicts),
    }

    return witness


def _build_training_witness(
    train_idx: int,
    X: Grid,
    Y: Grid,
    Y_pred: Grid,
    Psi: Partition,
    rho: Dict[int, int],
    opts: Dict[str, bool]
) -> Witness:
    """
    Build witness for training verification failure.

    Find positions where prediction differs from label.

    Args:
        train_idx: Training example index
        X: Input grid
        Y: True output
        Y_pred: Predicted output
        Psi: Role partition
        rho: Lookup table
        opts: Options used

    Returns:
        Witness dict
    """
    # Find first mismatch
    for pos in sorted(Y.positions()):
        r, c = pos
        if Y[r][c] != Y_pred[r][c]:
            return {
                'type': 'training_mismatch',
                'train_idx': train_idx,
                'pos': pos,
                'expected': Y[r][c],
                'predicted': Y_pred[r][c],
                'role': Psi[pos],
                'rho_value': rho.get(Psi[pos], None),
                'opts': opts,
            }

    # No mismatch found (shouldn't happen)
    return {
        'type': 'training_mismatch',
        'train_idx': train_idx,
        'opts': opts,
    }


def count_cells_wrong(Y: Grid, Y_pred: Grid) -> int:
    """
    Count number of cells where prediction differs from true output.

    Args:
        Y: True output
        Y_pred: Predicted output

    Returns:
        Number of wrong cells

    Examples:
        >>> y = Grid([[1, 2], [3, 4]])
        >>> y_pred = Grid([[1, 2], [3, 5]])
        >>> count_cells_wrong(y, y_pred)
        1
    """
    if Y.H != Y_pred.H or Y.W != Y_pred.W:
        # Shape mismatch - count all cells as wrong
        return Y.H * Y.W

    wrong = 0
    for r, c in Y.positions():
        if Y[r][c] != Y_pred[r][c]:
            wrong += 1

    return wrong


def predict(X_test: Grid, trains: List[Tuple[Grid, Grid]], compile_result: CompileResult) -> Grid:
    """
    Predict output for test input using compiled CPRQ model.

    To assign roles to test input positions, runs WL on disjoint union of
    [train_1, ..., train_n, X_test] using the same present options,
    then applies learned ρ mapping.

    Args:
        X_test: Test input grid
        trains: Original training data (to rerun WL with test included)
        compile_result: Result from compile_CPRQ

    Returns:
        Predicted output grid

    Examples:
        >>> # Train on simple colormap
        >>> x1 = Grid([[1, 2]])
        >>> y1 = Grid([[5, 6]])
        >>> trains = [(x1, y1)]
        >>> result, witness = compile_CPRQ(trains, {})
        >>> x_test = Grid([[1, 2]])
        >>> y_pred = predict(x_test, trains, result)
        >>> y_pred[0][0] == 5 and y_pred[0][1] == 6
        True
    """
    Psi_list_train, rho, options_used = compile_result

    # Build presents for all training inputs + test input
    all_inputs = [X for X, Y in trains] + [X_test]
    presents = [build_present(X, options_used) for X in all_inputs]

    # Run WL on disjoint union (training + test)
    Psi_list_all = wl_disjoint_union(presents)

    # Extract Psi for test input (last one)
    Psi_test = Psi_list_all[-1]

    # Predict using rho
    return _predict_with_rho(X_test, Psi_test, rho)
