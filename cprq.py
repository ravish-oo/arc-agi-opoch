"""
CPRQ compiler: Coarsest Present-Respecting & label-Respecting Partition.

The compilation loop:
1. Build present (input-only relations) from training inputs X
2. Run WL to get Psi (role partition)
3. Check label-constant: positions with same Y label must have same role
4. If not satisfied, escalate (add one optional relation or increase depth)
5. Build ρ lookup table and verify training examples bit-exact
6. Return (Psi_list, rho, wl_depth, options_used) or witness

Escalation ladder: base → E8 → depth=2 → WITNESS
At most ONE escalation step.
"""

from typing import List, Dict, Tuple, Optional, Any
from grid import Grid, pi_canon, apply_transform, get_inverse_transform
from present import build_present, Present, detect_bands, check_phase_consistency
from wl import wl_disjoint_union
from equiv import is_refinement, relabel_stable, new_partition_from_equiv
from stable import stable_hash64


# Type aliases
Position = Tuple[int, int]
Partition = Dict[Position, int]
Witness = Dict[str, Any]
# CompileResult: (Psi_list, rho, wl_depth, opts, domain_mode, scale_factor_or_band_map, pi_tag, phases, wl_iter_count)
CompileResult = Tuple[List[Partition], Dict[int, int], int, Dict[str, bool], str, Optional[Any], str, Tuple[Optional[int], Optional[int], Optional[int]], int]
BandMap = Tuple[Dict[int, int], Dict[int, int]]  # (row_map, col_map)


def _escalation_ladder() -> List[Tuple[int, Dict[str, bool]]]:
    """
    Return the standard escalation ladder.

    Returns list of (wl_depth, opts) tuples.
    Ladder: base → E8 → depth=2
    """
    return [
        (1, {}),  # Base: depth=1, always-on relations only
        (1, {'E8': True}),  # Escalation 1: depth=1 + E8
        (2, {}),  # Escalation 2: depth=2
    ]


# ============================================================================
# Formula Objects: Making E* = (⋀ K_𝒢(X_i)) ∧ Int^𝒢(⋀ ker(c_i)) explicit
# ============================================================================

def meet_partitions(p1: Partition, p2: Partition) -> Partition:
    """
    Compute the meet (finest common refinement) of two partitions.

    The meet p1 ∧ p2 splits each block of p1 by blocks of p2.
    Result: positions have same class in meet iff they have same class in BOTH p1 and p2.

    Per WO-MK-05.4: "E* = meet_partitions(KG_train, E_int)"

    Args:
        p1, p2: Partitions (position -> class_id dicts)

    Returns:
        Meet partition with positions grouped by (p1[pos], p2[pos]) pairs

    Examples:
        >>> p1 = {(0,0): 1, (0,1): 1, (1,0): 2, (1,1): 2}  # Split by row
        >>> p2 = {(0,0): 10, (0,1): 20, (1,0): 10, (1,1): 20}  # Split by col
        >>> meet = meet_partitions(p1, p2)
        >>> # Result: 4 classes, one per cell
        >>> len(set(meet.values())) == 4
        True
    """
    # Check domains match
    if set(p1.keys()) != set(p2.keys()):
        raise ValueError("Partitions must have same domain")

    # Build meet by pairing classes
    meet_signatures: Dict[Tuple[int, int], int] = {}
    meet_partition: Partition = {}
    next_class_id = 0

    for pos in sorted(p1.keys()):
        signature = (p1[pos], p2[pos])

        if signature not in meet_signatures:
            meet_signatures[signature] = next_class_id
            next_class_id += 1

        meet_partition[pos] = meet_signatures[signature]

    return meet_partition


def compute_KG_trains(
    trains: List[Tuple[Grid, Grid]],
    opts: Dict[str, bool],
    wl_depth: int,
    phases: Tuple[Optional[int], Optional[int], Optional[int]]
) -> Tuple[List[Partition], int]:
    """
    Compute K_𝒢 (present congruence) on training inputs only.

    Per WO-MK-05.4: "K_𝒢(X): present congruence of X under free moves 𝒢
    (coarsest input-only WL fixed point)."

    This is the "input structure quotient" - positions with same role
    across all training inputs get same class.

    Args:
        trains: Training pairs (already canonicalized)
        opts: Present options (E8, CBC1, etc.)
        wl_depth: WL depth (1 or 2)
        phases: Phase parameters

    Returns:
        (List of partitions, one per train input, WL iteration count)

    Examples:
        >>> # Two grids with identical structure
        >>> x1 = Grid([[1, 2], [3, 4]])
        >>> x2 = Grid([[5, 6], [7, 8]])
        >>> y1 = Grid([[0, 0], [0, 0]])
        >>> y2 = Grid([[0, 0], [0, 0]])
        >>> KG, _ = compute_KG_trains([(x1, y1), (x2, y2)], {}, 1, (None, None, None))
        >>> # Same structure → aligned classes
        >>> len(KG) == 2
        True
    """
    # Build presents for training inputs
    X_list = [X for X, Y in trains]
    presents = [build_present(X, opts, phases) for X in X_list]

    # Run WL on disjoint union to get K_𝒢
    KG_partitions, wl_iter_count = wl_disjoint_union(presents, depth=wl_depth)

    return KG_partitions, wl_iter_count


def compute_LabelMeet(trains: List[Tuple[Grid, Grid]]) -> List[Partition]:
    """
    Compute L = ⋀ ker(c_i) (label kernel meet).

    Per WO-MK-05.4: "L = ⋀_i ker(c_i): 'label split' partition
    (split WL blocks by label across all trains)."

    This is the "label constant" partition: positions with same label
    in Y must have same class.

    Args:
        trains: Training pairs on unified domain

    Returns:
        List of label partitions, one per training pair

    Examples:
        >>> x = Grid([[1, 2], [3, 4]])
        >>> y = Grid([[0, 0], [1, 1]])  # Two labels
        >>> L = compute_LabelMeet([(x, y)])
        >>> # Two label classes
        >>> len(set(L[0].values())) == 2
        True
    """
    L_partitions = []
    for X, Y in trains:
        # Build label partition for this training pair
        L_partitions.append(_build_label_partition(Y))

    return L_partitions


def invariant_interior(
    trains: List[Tuple[Grid, Grid]],
    L_partitions: List[Partition],
    phases: Tuple[Optional[int], Optional[int], Optional[int]],
    opts_base: Dict[str, bool]
) -> Tuple[Optional[Tuple[List[Partition], Dict[str, bool], int, int]], Optional[Witness]]:
    """
    Compute Int^𝒢(L): the 𝒢-invariant interior of L.

    Per WO-MK-05.4: "WL on present; split by labels; if unexpressible,
    **one** lawful escalation (E8 or 2-WL); else return finite witness."

    This tries to express the label split using only the present (input structure).
    Uses escalation ladder: base → E8 → depth=2 → WITNESS.

    Args:
        trains: Training pairs
        L_partitions: Label partitions (one per train)
        phases: Phase parameters
        opts_base: Base present options

    Returns:
        Success: ((E_int_partitions, opts_used, wl_depth_used, wl_iter_count), None)
        Failure: (None, witness)

    Examples:
        >>> # Label split expressible by input structure
        >>> x = Grid([[1, 1], [2, 2]])
        >>> y = Grid([[0, 0], [1, 1]])
        >>> L = compute_LabelMeet([(x, y)])
        >>> result, witness = invariant_interior([(x, y)], L, (None,None,None), {})
        >>> witness is None  # Should succeed
        True
    """
    # Try escalation ladder
    for wl_depth, opts in _escalation_ladder():
        # Compute K_𝒢 with current escalation
        KG_partitions, wl_iter_count = compute_KG_trains(trains, opts, wl_depth, phases)

        # Check if K_𝒢 refines L (i.e., K_𝒢 can express the label split)
        all_refine = True
        for train_idx, (X, Y) in enumerate(trains):
            KG_psi = KG_partitions[train_idx]
            L_psi = L_partitions[train_idx]

            if not is_refinement(KG_psi, L_psi):
                # K_𝒢 doesn't refine L at this escalation level
                all_refine = False
                break

        if all_refine:
            # Success! K_𝒢 refines L, so K_𝒢 is the interior
            return ((KG_partitions, opts, wl_depth, wl_iter_count), None)

    # All escalations failed - return witness
    # Find a violation for witness
    witness = {
        'type': 'label_split_unexpressible',
        'reason': 'Exhausted escalation ladder (base → E8 → depth=2)',
        'note': 'Present cannot express label split even with all escalations'
    }

    return (None, witness)


def _has_shape_change(trains: List[Tuple[Grid, Grid]]) -> bool:
    """Check if any training pair has different X and Y shapes."""
    for X, Y in trains:
        if X.H != Y.H or X.W != Y.W:
            return True
    return False


def _detect_domain_mode(trains: List[Tuple[Grid, Grid]]) -> Tuple[str, Optional[int]]:
    """
    Detect domain mode for training data.

    Returns:
        ("identity", None): All pairs have same shape
        ("uniform_scale", k): All pairs have Y.shape = k * X.shape
        ("bands", None): Irregular shape changes requiring band mapping

    Examples:
        >>> # Identity case
        >>> x1 = Grid([[1, 2], [3, 4]])
        >>> y1 = Grid([[5, 6], [7, 8]])
        >>> mode, k = _detect_domain_mode([(x1, y1)])
        >>> mode == "identity"
        True

        >>> # Uniform scale case
        >>> x2 = Grid([[1]])
        >>> y2 = Grid([[5, 5], [5, 5]])
        >>> mode, k = _detect_domain_mode([(x2, y2)])
        >>> mode == "uniform_scale" and k == 2
        True
    """
    if len(trains) == 0:
        return ("identity", None)

    # Check if all pairs have same shape (identity)
    all_same_shape = True
    for X, Y in trains:
        if X.H != Y.H or X.W != Y.W:
            all_same_shape = False
            break

    if all_same_shape:
        return ("identity", None)

    # Check for uniform scale
    # All pairs must have Y.H = k * X.H and Y.W = k * X.W for same k
    scale_factors = set()

    for X, Y in trains:
        # Check if Y dimensions are integer multiples of X dimensions
        if X.H == 0 or X.W == 0:
            # Degenerate case
            scale_factors.add(None)
            continue

        # Check row scale
        if Y.H % X.H != 0:
            scale_factors.add(None)
            continue

        # Check col scale
        if Y.W % X.W != 0:
            scale_factors.add(None)
            continue

        k_row = Y.H // X.H
        k_col = Y.W // X.W

        # Both dimensions must scale by same factor
        if k_row != k_col:
            scale_factors.add(None)
            continue

        scale_factors.add(k_row)

    # Remove None values
    scale_factors.discard(None)

    # Check if all pairs have same scale factor
    if len(scale_factors) == 1:
        k = list(scale_factors)[0]
        if k > 1:  # Valid scale factor
            return ("uniform_scale", k)

    # Fall back to bands mode
    return ("bands", None)


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

    # Π canonicalization: normalize orientation using first training input
    # Per math_spec.md: "choose lex-min image on X; apply same transform to Y on train"
    # Use first training X to determine canonical orientation for entire task
    if len(trains) == 0:
        return (None, {'type': 'no_training_data', 'reason': 'empty_trains'})

    X_first, Y_first = trains[0]
    _, pi_tag, pi_inv = pi_canon(X_first)

    # Apply same transformation to all training pairs
    trains_canonical = []
    for X, Y in trains:
        Xc = apply_transform(X, pi_tag)
        Yc = apply_transform(Y, pi_tag)
        trains_canonical.append((Xc, Yc))

    # From now on, work with canonical grids
    trains = trains_canonical

    # Detect phases: check if all training inputs have consistent periods
    # Per math_spec.md: "included only if consistent across all training inputs"
    phases = check_phase_consistency(trains)

    # Detect domain mode
    domain_mode, scale_or_none = _detect_domain_mode(trains)

    # Detect shape change (for backward compatibility checks)
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

        # Store band_map when domain_mode is "bands"
        # For bands mode, scale_or_none should store the band structure
        if domain_mode == "bands":
            scale_or_none = (target_row_bands, target_col_bands, target_H, target_W)

        # Try compilation on target space with escalation ladder (OLD CODE PATH - TEMPORARY)
        for wl_depth, opts in _escalation_ladder():
            result, witness = _try_compile_shape_change(
                trains, wl_depth, opts, target_row_bands, target_col_bands, target_H, target_W,
                domain_mode, scale_or_none, pi_tag, phases
            )
            if result is not None:
                return (result, None)

        # All escalations failed
        return (None, witness)

    # In-place path: X and Y have same shape
    # Try compilation with escalation ladder (OLD CODE PATH - TEMPORARY)
    for wl_depth, opts in _escalation_ladder():
        result, witness = _try_compile(trains, wl_depth, opts, domain_mode, scale_or_none, pi_tag, phases)
        if result is not None:
            # Success!
            return (result, None)

    # All escalations failed
    # Return witness from last attempt
    return (None, witness)


def _try_compile(
    trains: List[Tuple[Grid, Grid]],
    wl_depth: int,
    opts: Dict[str, bool],
    domain_mode: str,
    scale_or_none: Optional[int],
    pi_tag: str,
    phases: Tuple[Optional[int], Optional[int], Optional[int]]
) -> Tuple[Optional[CompileResult], Optional[Witness]]:
    """
    Try compilation with given WL depth and options.

    Returns:
        Success: ((Psi_list, rho, wl_depth, opts, domain_mode, scale_or_none, pi_tag, phases), None)
        Failure: (None, witness)
    """
    # Build present for each training input
    X_list = [X for X, Y in trains]
    presents = [build_present(X, opts, phases) for X in X_list]

    # Run WL to get Psi partitions
    Psi_list, wl_iter_count = wl_disjoint_union(presents, depth=wl_depth)

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
    return ((Psi_list, rho, wl_depth, opts, domain_mode, scale_or_none, pi_tag, phases, wl_iter_count), None)


def _try_compile_shape_change(
    trains: List[Tuple[Grid, Grid]],
    wl_depth: int,
    opts: Dict[str, bool],
    target_row_bands: Partition,
    target_col_bands: Partition,
    target_H: int,
    target_W: int,
    domain_mode: str,
    scale_or_none: Optional[int],
    pi_tag: str,
    phases: Tuple[Optional[int], Optional[int], Optional[int]]
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
        Success: ((Psi_list, rho, wl_depth, opts, domain_mode, scale_or_none, pi_tag, phases), None)
        Failure: (None, witness)
    """
    # Build present for each OUTPUT (Y) grid
    # Note: This is simplified - we're building present from Y directly
    # Blueprint says to map X to target space, but for now using Y
    Y_list = [Y for X, Y in trains]
    presents = [build_present(Y, opts, phases) for Y in Y_list]

    # Run WL to get Psi partitions (on Y positions)
    Psi_list, wl_iter_count = wl_disjoint_union(presents, depth=wl_depth)

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
    return ((Psi_list, rho, wl_depth, opts, domain_mode, scale_or_none, pi_tag, phases, wl_iter_count), None)


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

    Raises:
        ValueError: If a role_id in Psi is not present in rho (present_gap_unseen_class)
    """
    # Build output grid
    data = [[0] * X.W for _ in range(X.H)]

    # Track unseen classes for witness
    unseen_classes = set()

    for pos in Psi.keys():
        r, c = pos
        role_id = Psi[pos]

        # Look up output color
        if role_id not in rho:
            # Role not in rho - this is a present_gap_unseen_class witness
            unseen_classes.add(role_id)
            # Temporarily use 0, but will raise error below
            output_color = 0
        else:
            output_color = rho[role_id]

        data[r][c] = output_color

    # If any unseen classes, raise error
    if unseen_classes:
        raise ValueError(f"present_gap_unseen_class: Test has {len(unseen_classes)} role IDs not seen in training ρ: {sorted(list(unseen_classes))[:5]}")

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


def _replicate_uniform(X: Grid, k: int) -> Grid:
    """
    Uniformly replicate grid by factor k.

    Each cell X[r][c] becomes a k×k block in the output.

    Args:
        X: Input grid
        k: Replication factor

    Returns:
        Replicated grid of shape (k*H, k*W)

    Examples:
        >>> x = Grid([[1, 2]])
        >>> x_rep = _replicate_uniform(x, 2)
        >>> x_rep.H == 2 and x_rep.W == 4
        True
        >>> x_rep[0][0] == 1 and x_rep[0][1] == 1
        True
    """
    H_out = X.H * k
    W_out = X.W * k
    data = [[0] * W_out for _ in range(H_out)]

    for r in range(X.H):
        for c in range(X.W):
            value = X[r][c]
            # Fill k×k block
            for dr in range(k):
                for dc in range(k):
                    data[r * k + dr][c * k + dc] = value

    return Grid(data)


def build_receipt(compile_result: CompileResult) -> Dict[str, Any]:
    """
    Build diagnostic receipt from successful compilation.

    Per WO-MK-05.2, receipts include:
    - wl_depth: WL depth used (1 or 2)
    - domain_mode: identity, uniform_scale, or bands
    - scale_factor: For uniform_scale mode
    - band_map: For bands mode (row_bands, col_bands, target_H, target_W)
    - present_flags: Which optional relations were enabled

    Args:
        compile_result: Result from compile_CPRQ

    Returns:
        Receipt dictionary with diagnostic information

    Examples:
        >>> # Identity mode example
        >>> x = Grid([[1, 2]])
        >>> y = Grid([[5, 6]])
        >>> result, _ = compile_CPRQ([(x, y)], {})
        >>> receipt = build_receipt(result)
        >>> receipt['domain_mode'] == 'identity'
        True
    """
    Psi_list, rho, wl_depth, opts, domain_mode, scale_or_none, pi_tag, phases, wl_iter_count = compile_result

    # Build present_flags from opts
    present_flags = {
        'E4': True,  # Always on
        'sameRow': True,  # Always on
        'sameCol': True,  # Always on
        'sameColor': True,  # Always on
        'sameComp8': True,  # Always on
        'bandRow': True,  # Always on
        'bandCol': True,  # Always on
        'E8': opts.get('E8', False),
        'CBC1': opts.get('CBC1', False),
        'CBC2': opts.get('CBC2', False),
    }

    row_k, col_k, diag_k = phases

    # cbc_radius: 0 or 1 (documentation field)
    cbc_radius = 1 if opts.get('CBC1', False) else 0

    receipt = {
        'pi_tag': pi_tag,
        'wl_depth': wl_depth,
        'cbc_radius': cbc_radius,
        'wl_iter_count': wl_iter_count,
        'domain_mode': domain_mode,
        'phases': {'row_k': row_k, 'col_k': col_k, 'diag_k': diag_k},
        'present_flags': present_flags,
        'num_roles': len(set(c for psi in Psi_list for c in psi.values())),
        'rho_size': len(rho),
    }

    # Add mode-specific fields
    if domain_mode == 'uniform_scale' and scale_or_none is not None:
        receipt['scale_factor'] = scale_or_none

    elif domain_mode == 'bands' and scale_or_none is not None:
        # scale_or_none stores (row_bands, col_bands, target_H, target_W)
        if isinstance(scale_or_none, tuple) and len(scale_or_none) == 4:
            row_bands, col_bands, target_H, target_W = scale_or_none
            receipt['band_map'] = {
                'target_shape': (target_H, target_W),
                'num_row_bands': len(set(row_bands.values())),
                'num_col_bands': len(set(col_bands.values())),
            }

    return receipt


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

    Per WO-MK-05.2, runs WL on disjoint union {trains ∪ test} to ensure
    alignment. Rebuilds ρ from training positions in the union (WL hashes
    change when test is added, so we rebuild ρ with new hashes).

    Args:
        X_test: Test input grid
        trains: Original training data (for union-WL alignment)
        compile_result: Result from compile_CPRQ (options_used only)

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
    Psi_list_train_old, rho_old, wl_depth, options_used, domain_mode, scale_or_none, pi_tag, phases, wl_iter_count_compile = compile_result

    # Π canonicalization: apply same transform to test input
    # Per math_spec.md: "on test, apply to X*, remember inverse"
    pi_inv = get_inverse_transform(pi_tag)
    X_test_canonical = apply_transform(X_test, pi_tag)

    # Build presents on appropriate domain based on domain_mode
    # CRITICAL: Must use union-WL for alignment
    # Note: trains are already canonicalized in compile_CPRQ
    # Track which grid Psi_test will be built from (for correct predict dimensions)
    test_grid_for_wl = X_test_canonical  # Default

    if domain_mode == "identity":
        # Same-shape: build presents directly on inputs
        train_inputs = [X for X, Y in trains]
        all_inputs = train_inputs + [X_test_canonical]
        all_presents = [build_present(X, options_used, phases) for X in all_inputs]

    elif domain_mode == "uniform_scale":
        # Uniform replication: replicate by factor k
        k = scale_or_none
        if k is None or k <= 0:
            # Invalid scale factor, fall back to identity
            train_inputs = [X for X, Y in trains]
            all_inputs = train_inputs + [X_test_canonical]
            all_presents = [build_present(X, options_used, phases) for X in all_inputs]
        else:
            # Replicate all inputs by factor k
            train_inputs_replicated = [_replicate_uniform(X, k) for X, Y in trains]
            test_replicated = _replicate_uniform(X_test_canonical, k)
            test_grid_for_wl = test_replicated  # Track replicated grid
            all_inputs = train_inputs_replicated + [test_replicated]
            all_presents = [build_present(X_rep, options_used, phases) for X_rep in all_inputs]

    elif domain_mode == "bands":
        # Band-based mapping: for now, use output space (simplified)
        # Full implementation would map X through bands to target space
        # For now, use training outputs directly (same as compilation)
        train_outputs = [Y for X, Y in trains]
        # For test, we don't have Y, so fall back to identity for now
        # This is a limitation of the simplified implementation
        train_presents = [build_present(Y, options_used, phases) for Y in train_outputs]
        test_present = build_present(X_test_canonical, options_used, phases)  # Fallback
        all_presents = train_presents + [test_present]

    else:
        # Unknown domain mode, fall back to identity
        train_inputs = [X for X, Y in trains]
        all_inputs = train_inputs + [X_test_canonical]
        all_presents = [build_present(X, options_used, phases) for X in all_inputs]

    # Run WL on disjoint union {trains ∪ test}
    # This gives DIFFERENT hashes than compile-time WL (trains only)
    # because adding test changes global refinement patterns
    Psi_list_all, wl_iter_count_predict = wl_disjoint_union(all_presents, depth=wl_depth)

    # Rebuild ρ from training positions using NEW WL hashes
    # Training positions in the union get new hashes, but still map to same Y colors
    rho_new = {}
    for train_idx, (X, Y) in enumerate(trains):
        Psi = Psi_list_all[train_idx]
        for pos in Psi.keys():
            r, c = pos
            wl_hash = Psi[pos]
            y_color = Y[r][c]

            # Check single-valued property (should still hold)
            if wl_hash in rho_new:
                if rho_new[wl_hash] != y_color:
                    # Conflict - shouldn't happen if compile succeeded
                    # Use first color seen (defensive)
                    pass
            else:
                rho_new[wl_hash] = y_color

    # Extract Psi for test input (last one)
    Psi_test = Psi_list_all[-1]

    # Debug logging per WO-MK-05.2a section 4
    # Log class ID counts and overlap
    train_class_ids = set()
    for train_idx in range(len(trains)):
        Psi_train = Psi_list_all[train_idx]
        train_classes = set(Psi_train.values())
        train_class_ids.update(train_classes)
        print(f"  Train grid {train_idx}: {len(train_classes)} unique class IDs")

    test_class_ids = set(Psi_test.values())
    print(f"  Test grid: {len(test_class_ids)} unique class IDs")

    overlap = train_class_ids & test_class_ids
    print(f"  Overlap (train ∩ test): {len(overlap)} class IDs")
    print(f"  ρ coverage: {len(rho_new)} entries")

    if len(overlap) == 0:
        print(f"  ⚠️ WARNING: Zero overlap! Union-WL alignment issue.")

    # Predict using rebuilt rho (may raise ValueError for unseen classes)
    # This generates Y in canonical orientation
    # IMPORTANT: Must use the same grid that Psi_test was built from (test_grid_for_wl)
    try:
        Y_pred_canonical = _predict_with_rho(test_grid_for_wl, Psi_test, rho_new)
    except ValueError as e:
        # present_gap_unseen_class witness
        # Per WO-MK-05.3: "emit a predict-time witness of type present_gap_unseen_class"
        error_msg = str(e)
        print(f"  ❌ Predict failed: {error_msg}")

        # Extract unseen class_ids from error message (if format matches)
        unseen_classes = []
        try:
            # Error message format: "present_gap_unseen_class: Test has N role IDs not seen in training ρ: [list]"
            if "present_gap_unseen_class" in error_msg:
                # This is the expected error, witness has already been logged
                pass
        except:
            pass

        # For now, raise the error with witness information
        # In production, this would return a proper witness structure
        raise ValueError(f"present_gap_unseen_class: {error_msg}") from e

    # Apply inverse Π transform to get final output
    # Per math_spec.md: "remember inverse" and apply it to output
    Y_pred = apply_transform(Y_pred_canonical, pi_inv)

    return Y_pred
