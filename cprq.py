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
from present import build_present, Present
from wl import wl_disjoint_union
from equiv import is_refinement, relabel_stable
from stable import stable_hash64


# Type aliases
Position = Tuple[int, int]
Partition = Dict[Position, int]
Witness = Dict[str, Any]
CompileResult = Tuple[List[Partition], Dict[int, int], Dict[str, bool]]


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

    # Try compilation with escalation ladder
    escalation_ladder = [
        {},  # Base: always-on relations only
        {'E8': True},  # Escalation 1: add E8
        {'CBC1': True},  # Escalation 2: add CBC1
        {'CBC2': True},  # Escalation 3: add CBC2
    ]

    for opts in escalation_ladder:
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

    # Check label-constant property
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

    # Label-constant satisfied!
    # Build ρ lookup table
    rho = _build_rho(trains, Psi_list)

    # Verify training examples bit-exact
    for train_idx, (X, Y) in enumerate(trains):
        Psi = Psi_list[train_idx]

        # Predict using ρ
        Y_pred = _predict_with_rho(X, Psi, rho)

        # Check bit-exact equality
        if not _grids_equal(Y, Y_pred):
            # Training verification failed
            witness = _build_training_witness(train_idx, X, Y, Y_pred, Psi, rho, opts)
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


def _build_rho(
    trains: List[Tuple[Grid, Grid]],
    Psi_list: List[Partition]
) -> Dict[int, int]:
    """
    Build ρ lookup table: role_id -> output_color.

    For each role, assign the most common output label across training examples.

    Args:
        trains: Training pairs
        Psi_list: Role partitions for each training input

    Returns:
        Dict mapping role_id to output color
    """
    # Collect votes: role_id -> {color: count}
    role_votes: Dict[int, Dict[int, int]] = {}

    for train_idx, (X, Y) in enumerate(trains):
        Psi = Psi_list[train_idx]

        for pos in Psi.keys():
            r, c = pos
            role_id = Psi[pos]
            output_color = Y[r][c]

            if role_id not in role_votes:
                role_votes[role_id] = {}

            if output_color not in role_votes[role_id]:
                role_votes[role_id][output_color] = 0

            role_votes[role_id][output_color] += 1

    # For each role, pick most common color (deterministic tie-break)
    rho: Dict[int, int] = {}

    for role_id in sorted(role_votes.keys()):
        votes = role_votes[role_id]

        # Sort by count (descending), then by color (ascending) for determinism
        sorted_votes = sorted(votes.items(), key=lambda x: (-x[1], x[0]))

        # Pick winner
        rho[role_id] = sorted_votes[0][0]

    return rho


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
