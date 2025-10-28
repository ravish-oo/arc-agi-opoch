# WL Fix Summary ✅

**Date:** 2025-10-27
**Status:** RESOLVED - All 5 golden tasks now pass!

## The Problem

WL was merging positions with **DIFFERENT input colors** into the same role, despite having `color_tag = X[r][c]` in the atom seed.

### Evidence from Task 0d3d703e

After WL on disjoint union, **Role 3** contained:
- Train 0, pos (1,0): **X=5** → Y=1
- Train 1, pos (1,0): **X=2** → Y=6

**Different input colors (5 vs 2) merged into same role!**

## Investigation

### Initial Diagnosis (Incorrect)

First suspected that WL refinement was merging colors during iterations. Debug trace showed:
- Initial atoms ARE different:
  - Grid 0, (1,0): `atom = (5, 0, 0, 0, 0, True)`, hash = `11235818908776529389`
  - Grid 1, (1,0): `atom = (2, 0, 0, 0, 0, True)`, hash = `12703030559520168764`
- WL refinement kept them separate for all 50 iterations
- At iteration 49: Grid 0 had color **9**, Grid 1 had color **8** (DIFFERENT!)
- But final roles: Both had role **3** (MERGED!)

### Root Cause (Correct)

The bug was **NOT** in WL refinement itself - it was in `_split_into_grids` function at wl.py:301.

**BEFORE (WRONG):**
```python
def _split_into_grids(coloring, presents):
    partitions = []
    for grid_idx, present in enumerate(presents):
        positions = list(present['sameComp8'].keys())
        partition = {}
        for pos in positions:
            gpos = (grid_idx, pos)
            partition[pos] = coloring[gpos]

        # BUG: Per-grid relabeling destroys WL alignment!
        partition = relabel_stable(partition)
        partitions.append(partition)

    return partitions
```

The function was relabeling roles **independently for each grid**, which destroyed the alignment WL established.

Example:
- Grid 0 WL colors: {(0,0): 5, (1,0): 9, (2,0): 12}
- Grid 1 WL colors: {(0,0): 8, (1,0): 9, (2,0): 15}

After per-grid relabeling:
- Grid 0: {(0,0): 0, (1,0): 1, (2,0): 2}  (sorted order: 5→0, 9→1, 12→2)
- Grid 1: {(0,0): 0, (1,0): 1, (2,0): 2}  (sorted order: 8→0, 9→1, 15→2)

So:
- WL color 5 (Grid 0) → role 0
- WL color 8 (Grid 1) → role 0
- **Different WL colors merged to same role!**

## The Fix

Use **GLOBAL relabeling** that maps WL colors to role IDs consistently across all grids.

**AFTER (CORRECT):**
```python
def _split_into_grids(coloring, presents):
    # Collect all unique WL colors across all grids
    all_wl_colors = sorted(set(coloring.values()))

    # Create global mapping: WL color → role ID
    wl_color_to_role = {wl_color: role_id for role_id, wl_color in enumerate(all_wl_colors)}

    # Apply global mapping to each grid
    partitions = []
    for grid_idx, present in enumerate(presents):
        positions = list(present['sameComp8'].keys())
        partition = {}
        for pos in positions:
            gpos = (grid_idx, pos)
            wl_color = coloring[gpos]
            partition[pos] = wl_color_to_role[wl_color]
        partitions.append(partition)

    return partitions
```

Now:
- WL color 5 → role 5 (globally)
- WL color 8 → role 8 (globally)
- WL color 9 → role 9 (globally)
- **Different WL colors stay separate!**

## Results

### Before Fix
```
Task 0d3d703e:
Role 0: X=[2, 3, 5, 9], Y=[1, 4, 6, 8] ✗ CONFLICT
```

### After Fix
```
Task 0d3d703e:
Role 0: X=[8], Y=[9] ✓ OK
Role 1: X=[8], Y=[9] ✓ OK
Role 2: X=[3], Y=[4] ✓ OK
Role 3: X=[5], Y=[1] ✓ OK
Role 4: X=[2], Y=[6] ✓ OK
...
Total conflicting roles: 0 ✓
```

### Golden Tasks
All 5 golden tasks now pass:
- ✓ 007bbfb7: SUCCESS
- ✓ 0d3d703e: SUCCESS
- ✓ 25ff71a9: SUCCESS
- ✓ 3c9b0459: SUCCESS
- ✓ a2fd1cf0: SUCCESS

### Unit Tests
- 161/162 tests pass (1 unrelated fixture error)
- All WL tests pass (23/23)
- All CPRQ tests pass
- All core functionality verified

## Key Lesson

WL on disjoint union is only half the story. The **splitting phase** must preserve the alignment by using a global relabeling scheme. Per-grid relabeling breaks the core CPRQ property that "positions with same role across examples have same ID."
