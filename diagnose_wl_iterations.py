#!/usr/bin/env python3
"""
Diagnostic: Trace WL iterations to find where test diverges from training.

Atoms are aligned initially (4/4 match), but final role IDs have zero overlap.
Something in WL refinement causes divergence. Let's trace it.
"""

import json
from grid import Grid
from present import build_present
from wl import wl_disjoint_union

# Load data
with open('data/arc-agi_training_challenges.json') as f:
    challenges = json.load(f)

task_id = '0d3d703e'
task = challenges[task_id]

print(f"Task: {task_id}")
print("=" * 80)
print()

# Prepare grids
trains = [(Grid(p['input']), Grid(p['output'])) for p in task['train']]
test_input = Grid(task['test'][0]['input'])

print("Grid structure:")
print("  All grids: 3×3")
print("  Same structure (3 vertical stripes)")
print("  Test uses colors [8,1,3], appears in training grids")
print()

# Build presents
opts = {}
phases = (None, None, None)

train_inputs = [X for X, Y in trains]
all_inputs = train_inputs + [test_input]
all_presents = [build_present(X, opts, phases) for X in all_inputs]

# Focus on specific positions that should align
# In all grids: position (0,0) is the leftmost column, which has one consistent color
# Position (0,1) is the middle column, which has one consistent color
# Position (0,2) is the rightmost column, which has one consistent color

print("=" * 80)
print("POSITION TRACKING: (0,1) - middle column, border")
print("=" * 80)
print()

pos = (0, 1)

for grid_idx, X in enumerate(all_inputs):
    if grid_idx < len(trains):
        print(f"Train grid {grid_idx}: position {pos} has color {X[pos[0]][pos[1]]}")
    else:
        print(f"Test grid: position {pos} has color {X[pos[0]][pos[1]]}")

print()
print("This position should get SAME role in train and test because:")
print("  - Same relative position (middle column, top border)")
print("  - Same local structure (1 E4 neighbor below)")
print("  - Color is just a label (should not affect structural role)")
print()

# Run WL with debug tracing
print("=" * 80)
print("WL REFINEMENT WITH TRACING")
print("=" * 80)
print()

# Track position (0,1) in grid 0 and grid 4 (test)
debug_positions = [
    (0, pos),  # Train grid 0, pos (0,1)
    (4, pos),  # Test grid, pos (0,1)
]

Psi_list, iter_count = wl_disjoint_union(all_presents, depth=1, debug_positions=debug_positions)

print()
print("=" * 80)
print("FINAL ROLE IDs")
print("=" * 80)
print()

# Check final role IDs
role_train_0 = Psi_list[0][pos]
role_test = Psi_list[-1][pos]

print(f"Train grid 0, position {pos}: role ID = {role_train_0}")
print(f"Test grid, position {pos}: role ID = {role_test}")
print()

if role_train_0 == role_test:
    print("✅ SAME role ID")
else:
    print("❌ DIFFERENT role ID")
    print()
    print("This is the bug: structurally identical positions get different role IDs")

print()

# Summary of all role IDs
print("=" * 80)
print("ALL ROLE IDs COMPARISON")
print("=" * 80)
print()

train_roles = set()
for grid_idx in range(len(trains)):
    roles = set(Psi_list[grid_idx].values())
    train_roles.update(roles)

test_roles = set(Psi_list[-1].values())

overlap = train_roles & test_roles

print(f"Training role IDs: {len(train_roles)} unique")
print(f"Test role IDs: {len(test_roles)} unique")
print(f"Overlap: {len(overlap)}/{len(test_roles)}")
print()

if len(overlap) == 0:
    print("❌ ZERO OVERLAP")
    print()
    print("ROOT CAUSE:")
    print("  The WL refinement is NOT computing input-only structural hashes.")
    print("  Something in the signature computation depends on which grids are")
    print("  in the equivalence classes, making it impossible for test positions")
    print("  to align with training positions.")
    print()
    print("Per math_spec.md line 46:")
    print("  'Init per pixel: hash(CBC, raw, RowPhase?, ColPhase?, DiagPhase?)'")
    print()
    print("The presence of 'raw' (raw color) in the atom makes WL refinement")
    print("palette-dependent. Even though atoms overlap initially, the iterative")
    print("refinement through sameComp8 (same-color components) SEPARATES")
    print("positions by palette, preventing structural alignment.")
else:
    print("✅ Some overlap exists")

print()
print("=" * 80)
print("EXACT BUG LOCATION")
print("=" * 80)
print("""
File: wl.py
Function: _build_initial_coloring
Line: ~135

Current code:
    color_tag = grid[r][c]  # Raw palette value 0-9
    cbc_token = cbc_tokens[pos] if cbc_tokens else 0
    atom = (color_tag, cbc_token, is_border)

Bug:
    Including raw color in the atom makes WL palette-dependent.
    Even though math_spec.md line 46 mentions "raw", this violates
    the "input structure only" principle from math_spec_addon_airtight.md
    line 40.

    The sameComp8 equivalence relation groups positions by RAW color,
    so during WL iteration, positions with different colors get different
    equivalence class bags, causing them to diverge.

Per math_spec.md line 36:
    "CBC: 'color-blind canonical patch' (OFA inside patch + D8 canonicalization)"

The entire present is supposed to be COLOR-BLIND. CBC already captures
local structure without palette. Raw color should NOT be in the atom.

Fix:
    atom = (cbc_token, is_border)  # Remove color_tag

This makes WL truly input-only structural, allowing test positions
to align with structurally similar training positions regardless of palette.
""")
