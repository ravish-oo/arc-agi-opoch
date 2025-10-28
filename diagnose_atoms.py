#!/usr/bin/env python3
"""
Diagnostic: Examine atom construction and initial WL coloring.

Why do test positions get completely different role IDs from training?
Let's look at the actual atom values (raw color, CBC, border).
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

# Show actual grid contents
print("TRAINING INPUT 0:")
X0, Y0 = trains[0]
for r in range(X0.H):
    print("  ", [X0[r][c] for c in range(X0.W)])
print()

print("TEST INPUT:")
for r in range(test_input.H):
    print("  ", [test_input[r][c] for c in range(test_input.W)])
print()

# Build presents
opts = {}  # Base options
phases = (None, None, None)

train_inputs = [X for X, Y in trains]
all_inputs = train_inputs + [test_input]

print("=" * 80)
print("ATOM ANALYSIS: What makes positions different?")
print("=" * 80)
print()

# Build presents and extract raw colors
for grid_idx, X in enumerate(all_inputs):
    if grid_idx < len(trains):
        print(f"TRAINING GRID {grid_idx}:")
    else:
        print(f"TEST GRID:")

    print(f"  Size: {X.H}×{X.W}")

    # Get raw colors
    raw_colors = set()
    for r in range(X.H):
        for c in range(X.W):
            raw_colors.add(X[r][c])

    print(f"  Raw colors present: {sorted(raw_colors)}")

    # Count positions by color
    color_counts = {}
    for r in range(X.H):
        for c in range(X.W):
            color = X[r][c]
            if color not in color_counts:
                color_counts[color] = 0
            color_counts[color] += 1

    print(f"  Color histogram: {dict(sorted(color_counts.items()))}")

    # Get border positions
    border_positions = []
    for r in range(X.H):
        for c in range(X.W):
            if r == 0 or r == X.H - 1 or c == 0 or c == X.W - 1:
                border_positions.append((r, c))

    print(f"  Border positions: {len(border_positions)}/{X.H * X.W}")
    print()

# Now let's check if ATOMS are shared across grids
print("=" * 80)
print("INITIAL WL COLORING: Atoms and their hashes")
print("=" * 80)
print()

# Build presents
all_presents = [build_present(X, opts, phases) for X in all_inputs]

# Manually extract atoms (mimicking _build_initial_coloring)
from stable import stable_hash64

def extract_atoms(grid_idx, present):
    """Extract atoms for a single grid."""
    grid = present['grid']
    cbc_tokens = present.get('CBC1', None)

    atoms = {}
    for r in range(grid.H):
        for c in range(grid.W):
            pos = (r, c)
            is_border = (r == 0 or r == grid.H - 1 or c == 0 or c == grid.W - 1)

            color_tag = grid[r][c]
            cbc_token = cbc_tokens[pos] if cbc_tokens else 0

            atom = (color_tag, cbc_token, is_border)
            atom_hash = stable_hash64(atom)

            atoms[pos] = (atom, atom_hash)

    return atoms

# Extract atoms for all grids
all_atoms = []
for grid_idx, present in enumerate(all_presents):
    atoms = extract_atoms(grid_idx, present)
    all_atoms.append(atoms)

    if grid_idx < len(trains):
        print(f"TRAINING GRID {grid_idx} ATOMS:")
    else:
        print(f"TEST GRID ATOMS:")

    # Group positions by atom
    atom_groups = {}
    for pos, (atom, atom_hash) in atoms.items():
        if atom not in atom_groups:
            atom_groups[atom] = []
        atom_groups[atom].append(pos)

    print(f"  Unique atoms: {len(atom_groups)}")
    for atom, positions in sorted(atom_groups.items(), key=lambda x: len(x[1]), reverse=True):
        color, cbc, border = atom
        print(f"    Atom (color={color}, cbc={cbc}, border={border}): {len(positions)} positions")

    print()

# Check atom overlap across grids
print("=" * 80)
print("ATOM OVERLAP ANALYSIS")
print("=" * 80)
print()

# Collect unique atoms from training
train_atoms = set()
for grid_idx in range(len(trains)):
    for pos, (atom, atom_hash) in all_atoms[grid_idx].items():
        train_atoms.add(atom)

print(f"Unique atoms in training: {len(train_atoms)}")
for atom in sorted(train_atoms):
    color, cbc, border = atom
    print(f"  (color={color}, cbc={cbc}, border={border})")
print()

# Collect unique atoms from test
test_atoms = set()
for pos, (atom, atom_hash) in all_atoms[-1].items():
    test_atoms.add(atom)

print(f"Unique atoms in test: {len(test_atoms)}")
for atom in sorted(test_atoms):
    color, cbc, border = atom
    print(f"  (color={color}, cbc={cbc}, border={border})")
print()

# Overlap
atom_overlap = train_atoms & test_atoms
print(f"Atom overlap (train ∩ test): {len(atom_overlap)}/{len(test_atoms)}")

if len(atom_overlap) == 0:
    print("❌ ZERO ATOM OVERLAP!")
    print("   Test uses completely different (color, cbc, border) combinations")
    print("   This is why WL hashes diverge from the start.")
else:
    print(f"✅ {len(atom_overlap)} atoms are shared")
    for atom in sorted(atom_overlap):
        color, cbc, border = atom
        print(f"  (color={color}, cbc={cbc}, border={border})")

print()

# ========================================
# ROOT CAUSE
# ========================================
print("=" * 80)
print("ROOT CAUSE DIAGNOSIS")
print("=" * 80)
print("""
If atom overlap is ZERO:
  - Test grid uses different raw colors than training
  - Even at WL iteration 0, test positions get different initial hashes
  - No amount of WL refinement can align them

  This violates math_spec assumption that "roles are defined by input
  structure alone" - but we're including RAW COLOR in the atom!

Per math_spec.md line 36 (CBC definition):
  "CBC: 'color-blind canonical patch' (OFA inside patch + D8 canonicalization)"

The atom should NOT include raw palette values directly - it should be
COLOR-BLIND. But wl.py line 135 includes:

    color_tag = grid[r][c]  # Raw palette value 0-9
    atom = (color_tag, cbc_token, is_border)

This makes atoms PALETTE-DEPENDENT, not structure-only.

Solution:
  Remove raw color from atom, use only CBC (which is already color-blind)
  and border. The WL will then distinguish positions by structural
  neighborhoods, not by palette.
""")
