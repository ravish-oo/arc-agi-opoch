#!/usr/bin/env python3
"""Trace WL refinement to see exactly when/why colors merge."""

import json
from grid import Grid
from present import build_present
from wl import wl_disjoint_union

# Load task
with open('data/arc-agi_training_challenges.json') as f:
    challenges = json.load(f)

task = challenges['0d3d703e']

# First 2 training pairs only
trains = []
for i, pair in enumerate(task['train'][:2]):
    X = Grid(pair['input'])
    Y = Grid(pair['output'])
    trains.append((X, Y))
    print(f"Train {i}: X={[[X[r][c] for c in range(X.W)] for r in range(X.H)]}")
    print(f"         Y={[[Y[r][c] for c in range(Y.W)] for r in range(Y.H)]}")
    print()

# Build presents
presents = [build_present(X, {}) for X, Y in trains]

# Debug positions: (grid_idx, (row, col))
debug_positions = [
    (0, (1, 0)),  # Train 0, pos (1,0): X=5
    (1, (1, 0)),  # Train 1, pos (1,0): X=2
]

print("=" * 60)
print("Tracing WL refinement for positions with DIFFERENT input colors:")
print(f"  Grid 0, pos (1,0): X=5 → Y=1")
print(f"  Grid 1, pos (1,0): X=2 → Y=6")
print("=" * 60)

# Run WL with debug tracing
Psi_list = wl_disjoint_union(presents, debug_positions=debug_positions)

print("\n" + "=" * 60)
print("Final roles:")
for i, Psi in enumerate(Psi_list):
    role = Psi.get((1, 0), '?')
    X, Y = trains[i]
    x_color = X[1][0]
    y_color = Y[1][0]
    print(f"  Grid {i}, pos (1,0): role={role}, X={x_color}, Y={y_color}")

# Check if merged
role0 = Psi_list[0].get((1, 0))
role1 = Psi_list[1].get((1, 0))
if role0 == role1:
    print(f"\n⚠️  MERGED! Both positions have role {role0} despite different input colors (5 vs 2)")
else:
    print(f"\n✓ SEPARATED: Grid 0 has role {role0}, Grid 1 has role {role1}")
