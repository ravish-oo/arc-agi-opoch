#!/usr/bin/env python3
"""Debug WL atom seed to see if color separation is working."""

import json
from grid import Grid
from present import build_present
from stable import stable_hash64

# Load task
with open('data/arc-agi_training_challenges.json') as f:
    challenges = json.load(f)

task = challenges['0d3d703e']

# First 2 training pairs
trains = []
for i, pair in enumerate(task['train'][:2]):
    X = Grid(pair['input'])
    Y = Grid(pair['output'])
    trains.append((X, Y))
    print(f"Train {i}: X={[[X[r][c] for c in range(X.W)] for r in range(X.H)]}")

# Build presents
presents = [build_present(X, {}) for X, Y in trains]

# Check atoms manually
print(f"\n=== Checking initial atoms ===\n")

for grid_idx, present in enumerate(presents):
    grid = present['grid']
    same_comp8 = present['sameComp8']
    band_row = present['bandRow']
    band_col = present['bandCol']

    print(f"Grid {grid_idx}:")
    print(f"  Position (1, 0): X={grid[1][0]}")

    pos = (1, 0)
    r, c = pos
    H, W = grid.H, grid.W
    is_border = (r == 0 or r == H - 1 or c == 0 or c == W - 1)

    color_tag = grid[r][c]
    comp8_tag = same_comp8[pos]
    row_tag = band_row[pos]
    col_tag = band_col[pos]
    cbc_token = 0

    atom = (color_tag, comp8_tag, row_tag, col_tag, cbc_token, is_border)
    atom_hash = stable_hash64(atom)

    print(f"  Atom: {atom}")
    print(f"  Atom hash: {atom_hash}")
    print()
