#!/usr/bin/env python3
"""Debug colormap task 0d3d703e to see why WL merges different colors."""

import json
import sys
from grid import Grid
from present import build_present
from wl import wl_disjoint_union

# Load task
with open('data/arc-agi_training_challenges.json') as f:
    challenges = json.load(f)

task = challenges['0d3d703e']

# Parse ALL training pairs
trains = []
for pair in task['train']:
    X = Grid(pair['input'])
    Y = Grid(pair['output'])
    trains.append((X, Y))

print(f"Task has {len(trains)} training examples\n")

# Show first 2 training examples
for i in range(min(2, len(trains))):
    X, Y = trains[i]
    print(f"=== Training example {i} ===")
    print(f"X (input):")
    for r in range(X.H):
        print([X[r][c] for c in range(X.W)])
    print(f"Y (output):")
    for r in range(Y.H):
        print([Y[r][c] for c in range(Y.W)])
    print()

# Build presents for all training examples
presents = [build_present(X, {}) for X, Y in trains]

# Run WL on disjoint union
Psi_list = wl_disjoint_union(presents)

print(f"\n=== After WL on disjoint union of {len(trains)} examples ===\n")

# Check if roles are single-valued across ALL training examples
from collections import defaultdict
role_to_y_colors = defaultdict(set)
role_to_x_colors = defaultdict(set)
role_to_samples = defaultdict(list)

for i, (X, Y) in enumerate(trains):
    Psi = Psi_list[i]
    for pos in Psi.keys():
        r, c = pos
        role = Psi[pos]
        x_color = X[r][c]
        y_color = Y[r][c]
        role_to_x_colors[role].add(x_color)
        role_to_y_colors[role].add(y_color)
        if len(role_to_samples[role]) < 3:  # Keep first 3 samples
            role_to_samples[role].append((i, pos, x_color, y_color))

print(f"Role → X colors, Y colors mapping (across ALL {len(trains)} examples):")
conflicts = 0
for role in sorted(role_to_y_colors.keys()):
    x_colors = sorted(role_to_x_colors[role])
    y_colors = sorted(role_to_y_colors[role])
    if len(y_colors) > 1:
        conflicts += 1
        print(f"  Role {role}: X={x_colors}, Y={y_colors} ✗ CONFLICT")
        for sample in role_to_samples[role][:2]:
            train_idx, pos, x_c, y_c = sample
            print(f"    Train {train_idx} pos {pos}: X={x_c} → Y={y_c}")
    else:
        print(f"  Role {role}: X={x_colors}, Y={y_colors} ✓ OK")

print(f"\nTotal conflicting roles: {conflicts}")
print(f"Total roles: {len(role_to_y_colors)}")

print(f"\n⚠️  PROBLEM: WL is merging positions with DIFFERENT input colors into same role!")
print(f"Role 0 has X colors: {sorted(role_to_x_colors[0])}")
