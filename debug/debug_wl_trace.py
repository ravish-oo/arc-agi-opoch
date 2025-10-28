#!/usr/bin/env python3
"""Trace WL refinement to see why positions are splitting."""

import json
from grid import Grid
from present import build_present
from wl import wl_disjoint_union

# Load task
with open('data/arc-agi_training_challenges.json') as f:
    challenges = json.load(f)

task_id = '0d3d703e'
task = challenges[task_id]

# Get first training example
X0 = Grid(task['train'][0]['input'])
Y0 = Grid(task['train'][0]['output'])

print("Training input:")
print(X0)
print("\nTraining output:")
print(Y0)
print()

# Build present
present = build_present(X0, {})

# Find two positions with same color (5)
pos_5_1 = (0, 0)  # Color 5, border
pos_5_2 = (1, 0)  # Color 5, border

# Find two positions with color 8
pos_8_1 = (0, 1)  # Color 8, border
pos_8_2 = (1, 1)  # Color 8, interior

print(f"Tracing positions:")
print(f"  {pos_5_1}: X={X0[pos_5_1[0]][pos_5_1[1]]}")
print(f"  {pos_5_2}: X={X0[pos_5_2[0]][pos_5_2[1]]}")
print(f"  {pos_8_1}: X={X0[pos_8_1[0]][pos_8_1[1]]}")
print(f"  {pos_8_2}: X={X0[pos_8_2[0]][pos_8_2[1]]}")
print()

# Convert to global positions
debug_positions = [
    (0, pos_5_1),
    (0, pos_5_2),
    (0, pos_8_1),
    (0, pos_8_2),
]

# Run WL with debug trace
Psi_list = wl_disjoint_union([present], debug_positions=debug_positions)

print("\n=== Final Partition ===")
Psi = Psi_list[0]
for pos in [pos_5_1, pos_5_2, pos_8_1, pos_8_2]:
    print(f"{pos}: role {Psi[pos]}")

print(f"\nTotal unique roles: {len(set(Psi.values()))}")
print(f"Should be ~8 for 8 colors")
