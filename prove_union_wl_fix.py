#!/usr/bin/env python3
"""
Prove that running WL once on train∪test solves the alignment problem.
"""

import json
from grid import Grid
from present import build_present
from wl import wl_disjoint_union

# Load data
with open('data/arc-agi_training_challenges.json') as f:
    challenges = json.load(f)

with open('data/arc-agi_training_solutions.json') as f:
    solutions = json.load(f)

task_id = '0d3d703e'
task = challenges[task_id]
ground_truth = solutions[task_id]

print(f"Task: {task_id}")
print("=" * 80)
print()

# Prepare grids
trains = [(Grid(p['input']), Grid(p['output'])) for p in task['train']]
test_input = Grid(task['test'][0]['input'])

# Build presents
opts = {}
phases = (None, None, None)

train_inputs = [X for X, Y in trains]
all_inputs = train_inputs + [test_input]

print("Step 1: Run WL ONCE on train∪test")
print("-" * 80)

all_presents = [build_present(X, opts, phases) for X in all_inputs]
Psi_list, iter_count = wl_disjoint_union(all_presents, depth=1)

print(f"WL iterations: {iter_count}")
print(f"Total grids: {len(Psi_list)} (4 train + 1 test)")
print()

# Split back
Psi_trains = Psi_list[:-1]
Psi_test = Psi_list[-1]

print("Step 2: Check role alignment")
print("-" * 80)

train_role_ids = set()
for grid_idx, psi in enumerate(Psi_trains):
    roles = set(psi.values())
    train_role_ids.update(roles)
    print(f"Train grid {grid_idx}: {len(roles)} unique roles")

test_role_ids = set(Psi_test.values())
print(f"Test grid: {len(test_role_ids)} unique roles")
print()

overlap = train_role_ids & test_role_ids
print(f"Overlap (train ∩ test): {len(overlap)}/{len(test_role_ids)}")

if len(overlap) == 0:
    print("❌ Still zero overlap - WL is fundamentally palette-dependent")
    print("   Even with union-WL, test positions with different color contexts")
    print("   get different role IDs than training positions.")
else:
    print(f"✅ {len(overlap)} roles shared! Union-WL alignment works!")

print()

# Step 3: Build ρ from training
print("Step 3: Build ρ from training positions")
print("-" * 80)

rho = {}
for train_idx, (X, Y) in enumerate(trains):
    psi = Psi_trains[train_idx]
    for pos in psi.keys():
        r, c = pos
        role_id = psi[pos]
        y_color = Y[r][c]

        if role_id in rho:
            if rho[role_id] != y_color:
                print(f"❌ Role {role_id} maps to multiple colors: {rho[role_id]} and {y_color}")
        else:
            rho[role_id] = y_color

print(f"ρ size: {len(rho)} mappings")
print()

# Step 4: Predict on test
print("Step 4: Predict on test using pre-computed roles")
print("-" * 80)

unseen_roles = []
predicted_grid = [[0 for _ in range(test_input.W)] for _ in range(test_input.H)]

for pos, role_id in Psi_test.items():
    r, c = pos
    if role_id in rho:
        predicted_grid[r][c] = rho[role_id]
    else:
        unseen_roles.append((pos, role_id))
        predicted_grid[r][c] = 0  # Default

if unseen_roles:
    print(f"❌ {len(unseen_roles)} positions have unseen roles")
    print(f"   First 5: {unseen_roles[:5]}")
else:
    print("✅ All test positions have known roles!")

print()

# Step 5: Compare to ground truth
print("Step 5: Compare to ground truth")
print("-" * 80)

gt_grid = Grid(ground_truth[0])

print("Predicted:")
for row in predicted_grid:
    print(" ", row)

print("\nGround truth:")
for r in range(gt_grid.H):
    print(" ", [gt_grid[r][c] for c in range(gt_grid.W)])

# Check equality
wrong = 0
for r in range(gt_grid.H):
    for c in range(gt_grid.W):
        if predicted_grid[r][c] != gt_grid[r][c]:
            wrong += 1

total = gt_grid.H * gt_grid.W

if wrong == 0:
    print(f"\n✅✅✅ PERFECT MATCH! All {total} cells correct")
else:
    print(f"\n❌ {wrong}/{total} cells wrong ({100*wrong/total:.1f}% error)")

print()
print("=" * 80)
print("CONCLUSION")
print("=" * 80)

if len(overlap) > 0 and len(unseen_roles) == 0 and wrong == 0:
    print("✅ Union-WL fixes the alignment problem!")
    print("   Running WL once on train∪test gives aligned role IDs.")
    print("   This allows test positions to reuse training mappings.")
    print()
    print("   FIX REQUIRED: Modify compile_CPRQ to accept test inputs and")
    print("   run WL on train∪test during compilation, not during predict.")
else:
    print("❌ Union-WL alone doesn't solve it.")
    print("   Even with WL on train∪test, alignment fails due to palette.")
    print()
    print("   This means the math spec requires additional handling,")
    print("   possibly through orbit mode or different present design.")
