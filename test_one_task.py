#!/usr/bin/env python3
"""Test ONE task to debug union-WL alignment."""

import json
from grid import Grid
from cprq import compile_CPRQ, predict

# Load data
with open('data/arc-agi_training_challenges.json') as f:
    challenges = json.load(f)

with open('data/arc-agi_training_solutions.json') as f:
    solutions = json.load(f)

# Pick ONE simple task
task_id = '0d3d703e'  # Simple colormap task from worked examples

print(f"Testing task: {task_id}")
print("=" * 80)

task = challenges[task_id]
ground_truth = solutions[task_id]

# Training
trains = [(Grid(p['input']), Grid(p['output'])) for p in task['train']]
print(f"Training: {len(trains)} pairs")
for i, (X, Y) in enumerate(trains):
    print(f"  Train {i}: {X.H}×{X.W} → {Y.H}×{Y.W}")

# Test inputs (needed for union-WL)
test_inputs = [Grid(p['input']) for p in task['test']]

# Compile (per math_spec: WL runs on train∪test)
print("\nCompiling...")
result, witness = compile_CPRQ(trains, test_inputs, {})

if result is None:
    print(f"❌ COMPILATION FAILED")
    print(f"   Witness: {witness}")
    exit(1)

# Unpack (11 elements now)
Psi_list, rho, wl_depth, opts, domain_mode, scale_or_none, pi_tag, phases, wl_iter_count, label_mode, Psi_list_test = result

print(f"✅ COMPILATION SUCCEEDED")
print(f"   Label mode: {label_mode}")
print(f"   WL depth: {wl_depth}")
print(f"   WL iterations: {wl_iter_count}")
print(f"   Π: {pi_tag}")
print(f"   Roles: {len(set(c for psi in Psi_list for c in psi.values()))}")
print(f"   ρ size: {len(rho)}")

# Test
print(f"\nTesting: {len(test_inputs)} case(s)")

for test_idx, test_input in enumerate(test_inputs):
    print(f"\n  Test {test_idx}: {test_input.H}×{test_input.W}")

    gt_grid = Grid(ground_truth[test_idx])
    print(f"  Expected: {gt_grid.H}×{gt_grid.W}")

    # Predict (using pre-computed Psi)
    try:
        predicted = predict(test_input, trains, result, test_idx)
        print(f"  Predicted: {predicted.H}×{predicted.W}")

        # Compare
        if predicted.H != gt_grid.H or predicted.W != gt_grid.W:
            print(f"  ❌ WRONG SHAPE")
            continue

        wrong = sum(1 for r in range(gt_grid.H) for c in range(gt_grid.W)
                   if predicted[r][c] != gt_grid[r][c])
        total = gt_grid.H * gt_grid.W

        if wrong == 0:
            print(f"  ✅ PERFECT ({total} cells)")
        else:
            print(f"  ❌ WRONG: {wrong}/{total} cells")

    except Exception as e:
        print(f"  ❌ PREDICTION FAILED: {e}")
