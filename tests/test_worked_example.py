#!/usr/bin/env python3
"""Test task 00576224 from worked examples doc."""

import json
from grid import Grid
from cprq import compile_CPRQ, predict, _grids_equal

# Load task
with open('data/arc-agi_training_challenges.json') as f:
    challenges = json.load(f)

with open('data/arc-agi_training_solutions.json') as f:
    solutions = json.load(f)

task_id = '00576224'
task = challenges[task_id]
ground_truth = solutions[task_id]

# Parse training
trains = [(Grid(p['input']), Grid(p['output'])) for p in task['train']]

print(f"Task {task_id} (from worked examples doc):")
print(f"Training examples: {len(trains)}\n")

# Show first training example
X, Y = trains[0]
print(f"Train 0:")
print(f"  X shape: {X.H}x{X.W}")
print(f"  Y shape: {Y.H}x{Y.W}")
print(f"  X: {[[X[r][c] for c in range(X.W)] for r in range(X.H)]}")
print(f"  Y: {[[Y[r][c] for c in range(Y.W)] for r in range(Y.H)]}")

# Compile
result, witness = compile_CPRQ(trains, {})

if result is None:
    print(f"\n❌ Compilation FAILED")
    print(f"Witness: {witness}")
else:
    print(f"\n✓ Compilation succeeded")
    Psi_list, rho, options_used = result

    # Check training
    all_exact = True
    from cprq import _predict_with_rho
    for i, (X, Y) in enumerate(trains):
        Psi = Psi_list[i]
        Y_pred = _predict_with_rho(X, Psi, rho)
        if not _grids_equal(Y, Y_pred):
            print(f"  Train {i}: ✗ MISMATCH")
            all_exact = False
        else:
            print(f"  Train {i}: ✓ EXACT")

    if not all_exact:
        print("\n❌ Training not bit-exact - this should not happen!")
    else:
        # Test prediction
        test_input = Grid(task['test'][0]['input'])
        gt_output = Grid(ground_truth[0])

        print(f"\nTest:")
        print(f"  Input shape: {test_input.H}x{test_input.W}")
        print(f"  GT shape: {gt_output.H}x{gt_output.W}")

        pred_output = predict(test_input, trains, result)
        print(f"  Pred shape: {pred_output.H}x{pred_output.W}")

        if _grids_equal(gt_output, pred_output):
            print(f"\n✅ PERFECT PREDICTION!")
        else:
            wrong = sum(1 for r in range(min(gt_output.H, pred_output.H))
                       for c in range(min(gt_output.W, pred_output.W))
                       if gt_output[r][c] != pred_output[r][c])
            total = gt_output.H * gt_output.W
            print(f"\n✗ WRONG ({wrong}/{total} cells)")

            if pred_output.H <= 10 and pred_output.W <= 10:
                print(f"\nGT output:")
                for r in range(gt_output.H):
                    print(f"  {[gt_output[r][c] for c in range(gt_output.W)]}")
                print(f"\nPred output:")
                for r in range(pred_output.H):
                    print(f"  {[pred_output[r][c] for c in range(pred_output.W)]}")
