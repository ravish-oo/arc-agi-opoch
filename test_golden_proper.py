#!/usr/bin/env python3
"""Proper test: Compile CPRQ on training, predict on test, compare to ground truth."""

import json
from grid import Grid
from cprq import compile_CPRQ, predict

# Load challenges and solutions
with open('data/arc-agi_training_challenges.json') as f:
    challenges = json.load(f)

with open('data/arc-agi_training_solutions.json') as f:
    solutions = json.load(f)

# Golden tasks
golden_tasks = ['007bbfb7', '0d3d703e', '25ff71a9', '3c9b0459', 'a2fd1cf0']

print('Testing golden tasks: Training + Test prediction vs Ground Truth\n')
print('=' * 70)

total_passed = 0
total_failed = 0

for task_id in golden_tasks:
    print(f'\n{task_id}:')

    task = challenges[task_id]
    ground_truth = solutions[task_id]

    # Parse training data
    trains = [(Grid(p['input']), Grid(p['output'])) for p in task['train']]
    print(f'  Training examples: {len(trains)}')

    # Compile CPRQ
    result, witness = compile_CPRQ(trains, {})

    if result is None:
        print(f'  ✗ COMPILATION FAILED (witness found)')
        if witness:
            print(f'    Witness type: {witness.get("type", "?")}')
        total_failed += 1
        continue

    print(f'  ✓ Compilation succeeded')

    # Parse test inputs
    test_inputs = [Grid(p['input']) for p in task['test']]
    print(f'  Test cases: {len(test_inputs)}')

    # Predict on each test input
    all_correct = True
    for test_idx, test_input in enumerate(test_inputs):
        # Get ground truth
        gt_grid = Grid(ground_truth[test_idx])

        # Make prediction
        predicted_grid = predict(test_input, trains, result)

        # Compare
        if predicted_grid.H != gt_grid.H or predicted_grid.W != gt_grid.W:
            print(f'    Test {test_idx}: ✗ WRONG SHAPE (pred {predicted_grid.H}x{predicted_grid.W} vs gt {gt_grid.H}x{gt_grid.W})')
            all_correct = False
            continue

        # Count wrong cells
        wrong_cells = 0
        for r in range(gt_grid.H):
            for c in range(gt_grid.W):
                if predicted_grid[r][c] != gt_grid[r][c]:
                    wrong_cells += 1

        total_cells = gt_grid.H * gt_grid.W
        accuracy = 100.0 * (total_cells - wrong_cells) / total_cells if total_cells > 0 else 0.0

        if wrong_cells == 0:
            print(f'    Test {test_idx}: ✓ PERFECT ({gt_grid.H}x{gt_grid.W})')
        else:
            print(f'    Test {test_idx}: ✗ WRONG ({wrong_cells}/{total_cells} cells, {accuracy:.1f}% accurate)')
            all_correct = False

    if all_correct:
        print(f'  ✅ ALL TEST CASES CORRECT')
        total_passed += 1
    else:
        print(f'  ❌ SOME TEST CASES WRONG')
        total_failed += 1

print('\n' + '=' * 70)
print(f'FINAL RESULTS: {total_passed}/5 tasks with perfect predictions')
print(f'               {total_failed}/5 tasks with errors')
