#!/usr/bin/env python3
"""Test CPRQ on ARC-AGI evaluation set with detailed output to file."""

import json
import sys
from grid import Grid
from cprq import compile_CPRQ, predict

# Output file
OUTPUT_FILE = 'test_results.txt'

def write_and_print(f, msg):
    """Write to both file and stdout."""
    print(msg)
    f.write(msg + '\n')
    f.flush()

def grid_to_string(grid):
    """Convert grid to readable string."""
    lines = []
    for r in range(grid.H):
        row = [str(grid[r][c]) for c in range(grid.W)]
        lines.append(' '.join(row))
    return '\n'.join(lines)

# Load evaluation challenges and solutions
print("Loading evaluation data...")
with open('data/arc-agi_evaluation_challenges.json') as f:
    challenges = json.load(f)

with open('data/arc-agi_evaluation_solutions.json') as f:
    solutions = json.load(f)

# Select 5 tasks to test
# Pick from worked examples if available, or first 5 from evaluation set
task_ids = list(challenges.keys())[:5]

# Try to use known tasks from worked examples if they exist in evaluation set
known_tasks = ['332efdb3', '9172f3a0', '3c9b0459', '68b16354', '74dd1130',
               '9dfd6313', 'ed36ccf7', '0d3d703e']
for task_id in known_tasks:
    if task_id in challenges and task_id in solutions:
        if task_id not in task_ids:
            task_ids = [task_id] + task_ids
            if len(task_ids) > 5:
                task_ids = task_ids[:5]

# Open output file
with open(OUTPUT_FILE, 'w') as f:
    write_and_print(f, '=' * 80)
    write_and_print(f, 'ARC-AGI EVALUATION TEST RESULTS')
    write_and_print(f, 'Testing CPRQ Formula: F = ρ ∘ π_E*  where E* = K_𝒢 ∧ Int^𝒢(L)')
    write_and_print(f, '=' * 80)
    write_and_print(f, '')
    write_and_print(f, f'Testing {len(task_ids)} tasks from evaluation set')
    write_and_print(f, f'Task IDs: {", ".join(task_ids)}')
    write_and_print(f, '')
    write_and_print(f, '=' * 80)

    total_passed = 0
    total_failed = 0
    total_compile_failed = 0

    for task_num, task_id in enumerate(task_ids, 1):
        write_and_print(f, '')
        write_and_print(f, f'[{task_num}/5] Task: {task_id}')
        write_and_print(f, '-' * 80)

        task = challenges[task_id]
        ground_truth = solutions.get(task_id, [])

        # Parse training data
        trains = [(Grid(p['input']), Grid(p['output'])) for p in task['train']]
        write_and_print(f, f'  Training examples: {len(trains)}')

        # Show training pair shapes
        for i, (X, Y) in enumerate(trains):
            write_and_print(f, f'    Train {i}: {X.H}×{X.W} → {Y.H}×{Y.W}')

        # Compile CPRQ
        write_and_print(f, '')
        write_and_print(f, '  Compiling CPRQ...')

        try:
            result, witness = compile_CPRQ(trains, {})
        except Exception as e:
            write_and_print(f, f'  ❌ COMPILATION CRASHED: {e}')
            total_compile_failed += 1
            total_failed += 1
            continue

        if result is None:
            write_and_print(f, '  ❌ COMPILATION FAILED')
            if witness:
                write_and_print(f, f'     Witness type: {witness.get("type", "unknown")}')
                if 'reason' in witness:
                    write_and_print(f, f'     Reason: {witness["reason"]}')
                if 'train_idx' in witness:
                    write_and_print(f, f'     Failed on train: {witness["train_idx"]}')
            total_compile_failed += 1
            total_failed += 1
            continue

        # Unpack compilation result
        Psi_list, rho, wl_depth, opts, domain_mode, scale_or_none, pi_tag, phases, wl_iter_count = result

        write_and_print(f, '  ✅ COMPILATION SUCCEEDED')
        write_and_print(f, f'     Π transform: {pi_tag}')
        write_and_print(f, f'     WL depth: {wl_depth}')
        write_and_print(f, f'     WL iterations: {wl_iter_count}')
        write_and_print(f, f'     Domain mode: {domain_mode}')
        write_and_print(f, f'     Optional relations: {opts}')

        row_k, col_k, diag_k = phases
        if any([row_k, col_k, diag_k]):
            write_and_print(f, f'     Phases: row_k={row_k}, col_k={col_k}, diag_k={diag_k}')

        write_and_print(f, f'     Roles (E* classes): {len(set(c for psi in Psi_list for c in psi.values()))}')
        write_and_print(f, f'     ρ (program): {len(rho)} role→color mappings')

        # Parse test inputs
        test_inputs = [Grid(p['input']) for p in task['test']]
        write_and_print(f, '')
        write_and_print(f, f'  Testing on {len(test_inputs)} test case(s)...')

        # Predict on each test input
        all_correct = True
        for test_idx, test_input in enumerate(test_inputs):
            write_and_print(f, '')
            write_and_print(f, f'    Test {test_idx}: Input {test_input.H}×{test_input.W}')

            # Make prediction
            try:
                predicted_grid = predict(test_input, trains, result)
            except Exception as e:
                write_and_print(f, f'      ❌ PREDICTION FAILED: {e}')
                all_correct = False
                continue

            # Get ground truth if available
            if test_idx >= len(ground_truth):
                write_and_print(f, f'      ⚠️  No ground truth available for test {test_idx}')
                write_and_print(f, f'      Predicted: {predicted_grid.H}×{predicted_grid.W}')
                continue

            gt_grid = Grid(ground_truth[test_idx])

            # Compare shapes
            if predicted_grid.H != gt_grid.H or predicted_grid.W != gt_grid.W:
                write_and_print(f, f'      ❌ WRONG SHAPE')
                write_and_print(f, f'         Predicted: {predicted_grid.H}×{predicted_grid.W}')
                write_and_print(f, f'         Expected:  {gt_grid.H}×{gt_grid.W}')
                all_correct = False
                continue

            # Count wrong cells
            wrong_cells = 0
            wrong_positions = []
            for r in range(gt_grid.H):
                for c in range(gt_grid.W):
                    if predicted_grid[r][c] != gt_grid[r][c]:
                        wrong_cells += 1
                        if len(wrong_positions) < 5:  # Show first 5 errors
                            wrong_positions.append((r, c, predicted_grid[r][c], gt_grid[r][c]))

            total_cells = gt_grid.H * gt_grid.W
            accuracy = 100.0 * (total_cells - wrong_cells) / total_cells if total_cells > 0 else 0.0

            if wrong_cells == 0:
                write_and_print(f, f'      ✅ PERFECT MATCH ({gt_grid.H}×{gt_grid.W}, {total_cells} cells)')
            else:
                write_and_print(f, f'      ❌ WRONG: {wrong_cells}/{total_cells} cells ({accuracy:.1f}% accurate)')
                if wrong_positions:
                    write_and_print(f, f'         First errors:')
                    for r, c, pred, exp in wrong_positions:
                        write_and_print(f, f'           ({r},{c}): predicted {pred}, expected {exp}')
                all_correct = False

        write_and_print(f, '')
        if all_correct:
            write_and_print(f, '  ✅✅✅ ALL TEST CASES PERFECT ✅✅✅')
            total_passed += 1
        else:
            write_and_print(f, '  ❌ SOME TEST CASES WRONG OR FAILED')
            total_failed += 1

    # Final summary
    write_and_print(f, '')
    write_and_print(f, '=' * 80)
    write_and_print(f, 'FINAL RESULTS')
    write_and_print(f, '=' * 80)
    write_and_print(f, f'Tasks with perfect predictions:  {total_passed}/{len(task_ids)}')
    write_and_print(f, f'Tasks with errors/failures:      {total_failed}/{len(task_ids)}')
    write_and_print(f, f'Tasks that failed compilation:   {total_compile_failed}/{len(task_ids)}')
    write_and_print(f, '')

    if total_passed == len(task_ids):
        write_and_print(f, '🎉 ALL TASKS PERFECT! 🎉')
    elif total_passed > 0:
        write_and_print(f, f'✓ {total_passed} task(s) solved perfectly')
    else:
        write_and_print(f, '⚠️  No tasks solved perfectly')

    write_and_print(f, '')
    write_and_print(f, '=' * 80)

print(f'\n✓ Results written to {OUTPUT_FILE}')
