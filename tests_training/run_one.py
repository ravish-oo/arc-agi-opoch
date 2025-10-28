#!/usr/bin/env python3
"""
Run CPRQ on a single ARC task from the Kaggle training data.
Usage: python run_one.py --task-id 00576224
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path (modules are in root)
sys.path.insert(0, str(Path(__file__).parent.parent))

from grid import Grid
from cprq import compile_CPRQ
from stable import row_major_string


def load_task(task_id: str, challenges_path: str, solutions_path: str):
    """Load a single task from the JSON files."""
    # Load challenges
    with open(challenges_path, 'r') as f:
        challenges = json.load(f)

    if task_id not in challenges:
        raise ValueError(f"Task {task_id} not found in challenges")

    task = challenges[task_id]

    # Load solution
    with open(solutions_path, 'r') as f:
        solutions = json.load(f)

    solution = solutions.get(task_id, None)

    return task, solution


def run_task(task_id: str, challenges_path: str, solutions_path: str):
    """Run CPRQ on a task and report results."""
    print(f"=== Task {task_id} ===")

    # Load task
    task, solution = load_task(task_id, challenges_path, solutions_path)

    # Parse training pairs
    trains = []
    for pair in task['train']:
        X = Grid(pair['input'])
        Y = Grid(pair['output'])
        trains.append((X, Y))

    print(f"Training examples: {len(trains)}")

    # Run CPRQ compiler
    base_opts = {}
    result = compile_CPRQ(trains, base_opts)

    if result[0] is None:
        # Compilation failed
        witness = result[1]
        print(f"❌ COMPILATION FAILED")
        print(f"Witness: {witness}")
        return {
            'task_id': task_id,
            'status': 'compile_failed',
            'witness': witness
        }

    # Compilation succeeded
    compile_result, _ = result
    Psi_list, rho, opts_used = compile_result

    print(f"✓ Compilation succeeded with opts: {opts_used}")

    # Verify training examples
    train_ok = True
    for i, (X, Y) in enumerate(trains):
        Psi = Psi_list[i]
        Y_pred_data = []
        for r in range(len(X)):
            row = []
            for c in range(len(X[0])):
                role_id = Psi[(r, c)]
                pred_color = rho.get(role_id, 0)
                row.append(pred_color)
            Y_pred_data.append(row)

        Y_pred = Grid(Y_pred_data)

        # Compare
        Y_str = row_major_string(Y)
        Y_pred_str = row_major_string(Y_pred)

        if Y_str != Y_pred_str:
            print(f"❌ Train example {i}: MISMATCH")
            train_ok = False
        else:
            print(f"✓ Train example {i}: MATCH")

    if not train_ok:
        return {
            'task_id': task_id,
            'status': 'train_mismatch',
            'opts_used': opts_used
        }

    print(f"✓ All training examples match!")

    # Check if we have test input and solution
    if 'test' not in task or len(task['test']) == 0:
        print("No test input")
        return {
            'task_id': task_id,
            'status': 'train_ok',
            'opts_used': opts_used
        }

    if solution is None or len(solution) == 0:
        print("No test solution available")
        return {
            'task_id': task_id,
            'status': 'train_ok',
            'opts_used': opts_used
        }

    # We have test input and solution - but we need MK-06 (predict.py) to predict!
    print(f"⚠️  Test prediction requires MK-06 (predict.py) - not yet implemented")

    return {
        'task_id': task_id,
        'status': 'train_ok',
        'opts_used': opts_used,
        'note': 'test_prediction_needs_mk06'
    }


def main():
    parser = argparse.ArgumentParser(description='Run CPRQ on a single ARC task')
    parser.add_argument('--task-id', required=True, help='Task ID (e.g., 00576224)')
    parser.add_argument('--challenges', default='data/arc-agi_training_challenges.json')
    parser.add_argument('--solutions', default='data/arc-agi_training_solutions.json')

    args = parser.parse_args()

    try:
        result = run_task(args.task_id, args.challenges, args.solutions)
        print(f"\nResult: {json.dumps(result, indent=2)}")

        # Exit code
        if result['status'] == 'train_ok':
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
