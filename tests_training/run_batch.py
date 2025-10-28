#!/usr/bin/env python3
"""
Run CPRQ on multiple ARC tasks and report success/failure stats.
Usage: python run_batch.py [--limit N]
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path (modules are in root)
sys.path.insert(0, str(Path(__file__).parent.parent))

from grid import Grid
from cprq import compile_CPRQ


def load_all_tasks(challenges_path: str):
    """Load all task IDs."""
    with open(challenges_path, 'r') as f:
        challenges = json.load(f)
    return list(challenges.keys())


def run_task_silent(task_id: str, challenges_path: str):
    """Run CPRQ on a task and return status."""
    try:
        with open(challenges_path, 'r') as f:
            challenges = json.load(f)

        if task_id not in challenges:
            return {'status': 'not_found'}

        task = challenges[task_id]

        # Parse training pairs
        trains = []
        for pair in task['train']:
            X = Grid(pair['input'])
            Y = Grid(pair['output'])
            trains.append((X, Y))

        # Run CPRQ compiler
        base_opts = {}
        result = compile_CPRQ(trains, base_opts)

        if result[0] is None:
            # Compilation failed
            witness = result[1]
            return {
                'status': 'compile_failed',
                'witness_type': witness.get('type', 'unknown'),
                'opts': witness.get('opts', {})
            }

        # Compilation succeeded
        compile_result, _ = result
        _, _, opts_used = compile_result

        return {
            'status': 'train_ok',
            'opts_used': opts_used
        }

    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='Run CPRQ on multiple ARC tasks')
    parser.add_argument('--limit', type=int, default=50, help='Number of tasks to test')
    parser.add_argument('--challenges', default='data/arc-agi_training_challenges.json')

    args = parser.parse_args()

    # Load all task IDs
    task_ids = load_all_tasks(args.challenges)
    print(f"Total tasks in dataset: {len(task_ids)}")
    print(f"Testing first {args.limit} tasks...\n")

    # Run on first N tasks
    results = []
    for i, task_id in enumerate(task_ids[:args.limit]):
        result = run_task_silent(task_id, args.challenges)
        result['task_id'] = task_id
        results.append(result)

        # Print progress
        status_symbol = {
            'train_ok': '✓',
            'compile_failed': '✗',
            'error': '⚠'
        }.get(result['status'], '?')

        print(f"{i+1:3d}. {task_id}  {status_symbol}  {result['status']}", end='')
        if result['status'] == 'train_ok':
            print(f"  opts={result.get('opts_used', {})}")
        elif result['status'] == 'compile_failed':
            print(f"  {result['witness_type']}")
        else:
            print()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    success_count = sum(1 for r in results if r['status'] == 'train_ok')
    fail_count = sum(1 for r in results if r['status'] == 'compile_failed')
    error_count = sum(1 for r in results if r['status'] == 'error')

    print(f"Success:         {success_count:3d} / {args.limit}  ({100*success_count/args.limit:.1f}%)")
    print(f"Compile failed:  {fail_count:3d} / {args.limit}  ({100*fail_count/args.limit:.1f}%)")
    print(f"Errors:          {error_count:3d} / {args.limit}  ({100*error_count/args.limit:.1f}%)")

    # Show successful tasks
    if success_count > 0:
        print(f"\n{'='*60}")
        print("SUCCESSFUL TASKS (first 10)")
        print(f"{'='*60}")
        successes = [r for r in results if r['status'] == 'train_ok']
        for r in successes[:10]:
            print(f"{r['task_id']}  opts={r['opts_used']}")

    # Save full results
    output_path = Path(__file__).parent / 'batch_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to: {output_path}")


if __name__ == '__main__':
    main()
