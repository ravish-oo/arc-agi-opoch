#!/usr/bin/env python3
"""
Run CPRQ on ARC corpus and generate detailed report.
Usage: python run_corpus.py [--limit N] [--output report.txt]
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

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

        # Check shape info
        shape_info = []
        for i, (X, Y) in enumerate(trains):
            shape_info.append(f"{X.H}×{X.W}→{Y.H}×{Y.W}")

        # Run CPRQ compiler
        base_opts = {}
        result = compile_CPRQ(trains, base_opts)

        if result[0] is None:
            # Compilation failed
            witness = result[1]
            return {
                'status': 'compile_failed',
                'witness_type': witness.get('type', 'unknown'),
                'opts': witness.get('opts', {}),
                'shape_info': shape_info,
                'note': witness.get('note', ''),
                'reason': witness.get('reason', ''),
            }

        # Compilation succeeded
        compile_result, _ = result
        _, _, opts_used = compile_result

        return {
            'status': 'train_ok',
            'opts_used': opts_used,
            'shape_info': shape_info,
        }

    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'shape_info': []
        }


def main():
    parser = argparse.ArgumentParser(description='Run CPRQ on ARC corpus')
    parser.add_argument('--limit', type=int, default=100, help='Number of tasks to test')
    parser.add_argument('--output', default='corpus_report.txt', help='Output report file')
    parser.add_argument('--challenges', default='data/arc-agi_training_challenges.json')

    args = parser.parse_args()

    # Load all task IDs
    task_ids = load_all_tasks(args.challenges)
    print(f"Total tasks in dataset: {len(task_ids)}")
    print(f"Testing first {args.limit} tasks...")
    print(f"Output: {args.output}\n")

    # Open output file
    with open(args.output, 'w') as f:
        # Write header
        f.write("="*80 + "\n")
        f.write("ARC-AGI CPRQ Corpus Evaluation Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Tasks tested: {args.limit} / {len(task_ids)}\n")
        f.write("="*80 + "\n\n")

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

            progress_line = f"{i+1:3d}. {task_id}  {status_symbol}  {result['status']}"

            if result['status'] == 'train_ok':
                progress_line += f"  opts={result.get('opts_used', {})}"
            elif result['status'] == 'compile_failed':
                progress_line += f"  {result['witness_type']}"

            print(progress_line)

            # Write detailed info to file
            f.write(f"{i+1}. {task_id}\n")
            f.write(f"   Status: {result['status']}\n")
            if result.get('shape_info'):
                f.write(f"   Shapes: {', '.join(result['shape_info'])}\n")

            if result['status'] == 'train_ok':
                f.write(f"   ✓ SUCCESS - opts: {result['opts_used']}\n")
            elif result['status'] == 'compile_failed':
                f.write(f"   ✗ FAILED - {result['witness_type']}\n")
                if result.get('note'):
                    f.write(f"   Note: {result['note']}\n")
                if result.get('reason'):
                    f.write(f"   Reason: {result['reason']}\n")
                if result.get('opts'):
                    f.write(f"   Tried opts: {result['opts']}\n")
            elif result['status'] == 'error':
                f.write(f"   ⚠ ERROR: {result.get('error', 'unknown')}\n")

            f.write("\n")

        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")

        success_count = sum(1 for r in results if r['status'] == 'train_ok')
        fail_count = sum(1 for r in results if r['status'] == 'compile_failed')
        error_count = sum(1 for r in results if r['status'] == 'error')

        summary_text = f"""
Success:         {success_count:3d} / {args.limit}  ({100*success_count/args.limit:.1f}%)
Compile failed:  {fail_count:3d} / {args.limit}  ({100*fail_count/args.limit:.1f}%)
Errors:          {error_count:3d} / {args.limit}  ({100*error_count/args.limit:.1f}%)
"""
        print(summary_text)

        f.write("="*80 + "\n")
        f.write("SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(summary_text)

        # Breakdown by witness type
        witness_types = {}
        for r in results:
            if r['status'] == 'compile_failed':
                wtype = r['witness_type']
                witness_types[wtype] = witness_types.get(wtype, 0) + 1

        if witness_types:
            breakdown = "\nFailure breakdown:\n"
            for wtype, count in sorted(witness_types.items(), key=lambda x: -x[1]):
                breakdown += f"  {wtype:30s} {count:3d} ({100*count/fail_count:.1f}%)\n"
            print(breakdown)
            f.write(breakdown)

        # Show successful tasks
        if success_count > 0:
            success_text = f"\n{'='*80}\nSUCCESSFUL TASKS\n{'='*80}\n"
            successes = [r for r in results if r['status'] == 'train_ok']
            for r in successes:
                success_text += f"{r['task_id']}  opts={r['opts_used']}\n"
            print(success_text)
            f.write(success_text)

        # Save JSON for further analysis
        json_path = args.output.replace('.txt', '.json')
        with open(json_path, 'w') as jf:
            json.dump(results, jf, indent=2)

        print(f"\nDetailed report: {args.output}")
        print(f"JSON data: {json_path}")


if __name__ == '__main__':
    main()
