"""
Golden test runner for the five hand-solved tasks.

Expected to pass with cells_wrong == 0 after MK-05.

Golden task IDs:
- 00576224
- 017c7c7b
- 05269061
- 007bbfb7
- 03560426

Usage:
    python -m tests_training.run_golden --ids 00576224,017c7c7b,05269061,007bbfb7,03560426
"""

import argparse
import json
from pathlib import Path
from grid import Grid
from cprq import compile_CPRQ, count_cells_wrong


def load_arc_task(task_id: str, tasks_dir: Path = Path('tasks')) -> dict:
    """
    Load ARC task from JSON file.

    Args:
        task_id: Task ID (e.g., '00576224')
        tasks_dir: Directory containing task JSON files

    Returns:
        Task dict with 'train' and 'test' lists
    """
    task_file = tasks_dir / f"{task_id}.json"

    if not task_file.exists():
        raise FileNotFoundError(f"Task file not found: {task_file}")

    with open(task_file, 'r') as f:
        return json.load(f)


def grid_from_list(data: list) -> Grid:
    """Convert list of lists to Grid."""
    return Grid(data)


def run_golden_test(task_id: str, tasks_dir: Path = Path('tasks')) -> dict:
    """
    Run CPRQ on a golden task and verify results.

    Args:
        task_id: Task ID
        tasks_dir: Directory with task files

    Returns:
        Results dict with status, cells_wrong, role_count, opts_used
    """
    try:
        task = load_arc_task(task_id, tasks_dir)
    except FileNotFoundError:
        return {
            'task_id': task_id,
            'status': 'SKIP',
            'reason': 'Task file not found (create tasks/ directory with ARC JSON files)',
        }

    # Build training pairs
    trains = []
    for example in task['train']:
        x = grid_from_list(example['input'])
        y = grid_from_list(example['output'])
        trains.append((x, y))

    # Compile CPRQ
    result, witness = compile_CPRQ(trains, {})

    if result is None:
        return {
            'task_id': task_id,
            'status': 'FAIL',
            'witness': witness,
        }

    Psi_list, rho, opts_used = result

    # Verify training examples
    total_wrong = 0
    for train_idx, (x, y) in enumerate(trains):
        Psi = Psi_list[train_idx]

        # Predict
        from cprq import _predict_with_rho
        y_pred = _predict_with_rho(x, Psi, rho)

        # Count errors
        wrong = count_cells_wrong(y, y_pred)
        total_wrong += wrong

    # Get role count
    from wl import get_role_count
    role_counts = [get_role_count(Psi) for Psi in Psi_list]

    return {
        'task_id': task_id,
        'status': 'PASS' if total_wrong == 0 else 'FAIL',
        'cells_wrong': total_wrong,
        'role_counts': role_counts,
        'opts_used': opts_used,
        'num_trains': len(trains),
    }


def main():
    parser = argparse.ArgumentParser(description='Run golden tests')
    parser.add_argument(
        '--ids',
        type=str,
        default='00576224,017c7c7b,05269061,007bbfb7,03560426',
        help='Comma-separated task IDs'
    )
    parser.add_argument(
        '--tasks-dir',
        type=Path,
        default=Path('tasks'),
        help='Directory containing task JSON files'
    )

    args = parser.parse_args()

    task_ids = args.ids.split(',')

    print("=" * 60)
    print("Golden Test Runner")
    print("=" * 60)

    results = []
    for task_id in task_ids:
        print(f"\nRunning task {task_id}...")
        result = run_golden_test(task_id, args.tasks_dir)
        results.append(result)

        if result['status'] == 'SKIP':
            print(f"  SKIP: {result['reason']}")
        elif result['status'] == 'PASS':
            print(f"  ✓ PASS")
            print(f"    Cells wrong: {result['cells_wrong']}")
            print(f"    Role counts: {result['role_counts']}")
            print(f"    Options: {result['opts_used']}")
        else:
            print(f"  ✗ FAIL")
            if 'cells_wrong' in result:
                print(f"    Cells wrong: {result['cells_wrong']}")
            if 'witness' in result:
                print(f"    Witness: {result['witness']}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    passed = sum(1 for r in results if r['status'] == 'PASS')
    failed = sum(1 for r in results if r['status'] == 'FAIL')
    skipped = sum(1 for r in results if r['status'] == 'SKIP')

    print(f"Passed:  {passed}/{len(results)}")
    print(f"Failed:  {failed}/{len(results)}")
    print(f"Skipped: {skipped}/{len(results)}")

    if failed > 0:
        exit(1)


if __name__ == '__main__':
    main()
