#!/usr/bin/env python3
"""Test the 5 golden tasks from worked_examples.md"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from grid import Grid
from cprq import compile_CPRQ
import json

GOLDEN_TASKS = [
    ('0d3d703e', 'strict colormap'),
    ('332efdb3', 'role-constant by (r%2, c%2)'),
    ('3c9b0459', 'isometry (rot180)'),
    ('6150a2bd', 'isometry (rot180)'),
    ('67a3c6ac', 'isometry (flip_h)'),
]

def test_task(task_id, description):
    print(f"\n{'='*60}")
    print(f"Task {task_id}: {description}")
    print(f"{'='*60}")

    # Load task
    with open('data/arc-agi_training_challenges.json') as f:
        challenges = json.load(f)

    task = challenges[task_id]
    trains = []
    for pair in task['train']:
        X = Grid(pair['input'])
        Y = Grid(pair['output'])
        trains.append((X, Y))

    print(f"Training examples: {len(trains)}")
    for i, (X, Y) in enumerate(trains):
        print(f"  {i+1}. X: {X.H}×{X.W} → Y: {Y.H}×{Y.W}")

    # Run CPRQ
    result = compile_CPRQ(trains, {})

    if result[0] is None:
        witness = result[1]
        print(f"❌ FAILED: {witness.get('type')}")
        print(f"   Witness: {witness}")
        return False
    else:
        compile_result, _ = result
        Psi_list, rho, opts_used = compile_result
        print(f"✓ SUCCESS!")
        print(f"   Options used: {opts_used}")
        print(f"   Roles: {len(set(rho.keys()))}")
        print(f"   ρ table: {rho}")
        return True

if __name__ == '__main__':
    print("Testing 5 Golden Tasks from worked_examples.md")
    print("="*60)

    successes = 0
    for task_id, description in GOLDEN_TASKS:
        if test_task(task_id, description):
            successes += 1

    print(f"\n{'='*60}")
    print(f"GOLDEN TASKS SUMMARY: {successes}/{len(GOLDEN_TASKS)} succeeded")
    print(f"{'='*60}")
