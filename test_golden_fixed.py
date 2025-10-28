#!/usr/bin/env python3
"""Test golden tasks after WL fix."""

import json
from grid import Grid
from cprq import compile_CPRQ

# Load challenges
with open('data/arc-agi_training_challenges.json') as f:
    challenges = json.load(f)

# Golden tasks
golden_tasks = ['007bbfb7', '0d3d703e', '25ff71a9', '3c9b0459', 'a2fd1cf0']

print('Testing golden tasks with WL fix...\n')
passed = 0
failed = 0

for task_id in golden_tasks:
    task = challenges[task_id]
    trains = [(Grid(p['input']), Grid(p['output'])) for p in task['train']]

    result, witness = compile_CPRQ(trains, {})

    if result is not None:
        print(f'✓ {task_id}: SUCCESS')
        passed += 1
    else:
        print(f'✗ {task_id}: WITNESS')
        if witness:
            print(f'  Witness type: {witness.get("type", "?")}')
            if 'present_flags' in witness:
                print(f'  Flags: {witness["present_flags"]}')
        failed += 1

print(f'\nResults: {passed}/5 passed, {failed}/5 failed')
