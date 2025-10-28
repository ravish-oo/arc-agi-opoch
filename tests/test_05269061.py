#!/usr/bin/env python3
"""Test task 05269061 (cycle-3 striping) - should work with bag fix."""

import json
from grid import Grid
from cprq import compile_CPRQ, predict

# Load task
with open('data/arc-agi_training_challenges.json') as f:
    challenges = json.load(f)

with open('data/arc-agi_training_solutions.json') as f:
    solutions = json.load(f)

task = challenges['05269061']
trains = [(Grid(p['input']), Grid(p['output'])) for p in task['train']]
test_input = Grid(task['test'][0]['input'])
test_gt = Grid(solutions['05269061'][0])

print(f"Task 05269061: {len(task['train'])} training examples")
print(f"Test: {test_input.H}x{test_input.W}")

# Compile with empty base_opts
result, witness = compile_CPRQ(trains, {})

if witness:
    print(f"\n❌ Compilation FAILED: {witness}")
else:
    Psi_list, rho, options_used = result
    print(f"\n✓ Compilation succeeded")
    print(f"  Roles: {len(set(Psi_list[0].values()))}")
    print(f"  ρ size: {len(rho)}")
    
    # Predict
    Y_pred = predict(test_input, trains, result)
    
    # Compare
    if Y_pred == test_gt:
        print(f"\n✅ Test PASSED - prediction matches ground truth!")
    else:
        # Count matches
        matches = sum(1 for r in range(test_gt.H) for c in range(test_gt.W)
                     if Y_pred[r][c] == test_gt[r][c])
        total = test_gt.H * test_gt.W
        print(f"\n❌ Test FAILED - {matches}/{total} cells match ({100*matches/total:.1f}%)")
        
        print(f"\nExpected:\n{test_gt}")
        print(f"\nGot:\n{Y_pred}")
