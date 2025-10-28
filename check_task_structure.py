#!/usr/bin/env python3
import json
from grid import Grid

with open('data/arc-agi_training_challenges.json') as f:
    challenges = json.load(f)

task = challenges['0d3d703e']

print(f"Training examples: {len(task['train'])}")
for i, pair in enumerate(task['train']):
    X = Grid(pair['input'])
    Y = Grid(pair['output'])
    print(f"\nTrain {i}: {X.H}x{X.W} → {Y.H}x{Y.W}")
    print(f"Input:\n{X}")
    print(f"Output:\n{Y}")

print(f"\n\nTest input: {len(task['test'])} example")
X_test = Grid(task['test'][0]['input'])
print(f"Test: {X_test.H}x{X_test.W}")
print(f"Input:\n{X_test}")
