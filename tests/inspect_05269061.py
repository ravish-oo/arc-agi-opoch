#!/usr/bin/env python3
import json
from grid import Grid

with open('data/arc-agi_training_challenges.json') as f:
    challenges = json.load(f)

task = challenges['05269061']

print("Training examples:")
for i, pair in enumerate(task['train']):
    X = Grid(pair['input'])
    Y = Grid(pair['output'])
    print(f"\nTrain {i}:")
    print(f"  Input ({X.H}x{X.W}):\n{X}")
    print(f"  Output ({Y.H}x{Y.W}):\n{Y}")

print(f"\nTest input:")
X_test = Grid(task['test'][0]['input'])
print(f"  ({X_test.H}x{X_test.W}):\n{X_test}")
