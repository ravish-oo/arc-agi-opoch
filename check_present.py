#!/usr/bin/env python3
import json
from grid import Grid
from present import build_present

with open('data/arc-agi_training_challenges.json') as f:
    challenges = json.load(f)

task = challenges['0d3d703e']
X0 = Grid(task['train'][0]['input'])

present = build_present(X0, {})

print("Present keys:")
for key in sorted(present.keys()):
    if key == 'grid':
        print(f"  {key}: (Grid object)")
    elif isinstance(present[key], dict):
        print(f"  {key}: {len(present[key])} positions")
    elif isinstance(present[key], set):
        print(f"  {key}: {len(present[key])} edges")
    else:
        print(f"  {key}: {present[key]}")

# Check bandRow and bandCol
if 'bandRow' in present:
    print("\nbandRow partition:")
    for pos, band_id in sorted(present['bandRow'].items())[:5]:
        print(f"  {pos}: band {band_id}")

if 'bandCol' in present:
    print("\nbandCol partition:")
    for pos, band_id in sorted(present['bandCol'].items())[:5]:
        print(f"  {pos}: band {band_id}")
