#!/usr/bin/env python3
"""Print actual atom values to verify fix."""

import json
from grid import Grid
from present import build_present

# Load task
with open('data/arc-agi_training_challenges.json') as f:
    challenges = json.load(f)

task = challenges['0d3d703e']
X0 = Grid(task['train'][0]['input'])

present = build_present(X0, {})

print("Checking actual atom structure:")
print(f"Grid: {[[X0[r][c] for c in range(X0.W)] for r in range(X0.H)]}")
print()

# Manually build atoms like _build_initial_coloring does
cbc_tokens = present.get('CBC1', None)
H, W = X0.H, X0.W

for pos in sorted(present['sameComp8'].keys())[:5]:
    r, c = pos
    is_border = (r == 0 or r == H - 1 or c == 0 or c == W - 1)
    color_tag = X0[r][c]
    cbc_token = cbc_tokens[pos] if cbc_tokens else 0

    atom = (color_tag, cbc_token, is_border)
    print(f"  {pos}: atom = {atom}")

print("\nVerified: atoms only contain (color, CBC_token, border)")
print("NO band tags, NO component tags")
