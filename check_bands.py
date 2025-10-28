#!/usr/bin/env python3
import json
from grid import Grid

with open('data/arc-agi_training_challenges.json') as f:
    challenges = json.load(f)

task = challenges['0d3d703e']
X0 = Grid(task['train'][0]['input'])

print("Grid:")
print(X0)

H, W = X0.H, X0.W

# Row boundaries
row_edges = []
for r in range(H - 1):
    has_change = False
    for c in range(W):
        if X0[r][c] != X0[r+1][c]:
            has_change = True
            break
    if has_change:
        row_edges.append(r)

print(f"\nRow change edges: {row_edges}")

# Col boundaries  
col_edges = []
for c in range(W - 1):
    has_change = False
    for r in range(H):
        if X0[r][c] != X0[r][c+1]:
            has_change = True
            break
    if has_change:
        col_edges.append(c)

print(f"Col change edges: {col_edges}")

print(f"\nFor this grid:")
print(f"  - Row boundaries: {len(row_edges)} (rows {'differ' if row_edges else 'are identical'})")
print(f"  - Col boundaries: {len(col_edges)} (cols {'differ' if col_edges else 'are identical'})")
