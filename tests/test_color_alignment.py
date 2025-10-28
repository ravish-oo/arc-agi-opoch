#!/usr/bin/env python3
"""Test if identical structure with different colors aligns."""

from grid import Grid
from present import build_present
from wl import wl_disjoint_union

# Two 2x2 grids with identical structure but different colors
# Grid 1: [[1, 2], [1, 2]]  - colors 1,2
# Grid 2: [[3, 4], [3, 4]]  - colors 3,4  (same structure, different colors)

g1 = Grid([[1, 2], [1, 2]])
g2 = Grid([[3, 4], [3, 4]])

p1 = build_present(g1, {})
p2 = build_present(g2, {})

# Run WL on union
Psi_list = wl_disjoint_union([p1, p2])

print("Grid 1 (colors 1,2):")
print(g1)
print(f"WL partition: {Psi_list[0]}")
print(f"Unique roles: {len(set(Psi_list[0].values()))}")

print("\nGrid 2 (colors 3,4):")
print(g2)
print(f"WL partition: {Psi_list[1]}")
print(f"Unique roles: {len(set(Psi_list[1].values()))}")

# Check if any roles overlap
roles1 = set(Psi_list[0].values())
roles2 = set(Psi_list[1].values())
overlap = roles1 & roles2

print(f"\nRole overlap: {len(overlap)} / {len(roles1 | roles2)}")
print(f"Expected: 0 (different raw colors → different initial atoms)")
