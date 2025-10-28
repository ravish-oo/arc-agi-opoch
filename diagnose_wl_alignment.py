#!/usr/bin/env python3
"""
Diagnostic: Trace WL hash computation to find alignment issue.

This script runs WL twice:
1. On training grids only (like compile does)
2. On training + test grids (like predict does)

Then compares the hash values to see why they differ.
"""

import json
from grid import Grid, apply_transform
from present import build_present
from wl import wl_disjoint_union

# Load data
with open('data/arc-agi_training_challenges.json') as f:
    challenges = json.load(f)

# Pick simplest task
task_id = '0d3d703e'
task = challenges[task_id]

print(f"Task: {task_id}")
print("=" * 80)

# Prepare grids (no Π canonicalization for simplicity)
trains = [(Grid(p['input']), Grid(p['output'])) for p in task['train']]
test_input = Grid(task['test'][0]['input'])

print(f"Training: {len(trains)} pairs")
print(f"Test: 1 input")
print()

# Build presents (using base options for simplicity)
opts = {}  # No E8, no CBC for now
phases = (None, None, None)

print("Building presents...")
train_inputs = [X for X, Y in trains]
train_presents = [build_present(X, opts, phases) for X in train_inputs]
test_present = build_present(test_input, opts, phases)

print(f"  Train presents: {len(train_presents)}")
print(f"  Test present: 1")
print()

# ========================================
# SCENARIO 1: WL on trains only (compile)
# ========================================
print("=" * 80)
print("SCENARIO 1: WL on TRAINS ONLY (compile-time)")
print("=" * 80)

Psi_trains_only, iter_count_1 = wl_disjoint_union(train_presents, depth=1)

print(f"WL iterations: {iter_count_1}")
print()

# Collect all role IDs from training
train_only_role_ids = set()
for grid_idx, psi in enumerate(Psi_trains_only):
    role_ids = set(psi.values())
    train_only_role_ids.update(role_ids)
    print(f"  Train grid {grid_idx}: {len(role_ids)} unique role IDs")
    # Show first 5 role IDs
    sample_ids = sorted(role_ids)[:5]
    print(f"    Sample IDs: {sample_ids}")

print(f"\nTotal unique role IDs (trains only): {len(train_only_role_ids)}")
print()

# ========================================
# SCENARIO 2: WL on trains+test (predict)
# ========================================
print("=" * 80)
print("SCENARIO 2: WL on TRAINS + TEST (predict-time)")
print("=" * 80)

all_presents = train_presents + [test_present]
Psi_union, iter_count_2 = wl_disjoint_union(all_presents, depth=1)

print(f"WL iterations: {iter_count_2}")
print()

# Split back into train and test
Psi_trains_union = Psi_union[:-1]
Psi_test_union = Psi_union[-1]

# Collect role IDs from union
train_union_role_ids = set()
for grid_idx, psi in enumerate(Psi_trains_union):
    role_ids = set(psi.values())
    train_union_role_ids.update(role_ids)
    print(f"  Train grid {grid_idx}: {len(role_ids)} unique role IDs")
    sample_ids = sorted(role_ids)[:5]
    print(f"    Sample IDs: {sample_ids}")

test_role_ids = set(Psi_test_union.values())
print(f"  Test grid: {len(test_role_ids)} unique role IDs")
sample_ids = sorted(test_role_ids)[:5]
print(f"    Sample IDs: {sample_ids}")

print(f"\nTotal unique role IDs (trains in union): {len(train_union_role_ids)}")
print(f"Total unique role IDs (test): {len(test_role_ids)}")
print()

# ========================================
# COMPARISON
# ========================================
print("=" * 80)
print("COMPARISON: Do training role IDs change when test is added?")
print("=" * 80)

overlap = train_only_role_ids & train_union_role_ids
print(f"Overlap (train-only ∩ train-union): {len(overlap)} / {len(train_only_role_ids)}")

if len(overlap) == len(train_only_role_ids):
    print("✅ All training role IDs are STABLE (same in both scenarios)")
else:
    print(f"❌ Training role IDs CHANGED when test was added!")
    print(f"   Only {len(overlap)}/{len(train_only_role_ids)} IDs remained the same")

print()

# Check test overlap with training
test_train_overlap = test_role_ids & train_union_role_ids
print(f"Overlap (test ∩ train-union): {len(test_train_overlap)} / {len(test_role_ids)}")

if len(test_train_overlap) == 0:
    print("❌ ZERO OVERLAP: Test has NO role IDs in common with training")
    print("   This is the 'present_gap_unseen_class' witness")
else:
    print(f"✅ Test shares {len(test_train_overlap)} role IDs with training")

print()

# ========================================
# POSITION-LEVEL ANALYSIS
# ========================================
print("=" * 80)
print("POSITION-LEVEL ANALYSIS: First training grid, position (0,0)")
print("=" * 80)

pos = (0, 0)
train_grid_0_only = Psi_trains_only[0]
train_grid_0_union = Psi_trains_union[0]

if pos in train_grid_0_only and pos in train_grid_0_union:
    role_only = train_grid_0_only[pos]
    role_union = train_grid_0_union[pos]

    print(f"Position {pos} in training grid 0:")
    print(f"  Role ID (trains-only):  {role_only}")
    print(f"  Role ID (trains+test):  {role_union}")

    if role_only == role_union:
        print("  ✅ SAME role ID in both scenarios")
    else:
        print("  ❌ DIFFERENT role ID! Union changed the hash")

print()

# ========================================
# ROOT CAUSE SUMMARY
# ========================================
print("=" * 80)
print("ROOT CAUSE ANALYSIS")
print("=" * 80)

print("""
Per math_spec_addon_airtight.md line 40:
  "WL runs on the same present on train∪test. Roles are defined by input
   structure alone; class IDs are aligned across grids by input-only
   structural hashes... every test pixel belongs to some Ẽ-class that
   is already typed; ρ̃ applies. No 'unseen class' remains."

Our code:
  1. Compile time: WL on trains only → produces role IDs A, B, C...
  2. Predict time: WL on trains∪test → produces role IDs X, Y, Z...

Problem:
  The WL hash computation is CONTEXT-DEPENDENT. Even though we use
  stable_hash64, the iterative refinement depends on the global
  equivalence class bags (sameComp8, bandRow, bandCol).

  When test is added to the union:
  - Equivalence classes may change size/composition
  - Multiset bags change
  - Hash signatures in each iteration change
  - Final fixed-point hashes are DIFFERENT

Solution (per math spec):
  WL should run ONCE on train∪test at the beginning (before compile).
  Both compile and predict should use the SAME pre-computed role IDs.
  No re-running WL at predict time.
""")

print("=" * 80)
