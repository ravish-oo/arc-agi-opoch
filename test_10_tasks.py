#!/usr/bin/env python3
"""
Test CPRQ on 10 ARC tasks with full witness output per math_spec.md.
"""

import json
import sys
from grid import Grid
from cprq import compile_CPRQ, predict

OUTPUT_FILE = 'test_10_results.txt'

def write_and_print(f, msg):
    """Write to both file and stdout."""
    print(msg)
    f.write(msg + '\n')
    f.flush()

def grid_to_string(grid):
    """Convert grid to readable string."""
    lines = []
    for r in range(grid.H):
        row = [str(grid[r][c]) for c in range(grid.W)]
        lines.append(' '.join(row))
    return '\n'.join(lines)

# Load training data
print("Loading training data...")
with open('data/arc-agi_training_challenges.json') as f:
    challenges = json.load(f)

with open('data/arc-agi_training_solutions.json') as f:
    solutions = json.load(f)

# Pick first 10 tasks
task_ids = list(challenges.keys())[:10]

print(f"Testing {len(task_ids)} tasks from training set")
print(f"Output will be written to: {OUTPUT_FILE}\n")

# Track results per task
task_results = {}  # task_id -> {'status': ..., 'witness': ..., 'reason': ...}

with open(OUTPUT_FILE, 'w') as f:
    write_and_print(f, '=' * 100)
    write_and_print(f, 'ARC-AGI CPRQ TEST: 10 Tasks')
    write_and_print(f, 'Per math_spec.md: E* = (⋀ K_𝒢(X_i)) ∧ Int^𝒢(⋀ ker(c_i)),  F = ρ ∘ π_E*')
    write_and_print(f, '=' * 100)
    write_and_print(f, '')

    for task_num, task_id in enumerate(task_ids, 1):
        write_and_print(f, '')
        write_and_print(f, f'[{task_num}/10] TASK: {task_id}')
        write_and_print(f, '-' * 100)

        task = challenges[task_id]
        ground_truth = solutions.get(task_id, [])

        # Training data
        trains = [(Grid(p['input']), Grid(p['output'])) for p in task['train']]
        write_and_print(f, f'Training: {len(trains)} pairs')

        # Test inputs (needed for union-WL)
        test_inputs = [Grid(p['input']) for p in task['test']]

        # Compile (per math_spec: WL runs on train∪test)
        try:
            result, witness = compile_CPRQ(trains, test_inputs, {})
        except Exception as e:
            write_and_print(f, f'❌ COMPILATION CRASHED: {type(e).__name__}: {e}')
            task_results[task_id] = {
                'status': 'COMPILE_CRASH',
                'witness': str(e),
                'reason': f'Exception: {type(e).__name__}'
            }
            continue

        if result is None:
            write_and_print(f, '❌ COMPILATION FAILED')
            write_and_print(f, f'   Witness type: {witness.get("type", "unknown")}')
            if 'reason' in witness:
                write_and_print(f, f'   Reason: {witness["reason"]}')
            task_results[task_id] = {
                'status': 'COMPILE_FAILED',
                'witness': witness,
                'reason': witness.get('type', 'unknown')
            }
            continue

        # Unpack compilation result (11 elements now)
        Psi_list, rho, wl_depth, opts, domain_mode, scale_or_none, pi_tag, phases, wl_iter_count, label_mode, Psi_list_test = result
        num_roles = len(set(c for psi in Psi_list for c in psi.values()))

        write_and_print(f, f'✅ Compiled: π={pi_tag}, depth={wl_depth}, iter={wl_iter_count}, mode={label_mode}, roles={num_roles}, ρ_size={len(rho)}')

        write_and_print(f, f'Testing: {len(test_inputs)} case(s)')

        all_perfect = True
        predict_witness = None

        for test_idx, test_input in enumerate(test_inputs):
            # Predict (using pre-computed Psi)
            try:
                predicted_grid = predict(test_input, trains, result, test_idx)
            except ValueError as e:
                # Per math_spec.md: present_gap_unseen_class witness
                error_msg = str(e)
                write_and_print(f, f'  Test {test_idx}: ❌ PREDICTION FAILED')
                write_and_print(f, f'    Witness: present_gap_unseen_class')
                write_and_print(f, f'    Detail: {error_msg[:200]}')
                predict_witness = {
                    'type': 'present_gap_unseen_class',
                    'detail': error_msg
                }
                all_perfect = False
                continue
            except Exception as e:
                write_and_print(f, f'  Test {test_idx}: ❌ PREDICTION CRASHED: {type(e).__name__}: {e}')
                predict_witness = {
                    'type': 'crash',
                    'detail': str(e)
                }
                all_perfect = False
                continue

            # Get ground truth
            if test_idx >= len(ground_truth):
                write_and_print(f, f'  Test {test_idx}: ⚠️  No ground truth')
                continue

            gt_grid = Grid(ground_truth[test_idx])

            # Compare
            if predicted_grid.H != gt_grid.H or predicted_grid.W != gt_grid.W:
                write_and_print(f, f'  Test {test_idx}: ❌ WRONG SHAPE pred={predicted_grid.H}×{predicted_grid.W} vs gt={gt_grid.H}×{gt_grid.W}')
                all_perfect = False
                continue

            # Count errors
            wrong = sum(1 for r in range(gt_grid.H) for c in range(gt_grid.W)
                       if predicted_grid[r][c] != gt_grid[r][c])
            total = gt_grid.H * gt_grid.W

            if wrong == 0:
                write_and_print(f, f'  Test {test_idx}: ✅ PERFECT ({total} cells)')
            else:
                write_and_print(f, f'  Test {test_idx}: ❌ WRONG {wrong}/{total} cells ({100*wrong/total:.1f}% error)')
                all_perfect = False

        # Record result
        if all_perfect and len(test_inputs) > 0:
            task_results[task_id] = {
                'status': 'PERFECT',
                'witness': None,
                'reason': 'All tests passed'
            }
            write_and_print(f, '✅✅✅ PERFECT ✅✅✅')
        else:
            task_results[task_id] = {
                'status': 'PREDICT_FAILED',
                'witness': predict_witness,
                'reason': predict_witness.get('type', 'mismatch') if predict_witness else 'cell_mismatch'
            }

    # CRISP SUMMARY
    write_and_print(f, '')
    write_and_print(f, '')
    write_and_print(f, '=' * 100)
    write_and_print(f, 'SUMMARY')
    write_and_print(f, '=' * 100)
    write_and_print(f, '')

    perfect = [tid for tid, res in task_results.items() if res['status'] == 'PERFECT']
    compile_failed = [tid for tid, res in task_results.items() if res['status'] in ['COMPILE_FAILED', 'COMPILE_CRASH']]
    predict_failed = [tid for tid, res in task_results.items() if res['status'] == 'PREDICT_FAILED']

    write_and_print(f, f'PASSED:  {len(perfect)}/10')
    write_and_print(f, f'FAILED:  {len(compile_failed) + len(predict_failed)}/10')
    write_and_print(f, f'  - Compilation failures: {len(compile_failed)}')
    write_and_print(f, f'  - Prediction failures:  {len(predict_failed)}')
    write_and_print(f, '')

    if perfect:
        write_and_print(f, f'✅ PASSED ({len(perfect)}):')
        for tid in perfect:
            write_and_print(f, f'   {tid}')
        write_and_print(f, '')

    if compile_failed:
        write_and_print(f, f'❌ COMPILATION FAILURES ({len(compile_failed)}):')
        for tid in compile_failed:
            res = task_results[tid]
            witness_type = res['reason']
            write_and_print(f, f'   {tid}: {witness_type}')
        write_and_print(f, '')

    if predict_failed:
        write_and_print(f, f'❌ PREDICTION FAILURES ({len(predict_failed)}):')
        for tid in predict_failed:
            res = task_results[tid]
            reason = res['reason']
            write_and_print(f, f'   {tid}: {reason}')
        write_and_print(f, '')

    # WHY THEY FAILED - per math_spec.md
    if compile_failed or predict_failed:
        write_and_print(f, 'FAILURE REASONS (per math_spec.md):')
        write_and_print(f, '-' * 100)

        failure_types = {}
        for tid, res in task_results.items():
            if res['status'] != 'PERFECT':
                reason = res['reason']
                if reason not in failure_types:
                    failure_types[reason] = []
                failure_types[reason].append(tid)

        for reason, tids in sorted(failure_types.items()):
            write_and_print(f, '')
            write_and_print(f, f'{reason}: {len(tids)} task(s)')
            for tid in tids:
                write_and_print(f, f'  - {tid}')

            # Explain per math_spec.md
            if reason == 'label_split_unexpressible':
                write_and_print(f, '  → Per math_spec.md: Int^𝒢(L) failed. Present cannot express label split')
                write_and_print(f, '    even with escalation (E8, 2-WL). Need richer atom/relations.')
            elif reason == 'label_conflict_unexpressible':
                write_and_print(f, '  → Per math_spec.md: K_𝒢 has multi-color classes. Escalation exhausted.')
                write_and_print(f, '    Present distinguishes positions with same label.')
            elif reason == 'present_gap_unseen_class':
                write_and_print(f, '  → Per math_spec.md line 83: Test has WL classes unseen in training.')
                write_and_print(f, '    Union-WL alignment issue: train and test get different role IDs.')
            elif reason == 'cell_mismatch':
                write_and_print(f, '  → Prediction produced output but cells don\'t match ground truth.')
                write_and_print(f, '    ρ may be correct on training but incorrect generalization.')
            elif reason == 'shape_change_unsat':
                write_and_print(f, '  → Shape unification failed. Output shapes inconsistent across training.')

    write_and_print(f, '')
    write_and_print(f, '=' * 100)

print(f'\n✅ Results written to {OUTPUT_FILE}')
print(f'\n📊 QUICK SUMMARY:')
print(f'   PASSED: {len(perfect)}/10')
print(f'   FAILED: {len(compile_failed) + len(predict_failed)}/10')
if perfect:
    print(f'   Perfect: {", ".join(perfect)}')
