Zero-risk, lean implementation plan.

# North star

Solve all ARC tasks. Optimize only for correctness. Keep engineering surface small. No stubs. ≤500 LOC per WO. Claude Code runs each WO end-to-end with hard guards. Final deliverable: a single notebook that replays the solver; repo can host modules used by the notebook.

# Strategy to drive implementation risk to ~0

1. Build bedrock first, never touch again. Everything else composes on top.
2. Every WO is atomic: no forward deps; no “TODO later.”
3. Minimal receipts: only what helps block drift and debug.
4. Few reviewers, but sharp. One math-correctness reviewer, one leakage/determinism reviewer, one tiny “no-templates” reviewer.
5. No CI; use a tiny self-test block per WO.
6. Notebook assembles the exact pipeline; cells mirror modules.

# Bedrock order of work

Each item: file, ceiling, contract, tests, reviewer gates.

**MK-00 Stable**

* Files: `stable.py`
* Ceiling: 120 LOC
* Contract: `stable_hash64(obj)`, canonical JSON dump; `sorted_items`; `row_major_string(Grid)`
* Tests: hash determinism; dict order invariance
* Gates: no RNG; no `hash()`; no datetime

**MK-01 Grid**

* Files: `grid.py`
* Ceiling: 160 LOC
* Contract: `Grid` immutable view + safe setter, `H,W`, `positions`, bounds asserts, 0..9 palette guard
* Tests: round-trip set/get; bounds; palette
* Gates: shape 1..30

**MK-02 Equivalences**

* Files: `equiv.py`
* Ceiling: 180 LOC
* Contract: disjoint-set or labeler for equivalence relations; export as partition; `is_refinement(P,Q)`
* Tests: refinement positives and negatives; stability
* Gates: no numeric row/col in public API

**MK-03 Present**

* Files: `present.py`
* Ceiling: 420 LOC
* Contract: `build_present(X, opts)`
  Always: E4, sameRow, sameCol, sameComp8, bandRow, bandCol
  Optional: E8, CBC1, CBC2
  `detect_bands`, `sameComp8`, `cbc_r` with OFA then D8 then hash
* Tests: CBC invariance under D8; bands determinism under color relabel; leak-lint pass
* Gates: AST linter bans Y, Δ, raw indices, `%`, “phase,” “template,” “pattern”

**MK-04 WL**

* Files: `wl.py`
* Ceiling: 380 LOC
* Contract: `wl_disjoint_union(presents)->list[Partition]`
  Atom seed: `(sameComp8_tag, bandRow_tag, bandCol_tag, CBC_token_or_0, is_border)`
  Iterate to fixed point; cap 50
* Tests: permutation of train order gives identical IDs; fixed-point reached
* Gates: no color arithmetic; no coordinates

**MK-05 CPRQ**

* Files: `cprq.py`
* Ceiling: 440 LOC
* Contract: `compile_CPRQ(trains, base_opts)->(Psi_list, rho, opts_used | witness)`
  Ladder: base; then E8; then CBC1; then CBC2; single escalation only
  Label splits; present-expressibility via `is_refinement`
  Build ρ, verify trains bit-exact, else witness `(train, p, q, flags)`
* Tests: synthetic conflict shows ladder and witness
* Gates: forbids enabling two new relations in one compile

**MK-06 Predict**

* Files: `predict.py`
* Ceiling: 140 LOC
* Contract: `predict_tests(tests, opts_used, rho)`; WL over union {trains ∪ test}
* Tests: determinism; union flag recorded
* Gates: uses exactly `opts_used`

**MK-07 Mini-receipts**

* Files: `receipts.py`
* Ceiling: 180 LOC
* Contract: JSON with only: `present_flags`, `roles`, `rho_table`, `bands_row`, `bands_col`, `wl_union_with_test`, optional `witness`
* Tests: schema check; round-trip
* Gates: forbid extra analytics

**NB-00 Notebook assembly**

* File: `arc_solver.ipynb`
* Cells: load data; compile; verify trains; predict tests; print minimal receipts; visualize few grids
* No CI; run-all yields full corpus outputs

# Reviewer setup for Claude Code

Minimize agents. One implementer terminal agent + two reviewers.

**Reviewer 1: Math-correctness (CPRQ law)**

* Blocks if: WL not on disjoint union; label-constant not enforced; is_refinement wrongly implemented; ladder violated
* Requires: train equality proof or witness; no action catalogs

**Reviewer 2: Leakage + determinism**

* Blocks if: present uses Y, Δ, indices, `%`; CBC without OFA+D8; Python `hash()` used; nondeterminism; dual optional relations enabled on first pass
* Requires: receipts show flags, union policy, bands

Prompt both reviewers to never modify code; only read, run, grep, and write a short report under `reviews/`.

# “No stubs” rule

Each WO ships with: code, self-tests, and a tiny script fragment that exercises it with synthetic grids. No TODOs. No “later integration”. If later WOs need it, they only call public functions; no edits to earlier files.

# Minimal receipts only

Keep just six fields:
`present_flags, roles, rho_table, bands_row, bands_col, wl_union_with_test`
Plus `witness` when applicable. Nothing else.

# Golden examples baked in

Lock the five hand-solved tasks as golden tests after MK-05: assert `cells_wrong == 0`, flags match, role count match, ρ match. This guards against “vibe coding” without adding heavy infra.

# Anti-drift guardrails

* AST linter runs inside `present.py` tests; fails on banned tokens.
* A single `DETERMINISM_SEED = None` constant; grep test forbids `random` and `numpy.random`.
* One “no templates” grep: forbid strings like “mirror”, “rotate90”, “tile_repeat”, “stripe”, “scale2x”. CBC must do the work.

# LOC and time budgeting

All WOs under 500 LOC; the largest is Present and CPRQ. Claude Code handles them without cutting corners.

# Handoff bundle for Claude Code

* `docs/anchor_blueprint.md` (your spec; index-free phrasing)
* `docs/worked_examples.md` (the five tasks; expected receipts)
* `docs/impl_rules.md` (ladder, atom seed, linter bans, receipts schema)
* `docs/reviewer_prompts.md` (two reviewer roles, block gates, checklists)
* `tasks/` with a tiny loader for ARC JSON

# Definition of done

* MK-00..07 pass self-tests locally.
* Golden five pass with exact equality and expected receipts.
* Notebook runs end-to-end on the corpus and emits predictions plus minimal receipts.
* No file imports Y; no `%`; no Python `hash()`.
* No module modified after acceptance of its WO.

This hits your constraints: lean, sufficient, complete. No CI, no fluff, no stubs, ≤500 LOC per piece, and minimal reviewers with sharp block rules.
