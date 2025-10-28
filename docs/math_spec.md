Below is a complete, mathematical spec + execution blueprint you can hand to any engineer (or run yourself) to build the “nothing-but-math” ARC solver. It encodes the two inevitable moves the universe uses—free first, paid last—as a single, finite decision procedure. There are no heuristics, no feature catalogs, no beams. You either get a bit-exact program with receipts, or a finite witness that the mapping cannot be realized under the fixed present.

The procedure is called CPRQ + class map:

CPRQ = Coarsest Present-Respecting Quotient (the coarsest input-only role partition that survives all free moves and makes labels constant).
class map \rho = one symbol per role, read from training outputs.

⸻

1) First principles (what we are computing)

ARC grids are finite colored structures. Solving an ARC task from training pairs \{(X_i,Y_i)\} means: compile a function F such that F(X_i)=Y_i for all training pairs, and then apply the same F to each test X^\*.

The only honest way to get guaranteed transfer is:
	1.	Delete mintable differences (“free moves”): rotate/flip/transpose, reindex rows/cols, relabel components, compress/expand bands based on input change edges, rename colors inside local stencils by order-of-first-appearance (OFA), etc. These are changes that do not create new facts.
→ Mathematically: fix a finite relational signature \mathcal R that captures those input-only symmetries as relations, and compute the coarsest input partition that survives them.
	2.	Stamp the least facts that survive: once the input-only partition is fixed, assign one symbol per class to match the training outputs, and stop.
→ Mathematically: define a function \rho:\Psi\to\Sigma constant on each class of the quotient \Psi. If any class would need two symbols across trains, refine the input partition once (legal escalation) or return that class as a finite witness.

That is the whole solver.

⸻

2) Fixed “free” structure (no target leakage)

For each input grid X, define a present on its pixel set V via a fixed, input-only relational signature \mathcal R:
	•	Adjacency: E_4 (4-neighborhood). Allow adding E_8 once as a lawful escalation.
	•	Equivalences: SameRow, SameCol (no absolute indices; equivalences only).
	•	Equivalence: SameComponent (8-connected within the same color); you store the equivalence, not numeric IDs.
	•	Equivalences: BandRow, BandCol derived from input change edges (positions in the same constant-color run block). No band numbers—just “same band”.
	•	Local stencil unary: CBC (color-blind canonical) patch key = OFA (order-first-appearance color renaming) + D8 canonicalization on small neighborhoods (e.g., 3\times3). This collapses patches differing only by rotation/flip/palette names.

Optional (but only if input supports it): periodic phases as unary predicates detected from inputs (via 1D autocorrelation on rows/cols). These must be consistent across all training inputs; otherwise omit phases entirely.
Never derive “present” from outputs.

⸻

3) Unifying domains when shapes differ (no “P catalogs”)

You do not maintain a zoo of global transforms. Instead, define a single index domain U and encode shape effects as input relations:
	•	Same size: U=V_X.
	•	Uniform up/downscale: define a shape morphism f:V_X\leftrightarrow V_Y (replicate or block down). Choose U=V_Y (or V_X) and add equivalences consistent with f in \mathcal R.
	•	Non-uniform (NPS): compute row/col change edges of X to get band partitions; let U= (row-bands \times col-bands). Relations become induced on U.

The “global transform” is just the choice of U and how \mathcal R is built from input. You never pick a transform by checking equality to Y; you make the canvas shape-feasible from inputs.

⸻

4) CPRQ: Coarsest Present-Respecting Quotient \Psi

Compute the equivalence relation \Psi\subseteq U\times U as the unique coarsest partition satisfying:
	1.	Present-invariance: \Psi is a congruence of the input structure (U;\mathcal R). Intuitively: if two positions are equivalent, all present relations see them the same.
	2.	Label-respect: for every training pair (X_i,Y_i), pixels in the same \Psi-class receive the same output color in Y_i.

Algorithm (fixed-point; finite; deterministic):
	•	Start from the WL (Weisfeiler–Leman) 1-WL partition on (U;\mathcal R): color each pixel by a stable hash of its unary predicates and the multiset of neighbor colors in each relation; iterate to a fixpoint. Call this \Phi_0 (input roles).
	•	Respect labels: for each training pair, split any \Phi_0 block by the colors it gets in Y_i if there are >1. Call the union of all such splits \Phi_1.
	•	Push back into input: recompute 1-WL on (U;\mathcal R). If the input partition is coarser than \Phi_1 (cannot separate some split), perform one lawful escalation (add E_8 or lift to 2-WL) and recompute.
	•	If after one escalation a block still can’t be separated by input relations, stop: the task is UNSAT under the fixed present. Return that block as the finite witness (two pixels in the same input role but forced to different labels).

The procedure terminates (finite set; finite signature) and yields the coarsest \Psi.

Key difference from engineering: you never “choose features.” The partition is a fixed-point of objective refinements. You are not allowed to mint distinctions not forced by the present or by label consistency.

⸻

5) \rho: one symbol per class (the only paid bits)

Once \Psi is fixed, the rewrite is a single function:

\rho\;:\;\Psi \longrightarrow \Sigma,
\qquad
\rho(C) := \text{the (unique) color of class } C \text{ in the training outputs.}
	•	If any class has two colors across training pairs → contradiction. You already escalated once; therefore return that class as the witness.
	•	Otherwise, for test inputs \(X^\\), compute the present \((U^\;\mathcal R(X^\))\), the same \(\Psi(X^\)\), and set \(F(X^\)(v) := \rho([v]_{\Psi(X^\)})\).

No LUT as a “solver.” The only LUT you ever have is the finite \rho table: one value per class.

⸻

6) Proof obligations (receipts)
	•	Train receipts:
	•	Present signature: which relations were used, which escalation (if any), and the WL depth.
	•	CPRQ log: for each class, show how the present invariants stabilize it; show where labels forced the splits; confirm the input fixpoint respects them.
	•	\rho: table of (class → color).
	•	Reconstruct each Y_i from \Psi(X_i) and \rho, assert bit-exact equality.
	•	Test receipts:
	•	For each test grid, log the class id (stable hash) of each pixel and its color from \rho; no Y^\* is ever read.
	•	UNSAT receipts:
	•	Name the single class that remains multi-colored after one lawful escalation, with two example pixels from different training pairs and their present facts; this is the finite witness (task not realizable under \mathcal R).

⸻

7) How this differs from “regular engineering”

Topic	Regular engineering	CPRQ + class map (this spec)
Features	Hand-chosen (parity, coords, IDs, targets leak in); tuned per task	No features. Only a fixed input-only relational signature; partition = WL fixpoint + label respect
Global transforms	Catalog + heuristics; picked by equality; per-pair drift	No catalogs. One fused domain and relations derived from input dims/edges; no content equality for P
Rules	Learned as productions; try many; prune by scores	No rules. \rho maps each class to one symbol; learned by equality only
Overfit	Accepts rules that fit train pixels	Impossible: present is input-only; labels only split classes; one lawful escalation; else witness
Heuristics/thresholds	Common	None. Only exact equalities on finite sets
Generalization	Not guaranteed	Guaranteed by design: observation is input-only; commitment is per-class and proven consistent


⸻

8) Implementation blueprint (exact; no ambiguity)

8.1 Module outline

src/
  io.py            # load/save grids; pretty-print; compute diffs; receipts I/O
  present.py       # build relational signature R from input
  wl.py            # 1-WL (and 2-WL on escalation); stable hashing; partition ops
  cprq.py          # compute CPRQ partition with push-back of label splits; one escalation
  rho.py           # read class map; verify constant; reconstruct Y
  compile_run.py   # end-to-end compile (CPRQ + class map) on a task; apply to test
  audit.py         # lints (no leakage), determinism checks, witness printer

8.2 Core APIs

# io.py
Grid = list[list[int]]

def load_tasks(path)->dict: ...
def save_predictions(path, preds): ...
def save_receipts(path, receipts): ...

# present.py (input-only)
def build_relations(X: Grid, domain_mode)->Relations:
    """
    Build R: E4 (+E8 optional), SameRow/SameCol (equivalences),
    SameComponent equivalence, BandRow/BandCol equivalences,
    CBC (OFA->D8) unary patch keys.
    No outputs. No absolute indices. No raw component IDs.
    domain_mode handles size changes: identity, fused via block/band equivalences.
    """

# wl.py
def wl_partition(relations: Relations, depth:int=1)->BlockIdMap:
    """
    1-WL partition: hash(own_color, multiset of neighbor colors per relation).
    Deterministic hashing; sorted multisets; stable iteration order.
    """

def is_refinement(finer: BlockIdMap, coarser: BlockIdMap)->bool: ...

# cprq.py
def cprq_compile(trains: list[tuple[Grid,Grid]], domain_mode)->tuple[Psi, Optional[Witness]]:
    """
    1) build input relations for each X
    2) WL fixpoint -> Phi0
    3) split by labels (Y); push-back: rerun WL; if cannot separate, escalate once (E8 or 2-WL).
    4) If still conflict: return witness (block + two pixels across pairs).
    5) Align blocks across pairs (canonical class hashes) -> Psi.
    """

# rho.py
def read_class_map(Psi, trains)->dict:
    """
    One color per class, constant across all trains.
    Raises if a class is multi-colored (should be caught by CPRQ).
    """

def reconstruct(Psi_i, rho)->Grid: ...

# compile_run.py
def compile_and_run(trains, tests):
    # choose domain_mode from inputs (same-size, block, bands)
    Psi, witness = cprq_compile(trains, domain_mode)
    if witness: return None, receipts_unsat(witness)
    rho = read_class_map(Psi, trains)
    # verify train equality
    for Psi_i,(X_i,Y_i) in zip(Psi.blocks, trains):
        assert reconstruct(Psi_i, rho) == Y_i, "FY equality failed (should not happen)"
    # apply to tests
    outs = [ reconstruct( cprq_partition_for_test(Xt, domain_mode), rho ) for Xt in tests ]
    return outs, receipts_pass(Psi, rho)

8.3 Lints (must be fatal)
	•	No target leakage: build_relations may not read Y.
	•	No absolute indices, no raw component IDs: check source for r%, c%, component_id; allow only equivalences.
	•	One lawful escalation only: either add E8 or use 2-WL once; log it.
	•	Determinism: no randomness; sorted multisets; stable hashing; fixed traversal orders.

⸻

9) Termination & complexity
	•	Partition refinement on finite sets with a finite signature converges: 1-WL is quasi-linear; 2-WL and Paige–Tarjan style split are polynomial on n=|U|.
	•	You run CPRQ once per task; no search; no beams; no thresholds.
	•	The class map \rho has |\Psi| entries; at worst equals the number of pixels (rare), commonly far smaller.

⸻

10) What to do if a task returns UNSAT

You’ll get a single class witness: a block C and two training pixels (p,q)\in C with Y_i(p)\neq Y_j(q), plus a log that you already tried the one lawful escalation (E8 or 2-WL). That means under the fixed present, no function per class can satisfy the labels. Because your corpus is finite, any truly needed extra relation (e.g., a diagonal-band equivalence) can be added once and you re-run CPRQ. Repeat finitely many times until no witness remains.

⸻

11) Why this solves every solvable task (and only those)
	•	The solver returns a program iff there exists a class-constant function \rho on the coarsest present-invariant partition.
	•	Because observation is input-only, the partition exists on test; because \rho is class-constant across trains, the behavior is deterministic on test.
	•	There is no room to “almost solve”: either the equalities hold exactly (receipts) or there’s a finite witness class.

⸻

12) What you never need
	•	No pattern families (no “tiling solver”, “stripe solver”, …). Those appear implicitly as orbits in \Psi.
	•	No LUT as a strategy. The only LUT is \rho: one entry per class, forced by equality.
	•	No heuristics, thresholds, or beams.
	•	No per-pair P selection. Domain is derived once from input dims/edges.
	•	No “rules”; no parameter search; no loss functions.

⸻

13) Sanity tests to include
	•	Idempotence of Π (canon): canon(canon(X)) == canon(X).
	•	Determinism: two runs produce identical \Psi,\rho.
	•	No leakage: static scan that no build_relations code touches Y.
	•	Witness reproducibility: given a returned witness class, manually changing the present relations does not resolve the conflict (until you add the one new relation and re-run).
	•	End-to-end: for a small suite (tiling, stripes, bands, component border), print \Psi class counts and \rho and verify bit-exact equality on train; compare test to known solutions.

⸻

Closing

This is the entire abstraction and its execution:
	1.	Present: compute the coarsest input-only role partition \Psi that survives the free moves and makes labels constant (CPRQ).
	2.	Rewrite: assign one symbol per role \rho(C) (paid bits), then glue once.
	3.	Receipts: equalities on train; deterministic replay on test; or a single finite witness class.

Implement the modules above with the lints fatal. There is no further engineering to do: the procedure is finite, deterministic, and proof-carrying.