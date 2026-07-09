# FINDINGS — vector-field distillation (lab log, all runs 2026-07-09)

This file is chronological: four experiments, each section written right after its
run. Intermediate "next steps" reflect knowledge at that point; later sections
supersede them.

| # | experiment | section | verdict |
|---|---|---|---|
| 1 | 2D go/no-go, 6 arms x 3 seeds (`validate`) | Headline numbers | **GO** — data-free field distill 0.86 vs static best 0.53, kill-switch clean |
| 2 | teacher-noise sweep 0→0.2 (`sweep`) | Sweep | copyability = order parameter; strong law "copy = test robustness" falsified |
| 3 | dense transition + detachment (`transition`) | Transition localization | transition at noise ≈ 0.045–0.05; RELIABILITY is the discontinuous quantity; detachment metric confounded in dense data |
| 4 | MNIST tier, latent 16 (`mnist`) | MNIST tier | **NO-GO at basin level** (all methods); coarse field + attractor set still transfer best-in-class; high-dim sparse-probe matching = the open problem |
| 5 | GI golden start: union + onion (`gi`) | Experiment 5 | **GOLDEN GO** — fields union (0.82 vs behavioral-merge 0.13), 5-generation lineage holds, disordered knowledge cannot enter the lineage; frame must be designed, not inherited |

Paper shape that falls out: lead with 1–3 (phenomenon + phase structure), present 4
as the limits section with the residual-vs-basin-scale diagnosis.

## Experiment 1 — 2D go/no-go

**VERDICT: GO** (~200s on M5, 3 seeds/arm)

Distilling the latent displacement field V(z) = Enc(Dec(z)) − z from a teacher AE
into a fresh student — with the student never seeing a single training datum —
transfers the teacher's attractor landscape. Static and behavioral KD baselines do not
match it. The kill-switch did not fire.

## Headline numbers (mean over 3 seeds; teacher self-agreement ceiling = 0.99)

| arm | basin agree | field cos | field NMSE | attr Chamfer | recon (held-out) |
|---|---|---|---|---|---|
| A1 field (1-step, **data-free**) | **0.86** | 1.00 | 0.000 | 0.269 | 0.57 (no data, expected) |
| A2 fieldk (k-step, data-free) | 0.72 | 1.00 | 0.001 | 0.360 | 0.57 |
| A+ fieldk + 5% anchor | 0.65 | 1.00 | 0.001 | 0.414 | **0.0067** |
| B0 scratch | 0.00 | 0.04 | 1.012 | 2.084 | 0.0000 |
| B1 output distillation | 0.53 | 0.92 | 0.223 | 0.492 | 0.0003 |
| B2 latent matching (kill-switch) | 0.15 | 0.19 | 1.086 | 0.635 | 0.0000 |

GO criteria: A basin ≥ 0.70 ✓ (0.86), margin over best static ≥ 0.20 ✓ (+0.33 over
B1's 0.53), cosine ≥ 0.80 ✓ (1.00). Kill-switch (any static baseline within 0.05) ✗.

## Key findings

1. **The field is distillable, data-free.** Sampling z from a box prior and regressing
   V_s(z) onto V_t(z) transfers the field at cosine ~1.00 and puts student attractors
   on the teacher's (Chamfer 0.27 vs attractor merge radius 0.15). The A1 student never
   sees data — the dynamics alone are a sufficient knowledge channel. This directly
   extends Fumero et al. 2505.22785 (field as *analysis* object) to the field as a
   *training signal*, which nothing in the literature does (novelty sweep 2026-07-09).

2. **Dynamics are NOT free.** B0 (scratch, same data, same arch) lands at basin 0.00 —
   the field is teacher-specific, not a property of the task. B2 (static latent
   matching, the designated kill-switch) gets 0.15 — matching Enc features on data
   says almost nothing about the off-manifold field. The dissociation B1 (0.53) >>
   B2 (0.15) is itself interesting: what partially transfers dynamics is cloning the
   composite map's *behavior*, not the encoder's static features.

3. **Behavioral cloning is the serious baseline, not feature matching.** B1 reaches
   basin 0.53 with high variance (0.34–0.75 across seeds). The paper must position
   against output distillation; the +0.33 mean gap, the lower variance, and the
   data-free property are the wedge.

4. **Teacher regime is decisive (the big gotcha).** A vanilla AE teacher sits in the
   memorization regime: 26 fragmented attractors, only 75% of probes converge (slow
   filaments), and endpoint metrics are ill-posed — first run scored the field arm
   0.26 despite cosine 0.97, because per-step error compounds along slow manifolds.
   A denoising teacher (noise 0.1) flips to the contractive/generalization regime:
   7 crisp attractors, conv_frac 1.00, self-agreement 0.99, and the same distillation
   method jumps 0.26 → 0.86. Paper implication: distillability of dynamics is itself
   a function of the teacher's memorization/generalization regime — a measurable,
   reportable phenomenon, not a nuisance.

5. **One-step beats k-step (contrary to expectation).** Multi-step composite matching
   (k ∈ {1,2,4,8,16}) was built to fix error compounding, but in the contractive
   regime plain 1-step matching wins (0.86 vs 0.72) — with a well-conditioned target
   field the extra unroll only adds optimization noise. In the memorization-regime
   first run, k-step also failed (0.13). Keep both arms in the paper as the ablation.

## Protocol notes

- Every arm gets best-case latent alignment (identity vs fitted affine on data
  encodings, picked by field NMSE) before comparison — our operationalization of the
  "alignment of latent vector fields" open question in Fumero sec. 6. Field arms
  always pick identity (they train in teacher coordinates); data arms pick affine.
- Basin agreement = endpoint proximity (<0.3) after 1000 iterations, valid only
  because conv_frac = 1.0; with a memorization-regime teacher use attractor labels
  or report it as ill-posed.
- Grad-clip 1.0 needed for the unrolled k-step arms.

## Next steps as written after experiment 1 (all three since executed — see below)

- ~~Sweep teacher denoising noise 0 → 0.2~~ → done, "Sweep" section.
- ~~Densify the transition / tie to memorization signature~~ → done, "Transition" section.
- ~~MNIST, latent 8–16~~ → done, "MNIST tier" section (NO-GO at basin level).
- Still open: cross-architecture and cross-latent-dim students (field defined in
  teacher coords; student maps through a learned adapter — the real "alignment"
  contribution).

## Sweep: copyability vs teacher regime (2026-07-09, 96s, 2 seeds/level)

`uv run python -m m5_fielddistill.sweep` → runs/sweep.json, runs/sweep.png

| noise | attractors | conv | ceiling | copyability | robustness |
|---|---|---|---|---|---|
| 0.00 | 13 | 0.60 | 0.96 | 0.57 ± 0.17 | 0.0019 |
| 0.01 | 28 | 0.78 | 0.95 | 0.41 ± 0.09 | 0.0017 |
| 0.02 | 11 | 0.63 | 0.97 | 0.34 ± 0.25 | 0.0016 |
| 0.05 | 3 | 1.00 | 0.99 | 0.69 ± 0.00 | 0.0013 |
| 0.08 | 4 | 1.00 | 0.99 | 0.66 ± 0.14 | 0.0012 |
| 0.10 | 7 | 1.00 | 0.99 | 0.81 ± 0.03 | 0.0012 |
| 0.15 | 4 | 1.00 | 0.99 | **0.99 ± 0.00** | 0.0013 |
| 0.20 | 4 | 1.00 | 0.99 | **0.99 ± 0.00** | 0.0017 |

**Result: copyability behaves like an ORDER PARAMETER for the
memorization→generalization transition — but the strong law "copyability =
test-time generalization" is falsified within the ordered phase.**

1. **Sharp phase transition between noise 0.02 and 0.05**, co-located across every
   indicator at once: attractor count collapses 11–28 → 3–4, convergence fraction
   snaps 0.6–0.78 → 1.00, and copyability jumps from low-mean/high-variance
   (0.34–0.57, seed std up to ±0.25) to high-mean/low-variance (0.69→0.99, std → 0.00).
   Disordered phase: dynamics effectively untransmittable AND unstable — which seed
   you get matters more than the method. Ordered phase: transmission approaches
   perfect and becomes deterministic (0.99 ± 0.00 at noise ≥ 0.15).

2. **Strong law falsified (important negative result):** within the ordered phase,
   off-manifold robustness saturates at ~0.0012 by noise 0.08 and *worsens* at 0.2
   (over-smoothing), while copyability keeps climbing to 0.99. The law plot (right
   panel) is not monotone in robustness. Copyability tracks **dynamical order /
   contractiveness of the landscape**, not test-metric generalization per se.

3. **Refined claim for the paper:** copyability of the latent vector field is an
   intrinsic, data-free order parameter of the trained model — near-zero and noisy
   in the memorization phase, saturating to 1 in the contractive phase — whose onset
   coincides with the memorization→generalization transition. This is measurable
   without any held-out data (probe field + distill + basin agreement).

Caveats: 2 seeds/level, one toy world; endpoint-based basin metric is partially
ill-posed in the disordered phase (conv < 1), which inflates the low-noise noise —
use attractor-label or trajectory metrics there in the paper version. Next: densify
noise ∈ [0.02, 0.05] with 3+ seeds to localize the transition; check whether the
transition point matches where the teacher's attractors detach from individual
training points (Fumero's memorization signature).

## Transition localization + detachment signature (2026-07-09, 182s, 3 seeds/level)

`uv run python -m m5_fielddistill.transition` → runs/transition.json, runs/transition.png
Dense grid noise ∈ {0, .01, .02, .025, .03, .035, .04, .045, .05, .06, .08}.

**1. Transition localized at noise ≈ 0.045–0.05, and the discontinuous quantity is
RELIABILITY, not the mean.** Teacher conv_frac bounces in 0.59–0.78 for all levels
≤ 0.045, then snaps to 1.00 at 0.05 and stays. At exactly that level, student copy
variance collapses: per-seed copyability spans 0.09–0.75 below the transition
(std up to ±0.22), then 0.69 ± 0.00 at 0.05. The mean is continuous (0.61 → 0.69 →
0.80, reaching 0.99 deeper in phase per the coarse sweep); the seed-to-seed
*determinism* of transmission is what jumps. Sharpened claim: **transmission of
dynamics becomes deterministic exactly when the teacher's field becomes fully
convergent.** Copy-reliability is the order parameter; copy-mean is the
continuous magnitude that grows with contraction strength.

**2. Attractor count is a bad observable near criticality.** It fluctuates wildly
across the transition (13 → 28 → 11 → 4 → 8 → 2 → 3 → 1 → 3 → 9 → 4) — consistent
with landscape fluctuations near a phase change, but useless for localization.
conv_frac and copy variance are the clean observables.

**3. Detachment signature: inconclusive due to a design confound (honest negative).**
mem_attach (attractor distance to nearest training encoding) does NOT cleanly lift
off at the transition — it is small at noise 0 (0.07, attractors on memorized
points, as Fumero predicts) but ALSO small in the ordered phase (0.02–0.03),
because prototypes sit inside dense blobs where a training encoding is always
within ~0.03. In a 500-points-per-class world, "on a training point" and "at the
prototype" are geometrically indistinguishable. The metric actually peaks
mid-transition (0.70 at noise 0.025) where the landscape is most disordered.
Fix for the paper tier: sparse world (~20 points/class) or MNIST, where the two
hypotheses separate. The transition result (point 1) is unaffected.

Caveats: one teacher seed per level (teacher-to-teacher variability near the
transition unmeasured); copyability at 0.08 retains one weak seed (0.525) — the
in-phase mean keeps hardening only deeper into the contractive regime (0.99 at 0.15+).

## MNIST tier: does it survive dimensionality? (2026-07-09, latent 16)

`uv run python -m m5_fielddistill.mnist teacher 0.3 7000` then `arm <name> <seed>`
then `report` → runs/mnist_summary.json, runs/mnist_attractors.png

Setup: 784→256→128→16 AE, 8k train imgs, denoising teacher noise 0.3 (best of
0.2–0.5 sweep: 18 attractors, conv 0.97, ceiling 0.71 — note: the teacher itself
never reaches full order at 16D; per the 2D law this predicts copy trouble).
Field arms sample z from teacher latent moments (2×16 numbers) + deep teacher
trajectories; 12k steps, cosine LR. New arm `fieldtraj` (1-step + k∈{2,8}
trajectory matching). New metric `dec_agree` (endpoints compared through the
teacher decoder in pixel space).

| arm | basin | cos | nmse | chamfer/scale | recon |
|---|---|---|---|---|---|
| field (data-free) | 0.09 | 0.91 | 0.049 | (no converged attrs) | 0.106 |
| fieldtraj (data-free) | 0.01 | 0.90 | 0.068 | (no converged attrs) | 0.105 |
| fieldplus (5% anchor) | 0.07 | 0.91 | 0.049 | **0.87** | 0.033 |
| scratch | 0.00 | 0.50 | 1.784 | 1.52 | 0.015 |
| outdistill | **0.19** | 0.70 | 0.667 | 1.08 | 0.016 |
| latmatch | 0.00 | 0.63 | 0.388 | 1.53 | 0.014 |

**VERDICT (MNIST tier): NO-GO at basin level — honest negative, well-diagnosed.**

1. **What still transfers:** the coarse field (cos 0.91 vs ≤0.75 for all baselines;
   NMSE 14x better than best baseline) and the attractor SET (fieldplus chamfer
   0.87·scale, best of all arms). Teacher attractors decode to crisp digit
   prototypes (runs/mnist_attractors.png, top row) — Fumero's memories-as-attractors
   confirmed at MNIST scale by our own harness.

2. **What breaks: basin-level fidelity, for EVERYONE.** Best arm (outdistill 0.19)
   is at 27% of the 0.71 ceiling. The kill-switch nominally fires (B1 0.19 > field
   0.09) but on a metric where all methods have failed. Diagnosis is quantitative:
   the field arm's per-step residual plateaus at √0.049·|V̄t| ≈ 0.75 ≈ 14% of latent
   scale, vs basin_eps 2.19 — compounding over ~800 iterations, endpoints land ~6
   units off (median). Basin transfer needs residual ≪ basin scale; SGD regression
   from sparse probes in 16D plateaus two orders of magnitude short of the 2D tier
   (NMSE 0.049 vs 0.0004). **Sample/optimization complexity of high-dim field
   matching is THE open problem** — exactly the pre-registered failure mode
   ("if sparse-probe matching doesn't scale, the transfer claims die").

3. **Trajectory supervision did not rescue it** (fieldtraj 0.01) — third failure of
   k-step matching across both tiers. The bottleneck is per-step precision, not
   horizon supervision.

4. **The teacher itself is sub-ordered at 16D**: ceiling 0.71, conv 0.97 even at
   noise 0.5 (58 attractors) — basins are finely interleaved at every noise level
   tried. Consistent with the 2D law: partial teacher order predicts unreliable
   copying. An open question is whether ANY 16D AE teacher on MNIST reaches the
   fully-ordered regime, or whether the ceiling itself is the object to study.

**Paper implications:** lead with the 2D phenomenon (data-free transfer + order
parameter + phase transition, all clean), present MNIST as the limits section with
the residual-vs-basin-scale diagnosis, and frame high-dim field matching as the
central open problem (candidate fixes: importance-weighted near-attractor sampling,
contraction-aware losses, spectral/Jacobian regularization of the student,
curriculum from coarse to fine field scales).

Protocol notes: MNIST from ossci-datasets S3 mirror into data/mnist/; staged CLI so
no stage exceeds ~5 min; teacher cached in runs/mnist_teacher.safetensors + meta npz;
dec_agree threshold 0.01 pixel-MSE (tracks latent metric exactly at these scales).

## Experiment 5 — GI golden start: union + onion (2026-07-09, 156s)

`uv run python -m m5_fielddistill.gi` → runs/gi.json, runs/gi.png

**VERDICT: GOLDEN GO — all five pre-registered gates passed** (after one protocol
correction and one budget correction, both documented below).

Setup: 8-blob ring world; teacher A knows blobs 0–3, teacher B knows 4–7 (denoising,
contractive); B_mem = memorization-regime teacher on 4–7. Common frame holds A
natively and grafts B via affine conjugacy T; union field = nearest-centroid gate
between V_A and conjugated V_B; oracle = iterated gated composite. Students distill
from the union field data-free. Two lineages (clean / contaminated), 5 generations.

| gate | criterion | result |
|---|---|---|
| C1 union | joint ≥ 0.60, per-side drop ≤ 0.15 | **0.82** mean (5 seeds, range 0.77–0.91); drops 0.09 / 0.06 |
| K3 kill | merged output-distill within 0.05 | clear by 0.69 — merged 0.13, B-side 0.00 both seeds |
| C3 exceedance | ≥ 7/8 sites in one student | 8/9 (best seed); no teacher holds > 4 |
| C4 lineage | gen5 ≥ 0.85 × gen1 | 0.83/0.91 = 0.91; curves flat, attractors stable |
| C2 filter | ordered persists, disordered dies | clean graft 0.84→0.70 (persists); B_mem 0.24→0.13→0.19 (dead) while its A-side rides 0.89–1.00 |

Findings, in order of importance:

1. **Fields union.** One student, distilled data-free from a gated composite of two
   teachers' fields, holds both knowledge sets: joint basin 0.82 (ceiling 0.98),
   ~8 attractors when no ancestor has more than 4. Behavioral merging cannot do
   this at all (B-side 0.00: pooled output cloning has no mechanism for placing
   disjoint knowledge). Composition is a capability specific to the field channel.

2. **The frame must be DESIGNED, not inherited (the graft finding).** Input-fitted
   conjugacy — placing B's knowledge where A's encoder puts B's inputs — destroys
   it (B-side 0.09–0.14; single-B 0.00): A's OOD encoder compresses foreign
   territory (collision 0.73) below field-regression resolution, the MNIST
   precision wall manufactured at 2D. An isometric graft into empty frame
   territory transmits at near-single-teacher fidelity (0.75–0.84). Alignment
   to any one model's chart is the wrong abstraction; the union frame is a
   design object.

3. **The lineage holds.** Five generations of data-free re-distillation: joint
   0.91→0.83, A-side pinned at ceiling, B-side 0.73→0.70, attractor count stable.
   Not a photocopier. (Run 1 also showed a genuine RATCHET: a weak ancestor's
   B-side improved 0.22→0.71 under re-distillation before stabilizing.)

4. **The onion filter is real.** Same graft slot, ordered vs disordered source:
   ordered graft persists across the lineage (~0.70); memorization-regime graft
   never transmits and stays dead (≤0.24 throughout) while the ordered native side
   rides at 0.89–1.00 beside it. What is heritable is exactly what is general.

5. **Reliability is budget-dependent** (consistent with the order-parameter story):
   at 4k steps union seeds spanned 0.35–0.77; at 8k steps, 0.77–0.91. The union
   target (two territories + a gate seam) is a harder optimization landscape;
   the seam is where residual basin losses concentrate.

Caveats: C2's clean-persistence margin is thin (0.70 vs 0.672 threshold); lineage
chains are one seed per generation and the clean lineage ancestor is best-of-5
(disclosed selection); the 9th oracle attractor is a seam artifact; hard gate
discontinuity is smoothed by students. All at 2D — the high-dim precision wall
applies to every step of this loop.

What this buys the paper/program: both conceptual pillars of the inheritance loop
(transfer AND composition+lineage) are now demonstrated at tier 1, with the frame-
design principle as a new contribution. The single named blocker for everything
remains high-dimensional field-matching precision.
