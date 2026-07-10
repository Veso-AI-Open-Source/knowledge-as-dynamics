# Knowledge as Dynamics

**Distilling, Composing, and Inheriting the Latent Vector Fields of Neural Networks**

Elias Helou (Veso AI) · Ivan Nemytchenko — v1.2, July 2026
Paper: [`paper/main.pdf`](paper/main.pdf) · arXiv: coming · License: MIT

Any autoencoder defines a vector field on its latent space: iterating
`f = Enc∘Dec` displaces every point by `V(z) = f(z) − z`, and training carves
attractors into that field ([Fumero et al., 2505.22785](https://arxiv.org/abs/2505.22785)).
Prior work *analyzes* this field. This paper treats it as the knowledge itself
and asks what kind of **transmission channel** it is: distill it (regression),
compose it (superposition under a designed frame), inherit it (generations of
data-free re-distillation). Everything runs on one laptop; every number in the
paper traces to a JSON run record in [`runs/`](runs/).

## The five laws

1. **Sufficiency** — matching the field alone, with zero training data,
   transfers the teacher's basins at **0.86** (ceiling 0.99). Output
   distillation gets 0.53, static feature matching 0.15, scratch 0.00.
2. **Order** — transmission *reliability* changes discontinuously exactly where
   the teacher's field becomes fully convergent (its
   memorization→generalization transition). The uncertainty exponent **α** of
   the teacher's basin boundaries is a teacher-only certificate: α = 0.45
   predicts the MNIST ceiling (0.69 vs 0.71 measured) where convergence
   fraction 0.97 does not.
3. **Resolution** (restated in v1.1) — mean field error is the **wrong
   observable**: it *anti-correlates* with basin transfer. A 10⁶-pair lookup
   carrier of the same field reaches **0.41** basin agreement on MNIST where
   parametric students reach 0.09–0.19; what governs is directional fidelity
   near basin boundaries, whose fractal dimension sets every carrier's ceiling.
4. **Composition** — two disjoint specialists' fields union into one data-free
   student (**0.82** joint; behavioral merging 0.13 with the grafted side at
   exactly 0.00) — but only in a *designed* frame; any single model's
   out-of-distribution chart destroys the graft.
5. **Heredity with selection** (restated in v1.1) — knowledge survives five
   generations of data-free re-distillation essentially undegraded, and
   memorization-regime structure cannot board an SGD-carried lineage. The
   selection lives in the lossy **carrier**, not the channel: a lossless
   read-out carrier transmits the same memorized structure faithfully.

## The v1.1 story

Within a day of the v1.0 release, [issue #1](https://github.com/Veso-AI-Open-Source/knowledge-as-dynamics/issues/1)
arrived: an independent, from-scratch replication with a four-part re-diagnosis.
We validated every claim inside the original harness (`extval`, `alpha`
modules), restated Laws 3 and 5, sharpened Law 2 — and found a new result in
the process: **two-hop distillation** (a parametric student supervised by a
16k-pair non-parametric reconstruction of the field) reaches 0.31–0.50 basin
agreement in half its seeds, beating the 10⁶-pair lookup from 61× fewer teacher
queries. The parametric bottleneck is reliability, not capacity. The issue's
author is a co-author from v1.2.

## Run it

```sh
uv sync   # MLX + numpy + matplotlib; Apple Silicon; no torch
```

| command (`uv run python -m ...`) | what | time |
|---|---|---|
| `m5_fielddistill.validate` then `.plot` | 2D go/no-go: 6 arms × 3 seeds (Law 1) | ~3.5 min |
| `m5_fielddistill.sweep` | copyability vs teacher regime (Law 2) | ~2 min |
| `m5_fielddistill.transition` | dense sweep across the transition (Law 2) | ~3 min |
| `m5_fielddistill.mnist teacher 0.3 7000`, then `arm <name> <seed>`, then `report` | MNIST tier, latent 16 (Law 3) | ~2.5 min/stage |
| `m5_fielddistill.gi` | union field + 5-generation lineages (Laws 4–5) | ~3 min |
| `m5_fielddistill.extval gi` / `table` / `carriers <seed>` / `twohop <seed>` | v1.1 external validation: carrier swap, cost curve, GP, two-hop | ~25 min total |
| `m5_fielddistill.alpha` | uncertainty exponent of the cached MNIST teacher | ~1 min |

## Map

- `paper/main.tex` + `main.pdf` — the manuscript (compile with `tectonic main.tex`)
- `src/m5_fielddistill/` — one module per experiment; fixed seeds throughout
- `runs/*.json` — the run records behind every reported number; figures alongside
- `FINDINGS.md` — the chronological lab log: every diagnosis, dead end, and caveat,
  including the v1.0 diagnosis that v1.1 overturned

## Cite

```bibtex
@misc{helou2026knowledgedynamics,
  title  = {Knowledge as Dynamics: Distilling, Composing, and Inheriting
            the Latent Vector Fields of Neural Networks},
  author = {Helou, Elias and Nemytchenko, Ivan},
  year   = {2026},
  url    = {https://github.com/Veso-AI-Open-Source/knowledge-as-dynamics}
}
```
