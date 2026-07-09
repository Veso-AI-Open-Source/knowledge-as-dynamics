# Knowledge as Dynamics

Code, run records, figures, and paper source for
**"Knowledge as Dynamics: Distilling, Composing, and Inheriting the Latent Vector
Fields of Neural Networks"** (Helou, 2026). Paper PDF: [`paper/main.pdf`](paper/main.pdf).
arXiv link: coming. Repository: `github.com/Veso-AI-Open-Source/knowledge-as-dynamics`.

The harness (internal codename `m5-fielddistill`) tests **vector-field
distillation**: training a student autoencoder so its latent displacement field
`V(z) = Enc(Dec(z)) − z` matches a teacher's — i.e. distilling the *iteration
dynamics* of Fumero et al.'s latent vector field (arXiv 2505.22785) rather than
static features or outputs — then composing two teachers' fields into one
student and transmitting the result across five model generations.

Novelty status (checked 2026-07-09): no published work distills this field. Closest
neighbors distill static structure (RKD 1904.05068, CRD 1910.10699, MEDAL 2605.24244,
Distilling Latent Manifolds 2603.14536) or feed-forward Jacobians (1803.00443).
This harness is the core experiment for the paper.

## Question

Does explicitly matching the field transfer the teacher's attractor landscape, and —
the kill-switch — is that something static/behavioral distillation would NOT give us
for free?

## Setup

2D 4-blob world lifted nonlinearly to 16D; MLP AE 16→64→64→2→64→64→16 (latent 2D so
the field is plottable). One fixed DENOISING teacher (noise 0.1 — contractive
regime, 7 crisp attractors; a vanilla AE teacher sits in the memorization regime
where endpoint metrics are ill-posed, see FINDINGS.md). 6 student arms x 3 seeds:

| arm | loss | sees data? |
|---|---|---|
| A1 `field` | ‖V_s(z) − V_t(z)‖² on z ~ Uniform(box) | **no** |
| A2 `fieldk` | ‖f_s^k(z) − f_t^k(z)‖², k ∈ {1,2,4,8,16} | **no** |
| A+ `fieldplus` | fieldk + 0.2·recon on 5% anchor | 5% |
| B0 `scratch` | recon | yes |
| B1 `outdistill` | ‖s(x) − t(x)‖² | yes |
| B2 `latmatch` | ‖E_s(x) − E_t(x)‖² + recon | yes |

Every arm gets best-case alignment to teacher latent coordinates (identity vs fitted
affine, chosen by field NMSE) before comparison — operationalizing the "alignment of
latent vector fields" open question in Fumero et al. sec. 6.

Metrics: field NMSE/cosine on a 40x40 grid, attractor Chamfer, **basin agreement**
(% of 400 random probes whose 1000-step iterates land within 0.3 of the teacher's
endpoint; teacher self-agreement under 0.05 probe noise = 0.99 is the ceiling),
held-out recon.

GO = best data-free field arm basin ≥ 0.70, ≥ +0.20 over best static baseline,
cosine ≥ 0.80. NO-GO = any static baseline within 0.05 basin of the field arm.

**Result (2026-07-09): GO** — field 0.86 vs best static 0.53 (output distillation),
kill-switch B2 at 0.15. Details in FINDINGS.md; figure in runs/field_panels.png.

## The four experiments (all run 2026-07-09; full lab log in FINDINGS.md)

| command (`uv run python -m ...`) | what | time | verdict |
|---|---|---|---|
| `m5_fielddistill.validate` then `.plot` | 2D go/no-go, 6 arms x 3 seeds | ~3.5 min | **GO** — data-free field distill 0.86 basin vs 0.53 best static; kill-switch clean |
| `m5_fielddistill.sweep` | teacher noise 0→0.2 vs copyability | ~2 min | copyability = order parameter of the memorization→generalization transition; "copy = test robustness" falsified |
| `m5_fielddistill.transition` | dense grid across the transition + detachment signature | ~3 min | transition at noise ≈ 0.045–0.05; the discontinuous quantity is copy RELIABILITY (variance ±0.22→±0.00); detachment metric confounded in dense data |
| `m5_fielddistill.mnist teacher 0.3 7000`, then `arm <name> <seed>` per arm/seed, then `report` | MNIST tier, latent 16 | ~2.5 min/stage | **NO-GO at basin level for ALL methods** (best 0.19 vs ceiling 0.71); coarse field (cos 0.91) + attractor set (Chamfer 0.87·scale) still transfer best-in-class; teacher attractors decode to crisp digit prototypes |

MNIST arms: `field fieldtraj fieldplus scratch outdistill latmatch` (fieldk dropped —
lost to 1-step in both 2D regimes; its MNIST analogue `fieldtraj` also lost).

| `m5_fielddistill.gi` | GI golden start: two specialist teachers, union field, 5-gen lineages | ~3 min | **GOLDEN GO** — one student holds both teachers' attractor sets (0.82 joint, behavioral merging 0.13 with B-side 0.00); lineage flat over 5 generations; memorization-regime knowledge cannot enter the lineage; grafting must use a designed frame, not any model's OOD chart |

## Setup / outputs map

```sh
uv sync    # mlx, numpy, matplotlib; python 3.11-3.12; no torch needed
```

- `src/m5_fielddistill/` — world, ae (field/attractors/alignment), train (arms),
  validate, plot, sweep, transition, mnist
- `runs/results.json` + `field_panels.png` — 2D go/no-go table + quiver figure
- `runs/sweep.json` + `sweep.png` — order-parameter curves + law plot
- `runs/transition.json` + `transition.png` — per-seed transition scatter + detachment
- `runs/mnist_summary.json` + `mnist_attractors.png` — MNIST table + decoded-attractor
  strip (teacher row = digit prototypes); teacher cached in `mnist_teacher.safetensors`
  + `mnist_meta.npz`; MNIST idx files auto-download to `data/mnist/`
- `FINDINGS.md` — chronological lab log with all numbers, diagnoses, caveats, and the
  paper-shape recommendation (lead 2D phenomenon; MNIST as limits section)
- `paper/main.tex` + `paper/main.pdf` — the arXiv manuscript ("Knowledge as
  Dynamics: Distilling, Composing, and Inheriting the Latent Vector Fields of
  Neural Networks", 13 pp., five-laws structure spanning all five experiments),
  every number traceable to `runs/*.json`; compile with `tectonic main.tex`
  (figures in `paper/figures/`, copied from `runs/`)
