"""Uncertainty exponent alpha of the cached MNIST teacher (paper Section 5.3).

Measures final-state sensitivity (Grebogi/McDonald/Ott/Yorke): the fraction
f(eps) of eps-perturbed probes whose endpoint changes scales as eps^alpha,
with D_b = d - alpha the basin-boundary dimension. Run after `mnist teacher`:

    uv run python -m m5_fielddistill.alpha

Validates the claim from repository issue #1: alpha ~= 0.45 while
conv_frac ~= 1.0, and the recorded ceiling (self_agree 0.711) equals
1 - f(eps) at the protocol's probe noise (0.05*scale = 0.1255*basin_eps).
"""

import mlx.core as mx
import numpy as np

from m5_fielddistill.mnist import ITER_STEPS, load_teacher
from m5_fielddistill.ae import iterate

teacher, meta = load_teacher()
probes = meta["probes"]
ep_ref = meta["ep_probes"]
basin_eps = float(meta["basin_eps"])
clip_lo, clip_hi = meta["clip_lo"], meta["clip_hi"]
scale = float(meta["scale"])
print(f"teacher: ceiling {float(meta['self_agree']):.4f}  conv {float(meta['conv_frac']):.4f}  "
      f"basin_eps {basin_eps:.3f}  0.05*scale/basin_eps = {0.05*scale/basin_eps:.4f}")

rng = np.random.default_rng(0)
EPS_FRACS = (1/64, 1/32, 1/16, 1/8, 0.1255, 1/4, 1/2)  # x basin_eps; incl. our protocol point
REPS = 3  # perturbation batches per level for a steadier f(eps)

fr = []
for f in EPS_FRACS:
    eps = f * basin_eps
    flips = []
    for _ in range(REPS):
        pert = probes + eps * rng.standard_normal(probes.shape).astype(np.float32)
        ep = np.array(iterate(teacher, mx.array(pert), ITER_STEPS, clip_lo, clip_hi))
        flips.append(float((np.linalg.norm(ep - ep_ref, axis=1) > basin_eps).mean()))
    fr.append(float(np.mean(flips)))
    print(f"  eps = {f:.4f} * basin_eps : f(eps) = {fr[-1]:.4f}  (runs: {[round(x,3) for x in flips]})")

mask = [i for i, v in enumerate(fr) if v > 0]
xs = np.log([EPS_FRACS[i] for i in mask])
ys = np.log([fr[i] for i in mask])
alpha = float(np.polyfit(xs, ys, 1)[0])
f_protocol = fr[EPS_FRACS.index(0.1255)]
print(f"\nalpha = {alpha:.3f}   (issue #1 claim: 0.45; smooth boundary would be ~1.0)")
print(f"1 - f(0.1255*basin_eps) = {1 - f_protocol:.3f}   vs recorded ceiling {float(meta['self_agree']):.3f}")
