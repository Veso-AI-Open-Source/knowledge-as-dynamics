"""Math harness: confirm the alpha -> distillability test on Newton's-method basins.

The proposed next step for the paper (v1.2 discussion) is to use the uncertainty
exponent alpha of a teacher's basin boundaries -- a teacher-only, pre-transfer
measurement -- as a predictor of how well its dynamics distill. Before betting on
that test at language-model scale, confirm it in a system where EVERYTHING is
exact: Newton's method on p(z) = z^n - 1. The attractors are the n-th roots of
unity (known analytically), the teacher map is exact (no trained-teacher
confound), and the basin-boundary dimension D_b is measurable independently by
box counting, so the alpha estimator has a ground truth via the
Grebogi-Ott-Yorke relation alpha = d - D_b (d = 2). Degree n dials boundary
roughness: n = 2 gives a smooth boundary (the imaginary axis, D_b = 1,
alpha = 1); n >= 3 gives the classic Newton fractals.

Pre-registered gates:
  G1 estimator sanity .. n=2 gives alpha >= 0.85 and box-counted D_b <= 1.15.
  G2 math cross-check .. |alpha - (2 - D_b)| <= 0.15 for every n.
  G3 test confirmation . Spearman(alpha, student basin agreement at top budget)
                         = +1.0 across n in {2,3,4,5}; smooth system >= 0.90.
  G4 budget law ........ basin agreement weakly increases with table size N
                         for every n (the wall is budgetary; Law 3, v1.1).

Run: uv run python -m m5_fielddistill.newton      writes runs/newton.json
"""

import json
import pathlib
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .ae import MLP

RUNS = pathlib.Path(__file__).resolve().parents[2] / "runs"

DEGREES = (2, 3, 4, 5)
BOX = 1.5
SCALE = BOX
N_PROBES = 4096
TEACHER_STEPS = 100
ROOT_TOL_TEACHER = 1e-3
ROOT_TOL_STUDENT = 0.1
EPS_FRACS = (1 / 128, 1 / 64, 1 / 32, 1 / 16, 0.05, 1 / 8, 1 / 4)  # x SCALE
EPS_PROTOCOL = 0.05  # the paper's probe-noise fraction
REPS = 3
GRID = 1024
COARSE = (1, 2, 4, 8, 16, 32)
BUDGETS = (4096, 65536, 524288)
STUDENT_STEPS = 6000
STUDENT_ITERS = 300
BS = 512


def roots_of_unity(n):
    k = np.arange(n)
    return np.exp(2j * np.pi * k / n)


def newton_step(z, n):
    z = np.where(np.abs(z) < 1e-9, 1e-9 + 0j, z)
    zn1 = z ** (n - 1)
    z2 = z - (z * zn1 - 1.0) / (n * zn1)
    mag = np.abs(z2)
    return np.where(mag > 50.0, z2 * (50.0 / mag), z2)


def teacher_endpoints(z0, n):
    z = z0.copy()
    for _ in range(TEACHER_STEPS):
        z = newton_step(z, n)
    return z


def root_ids(z, n, tol):
    r = roots_of_unity(n)
    d = np.abs(z[:, None] - r[None, :])
    ids = d.argmin(axis=1)
    ids[d.min(axis=1) > tol] = -1
    return ids


def fit_alpha(eps, fr):
    pts = [(e, f) for e, f in zip(eps, fr) if 0.0 < f < 0.95]
    if len(pts) < 3:
        return float("nan"), float("nan"), (float("nan"), float("nan"))
    xs = np.log([p[0] for p in pts])
    ys = np.log([p[1] for p in pts])
    slope, intercept = np.polyfit(xs, ys, 1)
    pred = slope * xs + intercept
    ss = 1.0 - np.sum((ys - pred) ** 2) / max(np.sum((ys - ys.mean()) ** 2), 1e-12)
    return float(slope), float(ss), (float(slope), float(intercept))


def uncertainty_curve(probes, base_ids, n, rng):
    conv = base_ids >= 0
    fr = []
    for frac in EPS_FRACS:
        eps = frac * SCALE
        flips = []
        for _ in range(REPS):
            pert = probes + eps * (rng.standard_normal(len(probes))
                                   + 1j * rng.standard_normal(len(probes)))
            ids2 = root_ids(teacher_endpoints(pert, n), n, ROOT_TOL_TEACHER)
            flips.append(float((ids2[conv] != base_ids[conv]).mean()))
        fr.append(float(np.mean(flips)))
    return fr


def boxcount_db(n):
    xs = np.linspace(-BOX, BOX, GRID)
    X, Y = np.meshgrid(xs, xs)
    Z = (X + 1j * Y).ravel()
    ids = root_ids(teacher_endpoints(Z, n), n, ROOT_TOL_TEACHER).reshape(GRID, GRID)
    bnd = np.zeros((GRID, GRID), dtype=bool)
    bnd[:, :-1] |= ids[:, :-1] != ids[:, 1:]
    bnd[:-1, :] |= ids[:-1, :] != ids[1:, :]
    counts = []
    for c in COARSE:
        g = GRID // c
        counts.append(int(bnd[: g * c, : g * c].reshape(g, c, g, c).any(axis=(1, 3)).sum()))
    slope, _ = np.polyfit(np.log([1.0 / c for c in COARSE]), np.log(counts), 1)
    return float(slope), counts


def make_table(n, n_pairs, rng):
    z = rng.uniform(-BOX, BOX, n_pairs) + 1j * rng.uniform(-BOX, BOX, n_pairs)
    v = newton_step(z, n) - z
    keep = np.abs(v) < 5.0
    z, v = z[keep], v[keep]
    Z = np.stack([z.real, z.imag], axis=1).astype(np.float32)
    V = np.stack([v.real, v.imag], axis=1).astype(np.float32)
    return Z, V


def train_student(Z, V, seed, dims=(2, 128, 128, 2)):
    mx.random.seed(seed)
    model = MLP(list(dims))
    mx.eval(model.parameters())
    opt = optim.Adam(learning_rate=1e-3)
    rng = np.random.default_rng(seed)
    Zm, Vm = mx.array(Z), mx.array(V)

    def loss_fn(m, zb, vb):
        return mx.mean((m(zb) - vb) ** 2)

    lg = nn.value_and_grad(model, loss_fn)
    for _ in range(STUDENT_STEPS):
        idx = mx.array(rng.integers(0, len(Z), BS))
        _, g = lg(model, Zm[idx], Vm[idx])
        opt.update(model, g)
        mx.eval(model.parameters(), opt.state)
    return model


def student_endpoints(model, probes):
    p = mx.array(np.stack([probes.real, probes.imag], axis=1).astype(np.float32))
    lo, hi = mx.array([-3.0, -3.0]), mx.array([3.0, 3.0])
    for i in range(STUDENT_ITERS):
        p = mx.maximum(mx.minimum(p + model(p), hi), lo)
        if i % 25 == 0:
            mx.eval(p)
    mx.eval(p)
    q = np.array(p)
    return q[:, 0] + 1j * q[:, 1]


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def main():
    t0 = time.time()
    rng = np.random.default_rng(0)
    probes = rng.uniform(-BOX, BOX, N_PROBES) + 1j * rng.uniform(-BOX, BOX, N_PROBES)
    out = {"systems": {}, "gates": {}}

    for n in DEGREES:
        base_ids = root_ids(teacher_endpoints(probes, n), n, ROOT_TOL_TEACHER)
        conv = float((base_ids >= 0).mean())
        fr = uncertainty_curve(probes, base_ids, n, np.random.default_rng(10 + n))
        alpha, r2, (sl, ic) = fit_alpha([f * SCALE for f in EPS_FRACS], fr)
        ceiling = 1.0 - fr[EPS_FRACS.index(EPS_PROTOCOL)]
        db, counts = boxcount_db(n)
        print(f"[n={n}] conv {conv:.3f}  alpha {alpha:.3f} (R2 {r2:.3f})  "
              f"D_b {db:.3f}  2-D_b {2 - db:.3f}  ceiling(1-f) {ceiling:.3f}")

        sysrec = {"conv": conv, "alpha": alpha, "alpha_r2": r2, "D_b": db,
                  "alpha_geom": 2 - db, "ceiling": ceiling,
                  "f_curve": dict(zip(map(str, EPS_FRACS), fr)), "budgets": {}}

        for N in BUDGETS:
            Z, V = make_table(n, N, np.random.default_rng(100 + n))
            model = train_student(Z, V, seed=0)
            ep = student_endpoints(model, probes)
            sid = root_ids(ep, n, ROOT_TOL_STUDENT)
            mask = base_ids >= 0
            basin = float((sid[mask] == base_ids[mask]).mean())
            vt = newton_step(probes, n) - probes
            vs = np.array(model(mx.array(np.stack([probes.real, probes.imag], 1).astype(np.float32))))
            vs = vs[:, 0] + 1j * vs[:, 1]
            derr = np.abs(vs - vt)
            delta_eff = float(np.median(derr[np.abs(vt) < 5.0]))
            basin_pred = float(1.0 - min(1.0, np.exp(ic) * delta_eff ** sl)) if np.isfinite(sl) else float("nan")
            sysrec["budgets"][str(N)] = {"basin": basin, "delta_eff": delta_eff,
                                         "basin_pred": basin_pred}
            print(f"    N={N:>7}: basin {basin:.3f}  delta_eff {delta_eff:.4f}  "
                  f"predicted {basin_pred:.3f}")
        out["systems"][str(n)] = sysrec

    S = out["systems"]
    alphas = [S[str(n)]["alpha"] for n in DEGREES]
    basins_top = [S[str(n)]["budgets"][str(BUDGETS[-1])]["basin"] for n in DEGREES]
    g1 = S["2"]["alpha"] >= 0.85 and S["2"]["D_b"] <= 1.15
    g2 = all(abs(S[str(n)]["alpha"] - S[str(n)]["alpha_geom"]) <= 0.15 for n in DEGREES)
    rho = spearman(alphas, basins_top)
    g3 = rho == 1.0 and S["2"]["budgets"][str(BUDGETS[-1])]["basin"] >= 0.90
    g4 = all(S[str(n)]["budgets"][str(BUDGETS[i])]["basin"]
             <= S[str(n)]["budgets"][str(BUDGETS[i + 1])]["basin"] + 0.02
             for n in DEGREES for i in range(len(BUDGETS) - 1))
    out["gates"] = {"G1_estimator": bool(g1), "G2_crosscheck": bool(g2),
                    "G3_test_spearman": rho, "G3_pass": bool(g3), "G4_budget": bool(g4)}
    print(f"\nGATES: G1 {g1}  G2 {g2}  G3 rho={rho:+.2f} pass={g3}  G4 {g4}")
    RUNS.mkdir(exist_ok=True)
    (RUNS / "newton.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {RUNS / 'newton.json'} [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
