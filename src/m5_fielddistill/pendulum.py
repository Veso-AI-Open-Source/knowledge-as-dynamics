"""Physics harness: confirm the alpha -> distillability test on the magnetic pendulum.

Companion to newton.py (the math harness). The magnetic pendulum -- a damped
pendulum over three magnets -- is the canonical physical system with fractal
basin boundaries (McDonald, Grebogi, Ott & Yorke, Physica D 1985). Damping is a
PHYSICAL dial for boundary roughness: strong damping gives smooth-ish basins,
weak damping gives wild fractal ones. We measure the uncertainty exponent alpha
on the (x, y, v=0) initial-condition plane, cross-check it against box-counted
boundary dimension, distill the time-T flow map into an MLP student at two query
budgets, and test whether alpha (a teacher-only measurement) predicts student
basin agreement across damping levels.

Pre-registered gates:
  G5 physics sanity .... alpha weakly decreases as damping decreases
                         (rougher boundaries at lower damping), tol 0.03.
  G6 cross-check ....... |alpha - (2 - D_b)| <= 0.20 for every damping level.
  G7 test confirmation . Spearman(alpha, student basin agreement at top budget)
                         = +1.0 across the damping levels.
  G8 budget law ........ basin agreement at 131k pairs >= at 16k - 0.02.

Run: uv run python -m m5_fielddistill.pendulum      writes runs/pendulum.json
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

MAGNETS = np.stack([[np.cos(a), np.sin(a)]
                    for a in (np.pi / 2, np.pi / 2 + 2 * np.pi / 3, np.pi / 2 + 4 * np.pi / 3)]
                   ).astype(np.float64)  # radius-1 triangle
K_SPRING = 0.5
H = 0.2          # magnet height (regularizes the force)
M_STRENGTH = 1.0
DT = 0.02
DAMPINGS = (0.6, 0.35, 0.2, 0.12)   # smooth -> wild
BOX = 1.5
SCALE = BOX
N_PROBES = 2048
MAX_STEPS = 20000
EPS_FRACS = (1 / 128, 1 / 64, 1 / 32, 1 / 16, 0.05, 1 / 8, 1 / 4)
EPS_PROTOCOL = 0.05
REPS = 2
GRID = 256
COARSE = (1, 2, 4, 8, 16)
BUDGETS = (16384, 131072)
T_MAP = 20                # RK4 steps per map application (T = 0.4)
STUDENT_STEPS = 8000
STUDENT_ITERS = 1000
BS = 512
REST_SPEED = 5e-2


def accel(r, v, b):
    a = -K_SPRING * r - b * v
    for i in range(3):
        d = MAGNETS[i][None, :] - r
        a = a + M_STRENGTH * d / (np.sum(d * d, axis=1, keepdims=True) + H * H) ** 1.5
    return a


def rk4(r, v, b, steps):
    for _ in range(steps):
        k1r, k1v = v, accel(r, v, b)
        k2r, k2v = v + 0.5 * DT * k1v, accel(r + 0.5 * DT * k1r, v + 0.5 * DT * k1v, b)
        k3r, k3v = v + 0.5 * DT * k2v, accel(r + 0.5 * DT * k2r, v + 0.5 * DT * k2v, b)
        k4r, k4v = v + DT * k3v, accel(r + DT * k3r, v + DT * k3v, b)
        r = r + DT / 6.0 * (k1r + 2 * k2r + 2 * k3r + k4r)
        v = v + DT / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return r, v


def integrate_to_rest(r0, b):
    r, v = r0.copy(), np.zeros_like(r0)
    done = 0
    while done < MAX_STEPS:
        r, v = rk4(r, v, b, 400)
        done += 400
        if np.max(np.sum(v * v, axis=1)) < 1e-4:
            break
    return r, v


def attractor_ids(r, v):
    d = np.linalg.norm(r[:, None, :] - MAGNETS[None, :, :], axis=-1)
    ids = d.argmin(axis=1)
    bad = (np.linalg.norm(v, axis=1) > REST_SPEED) | (d.min(axis=1) > 1.0)
    ids[bad] = -1
    return ids


def teacher_ids(r0, b):
    r, v = integrate_to_rest(r0, b)
    return attractor_ids(r, v)


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


def uncertainty_curve(probes, base_ids, b, rng):
    conv = base_ids >= 0
    fr = []
    for frac in EPS_FRACS:
        eps = frac * SCALE
        flips = []
        for _ in range(REPS):
            pert = probes + eps * rng.standard_normal(probes.shape)
            ids2 = teacher_ids(pert, b)
            flips.append(float((ids2[conv] != base_ids[conv]).mean()))
        fr.append(float(np.mean(flips)))
    return fr


def boxcount_db(b):
    xs = np.linspace(-BOX, BOX, GRID)
    X, Y = np.meshgrid(xs, xs)
    r0 = np.stack([X.ravel(), Y.ravel()], axis=1)
    ids = teacher_ids(r0, b).reshape(GRID, GRID)
    bnd = np.zeros((GRID, GRID), dtype=bool)
    bnd[:, :-1] |= ids[:, :-1] != ids[:, 1:]
    bnd[:-1, :] |= ids[:-1, :] != ids[1:, :]
    counts = []
    for c in COARSE:
        g = GRID // c
        counts.append(int(bnd[: g * c, : g * c].reshape(g, c, g, c).any(axis=(1, 3)).sum()))
    slope, _ = np.polyfit(np.log([1.0 / c for c in COARSE]), np.log(counts), 1)
    return float(slope)


def make_table(b, n_pairs, rng):
    """Half trajectory states (the dynamically relevant region), half broad prior."""
    n_traj = n_pairs // 2
    ics = np.stack([rng.uniform(-BOX, BOX, 512), rng.uniform(-BOX, BOX, 512)], axis=1)
    r, v = ics.copy(), np.zeros_like(ics)
    pool_r, pool_v = [r.copy()], [v.copy()]
    for _ in range(200):                       # 200 snapshots x 20 steps = t 80
        r, v = rk4(r, v, b, T_MAP)
        pool_r.append(r.copy())
        pool_v.append(v.copy())
    pr = np.concatenate(pool_r)
    pv = np.concatenate(pool_v)
    pick = rng.integers(0, len(pr), n_traj)
    s_traj = np.concatenate([pr[pick], pv[pick]], axis=1)
    n_uni = n_pairs - n_traj
    s_uni = np.concatenate([rng.uniform(-1.8, 1.8, (n_uni, 2)),
                            rng.uniform(-2.5, 2.5, (n_uni, 2))], axis=1)
    S = np.concatenate([s_traj, s_uni])
    r2, v2 = rk4(S[:, :2].copy(), S[:, 2:].copy(), b, T_MAP)
    V = np.concatenate([r2, v2], axis=1) - S
    keep = np.linalg.norm(V, axis=1) < 10.0
    return S[keep].astype(np.float32), V[keep].astype(np.float32)


def train_student(Z, V, seed, dims=(4, 256, 256, 4)):
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


def student_ids(model, probes):
    s = np.concatenate([probes, np.zeros_like(probes)], axis=1).astype(np.float32)
    p = mx.array(s)
    lo = mx.array([-3.0, -3.0, -5.0, -5.0])
    hi = mx.array([3.0, 3.0, 5.0, 5.0])
    for i in range(STUDENT_ITERS):
        p = mx.maximum(mx.minimum(p + model(p), hi), lo)
        if i % 50 == 0:
            mx.eval(p)
    mx.eval(p)
    q = np.array(p, dtype=np.float64)
    r, v = q[:, :2], q[:, 2:]
    d = np.linalg.norm(r[:, None, :] - MAGNETS[None, :, :], axis=-1)
    ids = d.argmin(axis=1)
    bad = (np.linalg.norm(v, axis=1) > 0.2) | (d.min(axis=1) > 1.0)
    ids[bad] = -1
    return ids


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def main():
    t0 = time.time()
    rng = np.random.default_rng(0)
    probes = np.stack([rng.uniform(-BOX, BOX, N_PROBES),
                       rng.uniform(-BOX, BOX, N_PROBES)], axis=1)
    out = {"systems": {}, "gates": {}}

    for b in DAMPINGS:
        tb = time.time()
        base_ids = teacher_ids(probes, b)
        conv = float((base_ids >= 0).mean())
        fr = uncertainty_curve(probes, base_ids, b, np.random.default_rng(17))
        alpha, r2, (sl, ic) = fit_alpha([f * SCALE for f in EPS_FRACS], fr)
        ceiling = 1.0 - fr[EPS_FRACS.index(EPS_PROTOCOL)]
        db = boxcount_db(b)
        print(f"[b={b}] conv {conv:.3f}  alpha {alpha:.3f} (R2 {r2:.3f})  "
              f"D_b {db:.3f}  2-D_b {2 - db:.3f}  ceiling(1-f) {ceiling:.3f}  "
              f"[{time.time() - tb:.0f}s]")

        sysrec = {"conv": conv, "alpha": alpha, "alpha_r2": r2, "D_b": db,
                  "alpha_geom": 2 - db, "ceiling": ceiling,
                  "f_curve": dict(zip(map(str, EPS_FRACS), fr)), "budgets": {}}

        for N in BUDGETS:
            Z, V = make_table(b, N, np.random.default_rng(int(b * 1000)))
            model = train_student(Z, V, seed=0)
            sid = student_ids(model, probes)
            mask = base_ids >= 0
            basin = float((sid[mask] == base_ids[mask]).mean())
            sysrec["budgets"][str(N)] = {"basin": basin}
            print(f"    N={N:>7}: basin {basin:.3f}")
        out["systems"][str(b)] = sysrec

    S = out["systems"]
    alphas = [S[str(b)]["alpha"] for b in DAMPINGS]
    basins_top = [S[str(b)]["budgets"][str(BUDGETS[-1])]["basin"] for b in DAMPINGS]
    g5 = all(alphas[i] >= alphas[i + 1] - 0.03 for i in range(len(alphas) - 1))
    g6 = all(abs(S[str(b)]["alpha"] - S[str(b)]["alpha_geom"]) <= 0.20 for b in DAMPINGS)
    rho = spearman(alphas, basins_top)
    g7 = rho == 1.0
    g8 = all(S[str(b)]["budgets"][str(BUDGETS[-1])]["basin"]
             >= S[str(b)]["budgets"][str(BUDGETS[0])]["basin"] - 0.02 for b in DAMPINGS)
    out["gates"] = {"G5_monotone": bool(g5), "G6_crosscheck": bool(g6),
                    "G7_test_spearman": rho, "G7_pass": bool(g7), "G8_budget": bool(g8)}
    print(f"\nGATES: G5 {g5}  G6 {g6}  G7 rho={rho:+.2f} pass={g7}  G8 {g8}")
    RUNS.mkdir(exist_ok=True)
    (RUNS / "pendulum.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {RUNS / 'pendulum.json'} [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
