"""Experiment 8: fact-check suite for the alpha-pivot claims (the ratchet et al.).

The alpha reframing (post-Experiment 7) made five claims that are testable in
the existing harnesses but were never run. Pre-registered checks:

  P1 alpha-ratchet ....... A finite-capacity copy cannot represent a fractal
                           boundary, so its own boundary must be smoother:
                           alpha_student > alpha_teacher whenever
                           alpha_teacher is well below smooth. Gate: holds
                           (margin 0.05) in >= 5 of the 7 teacher->student
                           pairs {newton n=3,4,5; pendulum b=0.35,0.2; mnist
                           field arm; gi union gen-1}; and down the gi lineage
                           alpha does not decrease (gen5 >= gen1 - 0.05).
  P2 definite/unfaithful . Where P1 holds, the copy is MORE self-consistent
                           than its teacher while being less faithful:
                           self-ceiling_student > ceiling_teacher AND
                           basin-vs-teacher < self-ceiling_student.
  P3 graft = alpha collapse. The fitted (OOD-chart) union frame collapses
                           boundary geometry on the grafted side:
                           alpha_B(fit oracle) < alpha_B(iso oracle) - 0.10.
  P4 two-factor law ...... EXPLORATORY: does basin ~= 1 - f_teacher(delta_eff)
                           (validated in Newton) also hold on the pendulum?
                           Report |pred - actual|; nominal tolerance 0.15.
                           Known unit caveat: f(eps) is measured under 2D
                           position perturbations, delta_eff is a 4D per-map
                           error, so failure is informative, not fatal.
  P5 reliability vs alpha. Seed spread of parametric students grows as alpha
                           falls: Spearman(alpha, max-min basin over 4 seeds
                           at 16k pairs) = -1 across b in {0.6, 0.35, 0.2}.

Requires runs/newton.json + runs/pendulum.json (Experiment 7) and the cached
MNIST teacher. Run:  uv run python -m m5_fielddistill.ratchet all
Stages: newton | pendulum | gi | mnist | report    -> runs/ratchet.json
"""

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from . import newton as NW
from . import pendulum as PD
from .ae import field as ae_field, iterate as ae_iterate

RUNS = Path(__file__).resolve().parents[2] / "runs"
OUT = RUNS / "ratchet.json"

fit_alpha = NW.fit_alpha
spearman = NW.spearman


def _load():
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def _save(d):
    OUT.write_text(json.dumps(d, indent=1))


def _predict_from_curve(f_curve, scale, delta):
    eps = [float(k) * scale for k in f_curve]
    fr = list(f_curve.values())
    sl, _, (s, ic) = fit_alpha(eps, fr)
    if not np.isfinite(s):
        return float("nan")
    return float(1.0 - min(1.0, np.exp(ic) * delta ** s))


# ---------- stage: newton ----------

def stage_newton():
    t0 = time.time()
    nj = json.loads((RUNS / "newton.json").read_text())
    rng = np.random.default_rng(0)
    probes = rng.uniform(-NW.BOX, NW.BOX, NW.N_PROBES) \
        + 1j * rng.uniform(-NW.BOX, NW.BOX, NW.N_PROBES)
    rec = {}
    for n in (3, 4, 5):
        t = nj["systems"][str(n)]
        Z, V = NW.make_table(n, 65536, np.random.default_rng(100 + n))
        model = NW.train_student(Z, V, seed=0)
        base_t = NW.root_ids(NW.teacher_endpoints(probes, n), n, NW.ROOT_TOL_TEACHER)
        sid = NW.root_ids(NW.student_endpoints(model, probes), n, NW.ROOT_TOL_STUDENT)
        mask = base_t >= 0
        basin = float((sid[mask] == base_t[mask]).mean())
        thr = np.sin(np.pi / n)  # half the min root separation
        base_s = NW.student_endpoints(model, probes)
        rng2 = np.random.default_rng(50 + n)
        fr = []
        for frac in NW.EPS_FRACS:
            eps = frac * NW.SCALE
            flips = []
            for _ in range(3):
                pert = probes + eps * (rng2.standard_normal(len(probes))
                                       + 1j * rng2.standard_normal(len(probes)))
                ep2 = NW.student_endpoints(model, pert)
                flips.append(float((np.abs(ep2 - base_s) > thr).mean()))
            fr.append(float(np.mean(flips)))
        a_s, r2s, _ = fit_alpha([f * NW.SCALE for f in NW.EPS_FRACS], fr)
        ceil_s = 1.0 - fr[NW.EPS_FRACS.index(NW.EPS_PROTOCOL)]
        rec[str(n)] = dict(alpha_t=t["alpha"], alpha_s=a_s, alpha_s_r2=r2s,
                           ceiling_t=t["ceiling"], ceiling_s=ceil_s, basin=basin)
        print(f"[newton n={n}] alpha_t {t['alpha']:.3f} -> alpha_s {a_s:.3f} "
              f"(R2 {r2s:.2f})  ceiling_t {t['ceiling']:.3f} -> ceiling_s {ceil_s:.3f}  "
              f"basin {basin:.3f}")
    d = _load()
    d["newton"] = rec
    _save(d)
    print(f"newton stage done [{time.time() - t0:.0f}s]")


# ---------- stage: pendulum ----------

def _pend_endpoints(model, probes):
    s = np.concatenate([probes, np.zeros_like(probes)], axis=1).astype(np.float32)
    p = mx.array(s)
    lo = mx.array([-3.0, -3.0, -5.0, -5.0])
    hi = mx.array([3.0, 3.0, 5.0, 5.0])
    for i in range(PD.STUDENT_ITERS):
        p = mx.maximum(mx.minimum(p + model(p), hi), lo)
        if i % 50 == 0:
            mx.eval(p)
    mx.eval(p)
    return np.array(p, dtype=np.float64)[:, :2]


def stage_pendulum():
    t0 = time.time()
    pj = json.loads((RUNS / "pendulum.json").read_text())
    rng = np.random.default_rng(0)
    probes = np.stack([rng.uniform(-PD.BOX, PD.BOX, PD.N_PROBES),
                       rng.uniform(-PD.BOX, PD.BOX, PD.N_PROBES)], axis=1)
    S0 = np.concatenate([probes, np.zeros_like(probes)], axis=1)
    rec = {}
    for b in (0.6, 0.35, 0.2):
        t = pj["systems"][str(b)]
        base_t = PD.teacher_ids(probes, b)
        mask = base_t >= 0
        Z, V = PD.make_table(b, 16384, np.random.default_rng(int(b * 1000)))
        r2_, v2_ = PD.rk4(S0[:, :2].copy(), S0[:, 2:].copy(), b, PD.T_MAP)
        Vt = np.concatenate([r2_, v2_], axis=1) - S0
        seeds_rec, best = [], None
        for seed in (1, 2, 3):
            model = PD.train_student(Z, V, seed=seed)
            sid = PD.student_ids(model, probes)
            basin = float((sid[mask] == base_t[mask]).mean())
            Vs = np.array(model(mx.array(S0.astype(np.float32))), dtype=np.float64)
            derr = float(np.median(np.linalg.norm(Vs - Vt, axis=1)))
            pred = _predict_from_curve(t["f_curve"], PD.SCALE, derr)
            seeds_rec.append(dict(seed=seed, basin=basin, delta_eff=derr, basin_pred=pred))
            print(f"[pend b={b}] seed {seed}: basin {basin:.3f}  delta_eff {derr:.4f}  "
                  f"pred {pred:.3f}")
            if best is None or basin > best[1]:
                best = (model, basin)
        model = best[0]
        base_s = _pend_endpoints(model, probes)
        rng2 = np.random.default_rng(70 + int(b * 100))
        fr = []
        for frac in PD.EPS_FRACS:
            eps = frac * PD.SCALE
            flips = []
            for _ in range(2):
                ep2 = _pend_endpoints(model, probes + eps * rng2.standard_normal(probes.shape))
                flips.append(float((np.linalg.norm(ep2 - base_s, axis=1) > 0.87).mean()))
            fr.append(float(np.mean(flips)))
        a_s, r2s, _ = fit_alpha([f * PD.SCALE for f in PD.EPS_FRACS], fr)
        ceil_s = 1.0 - fr[PD.EPS_FRACS.index(PD.EPS_PROTOCOL)]
        basin16_seed0 = pj["systems"][str(b)]["budgets"]["16384"]["basin"]
        spread_basins = [basin16_seed0] + [s["basin"] for s in seeds_rec]
        rec[str(b)] = dict(alpha_t=t["alpha"], alpha_s=a_s, alpha_s_r2=r2s,
                           ceiling_t=t["ceiling"], ceiling_s=ceil_s,
                           basin_best=best[1], seeds=seeds_rec,
                           spread=float(max(spread_basins) - min(spread_basins)),
                           spread_basins=spread_basins)
        print(f"[pend b={b}] alpha_t {t['alpha']:.3f} -> alpha_s {a_s:.3f} (R2 {r2s:.2f})  "
              f"ceiling_t {t['ceiling']:.3f} -> ceiling_s {ceil_s:.3f}  "
              f"spread {rec[str(b)]['spread']:.3f}")
    d = _load()
    d["pendulum"] = rec
    _save(d)
    print(f"pendulum stage done [{time.time() - t0:.0f}s]")


# ---------- stage: gi (lineage ratchet + graft alpha) ----------

def stage_gi():
    t0 = time.time()
    from .gi import (BASIN_EPS, ORACLE_STEPS, UnionFrame, distill, eval_student,
                     make_world8, np_field)
    from .train import train_teacher

    EPS_LIST = [f * BASIN_EPS for f in (1 / 32, 1 / 16, 1 / 8, 0.1667, 1 / 4, 1 / 2, 1.0)]
    PROT_IDX = 3  # 0.1667 * 0.3 = 0.05 absolute, the gi oracle's probe noise

    xh, labels = make_world8()
    xA, xB = xh[labels < 4], xh[labels >= 4]
    print("training teachers A, B ...")
    A = train_teacher(xA, noise=0.1, seed=0)
    B = train_teacher(xB, noise=0.1, seed=0)
    frame = UnionFrame(A, B, xh, labels, mode="iso")
    frame_fit = UnionFrame(A, B, xh, labels, mode="fit")
    orc = frame.oracle(np.random.default_rng(7))

    def oracle_alpha(fr_obj, seed):
        r = np.random.default_rng(seed)
        probes = (fr_obj.lo + r.random((512, 2)).astype(np.float32)
                  * (fr_obj.hi - fr_obj.lo))
        base = fr_obj.iterate_np(probes)
        side = fr_obj.gate(probes)
        frj, frB = [], []
        for eps in EPS_LIST:
            fj, fB = [], []
            for _ in range(2):
                pert = (probes + eps * r.standard_normal(probes.shape)).astype(np.float32)
                fl = np.linalg.norm(fr_obj.iterate_np(pert) - base, axis=1) > BASIN_EPS
                fj.append(float(fl.mean()))
                fB.append(float(fl[side].mean()))
            frj.append(float(np.mean(fj)))
            frB.append(float(np.mean(fB)))
        aj, r2j, _ = fit_alpha(EPS_LIST, frj)
        aB, r2B, _ = fit_alpha(EPS_LIST, frB)
        return dict(alpha_joint=aj, r2_joint=r2j, alpha_B=aB, r2_B=r2B,
                    ceiling_joint=1 - frj[PROT_IDX], ceiling_B=1 - frB[PROT_IDX])

    o_iso = oracle_alpha(frame, 11)
    o_fit = oracle_alpha(frame_fit, 12)
    print(f"[gi] iso oracle: alpha_joint {o_iso['alpha_joint']:.3f} "
          f"alpha_B {o_iso['alpha_B']:.3f} | fit oracle: alpha_joint "
          f"{o_fit['alpha_joint']:.3f} alpha_B {o_fit['alpha_B']:.3f}")

    students = [distill(frame.field_np, frame.lo, frame.hi, 1)]
    for g in range(2, 6):
        prev = students[-1]
        students.append(distill(lambda z, c=prev: np_field(c, z),
                                frame.lo, frame.hi, 40 + g))

    gens = []
    for gidx, s in enumerate(students, 1):
        ev = eval_student(s, frame, orc)
        r = np.random.default_rng(60 + gidx)
        probes = (frame.lo + r.random((512, 2)).astype(np.float32)
                  * (frame.hi - frame.lo))

        def ep_fn(p):
            return np.array(ae_iterate(s, mx.array(p), ORACLE_STEPS,
                                       frame.clip_lo, frame.clip_hi))

        base = ep_fn(probes)
        fr = []
        for eps in EPS_LIST:
            fl = []
            for _ in range(2):
                pert = (probes + eps * r.standard_normal(probes.shape)).astype(np.float32)
                fl.append(float((np.linalg.norm(ep_fn(pert) - base, axis=1)
                                 > BASIN_EPS).mean()))
            fr.append(float(np.mean(fl)))
        a_s, r2s, _ = fit_alpha(EPS_LIST, fr)
        gens.append(dict(gen=gidx, alpha=a_s, r2=r2s, ceiling=1 - fr[PROT_IDX],
                         basin_joint=ev["basin_joint"]))
        print(f"[gi] gen {gidx}: alpha {a_s:.3f} (R2 {r2s:.2f})  "
              f"self-ceiling {1 - fr[PROT_IDX]:.3f}  basin vs oracle "
              f"{ev['basin_joint']:.3f}")

    d = _load()
    d["gi"] = dict(oracle_iso=o_iso, oracle_fit=o_fit,
                   oracle_ceil_joint=orc["ceil_joint"], lineage=gens)
    _save(d)
    print(f"gi stage done [{time.time() - t0:.0f}s]")


# ---------- stage: mnist ----------

def stage_mnist():
    t0 = time.time()
    import mlx.nn as nn
    import mlx.optimizers as optim
    from .mnist import (BS, ITER_STEPS, LR, N_TRAIN, D_LAT,
                        STUDENT_STEPS as MSTEPS, load_mnist, load_teacher, make_ae)

    xtr, _, _, _ = load_mnist()
    teacher, meta = load_teacher()
    mu, sd = meta["mu"], meta["sd"]
    clip_lo, clip_hi = meta["clip_lo"], meta["clip_hi"]
    basin_eps = float(meta["basin_eps"])
    probes = meta["probes"]
    ep_t = meta["ep_probes"]

    student = make_ae(101)
    opt = optim.Adam(learning_rate=optim.cosine_decay(LR, MSTEPS, 1e-4))
    rng = np.random.default_rng(1001)

    def sample_z(k):
        half = k // 2
        broad = (mu + 2.0 * sd * rng.standard_normal((half, D_LAT))).astype(np.float32)
        core = mx.array((mu + 1.5 * sd * rng.standard_normal((k - half, D_LAT)))
                        .astype(np.float32))
        depth = int(rng.integers(2, 25))
        for _ in range(depth):
            core = teacher.enc(teacher.dec(core))
        return mx.concatenate([mx.array(broad), core], axis=0)

    def loss_fn(m, zb, vt):
        return mx.mean((ae_field(m, zb) - vt) ** 2)

    lg = nn.value_and_grad(student, loss_fn)
    print(f"training MNIST field student ({MSTEPS} steps) ...")
    for _ in range(MSTEPS):
        zb = sample_z(BS)
        vt = ae_field(teacher, zb)
        mx.eval(zb, vt)
        _, g = lg(student, zb, vt)
        g, _ = optim.clip_grad_norm(g, max_norm=5.0)
        opt.update(student, g)
        mx.eval(student.parameters(), opt.state)

    ep_s = np.array(ae_iterate(student, mx.array(probes), ITER_STEPS, clip_lo, clip_hi))
    basin = float((np.linalg.norm(ep_s - ep_t, axis=1) < basin_eps).mean())

    EPS_FRACS = (1 / 64, 1 / 32, 1 / 16, 1 / 8, 0.1255, 1 / 4, 1 / 2)  # x basin_eps
    rng2 = np.random.default_rng(3)
    fr = []
    for frac in EPS_FRACS:
        eps = frac * basin_eps
        flips = []
        for _ in range(2):
            pert = (probes + eps * rng2.standard_normal(probes.shape)).astype(np.float32)
            ep2 = np.array(ae_iterate(student, mx.array(pert), ITER_STEPS, clip_lo, clip_hi))
            flips.append(float((np.linalg.norm(ep2 - ep_s, axis=1) > basin_eps).mean()))
        fr.append(float(np.mean(flips)))
    a_s, r2s, _ = fit_alpha([f * basin_eps for f in EPS_FRACS], fr)
    ceil_s = 1.0 - fr[EPS_FRACS.index(0.1255)]

    rec = dict(alpha_t=0.454, ceiling_t=float(meta["self_agree"]),
               alpha_s=a_s, alpha_s_r2=r2s, ceiling_s=ceil_s, basin=basin)
    print(f"[mnist] alpha_t 0.454 -> alpha_s {a_s:.3f} (R2 {r2s:.2f})  "
          f"ceiling_t {float(meta['self_agree']):.3f} -> ceiling_s {ceil_s:.3f}  "
          f"basin {basin:.3f}")
    d = _load()
    d["mnist"] = rec
    _save(d)
    print(f"mnist stage done [{time.time() - t0:.0f}s]")


# ---------- stage: report ----------

def stage_report():
    d = _load()
    pairs = []
    for n in ("3", "4", "5"):
        r = d["newton"][n]
        pairs.append((f"newton n={n}", r["alpha_t"], r["alpha_s"],
                      r["ceiling_t"], r["ceiling_s"], r["basin"]))
    for b in ("0.35", "0.2"):
        r = d["pendulum"][b]
        pairs.append((f"pendulum b={b}", r["alpha_t"], r["alpha_s"],
                      r["ceiling_t"], r["ceiling_s"], r["basin_best"]))
    r = d["mnist"]
    pairs.append(("mnist field", r["alpha_t"], r["alpha_s"],
                  r["ceiling_t"], r["ceiling_s"], r["basin"]))
    g = d["gi"]
    gen1 = g["lineage"][0]
    pairs.append(("gi union gen1", g["oracle_iso"]["alpha_joint"], gen1["alpha"],
                  g["oracle_iso"]["ceiling_joint"], gen1["ceiling"],
                  gen1["basin_joint"]))

    print(f"{'pair':<18}{'alpha_t':>9}{'alpha_s':>9}{'ceil_t':>8}{'ceil_s':>8}{'basin':>8}")
    p1_hits, p2_hits, p2_total = 0, 0, 0
    for name, at, as_, ct, cs, ba in pairs:
        hit = np.isfinite(as_) and as_ > at + 0.05
        p1_hits += hit
        if hit:
            p2_total += 1
            p2_hits += (cs > ct) and (ba < cs)
        print(f"{name:<18}{at:>9.3f}{as_:>9.3f}{ct:>8.3f}{cs:>8.3f}{ba:>8.3f}"
              f"  {'RATCHET' if hit else '-'}")

    lin = g["lineage"]
    p1_lineage = lin[-1]["alpha"] >= lin[0]["alpha"] - 0.05
    p1 = (p1_hits >= 5) and p1_lineage
    p2 = p2_total > 0 and p2_hits == p2_total
    p3 = g["oracle_fit"]["alpha_B"] < g["oracle_iso"]["alpha_B"] - 0.10
    gaps = [abs(s["basin_pred"] - s["basin"]) for b in ("0.6", "0.35", "0.2")
            for s in d["pendulum"][b]["seeds"] if np.isfinite(s["basin_pred"])]
    p4_median = float(np.median(gaps)) if gaps else float("nan")
    p4 = p4_median <= 0.15
    alphas = [d["pendulum"][b]["alpha_t"] for b in ("0.6", "0.35", "0.2")]
    spreads = [d["pendulum"][b]["spread"] for b in ("0.6", "0.35", "0.2")]
    rho = spearman(alphas, spreads)
    p5 = rho == -1.0

    print(f"\nlineage alpha gen1->5: " + " -> ".join(f"{x['alpha']:.3f}" for x in lin))
    print(f"P1 ratchet: {p1}  ({p1_hits}/7 pairs, lineage non-decreasing {p1_lineage})")
    print(f"P2 definite/unfaithful: {p2}  ({p2_hits}/{p2_total} of ratchet pairs)")
    print(f"P3 graft alpha collapse: {p3}  (iso B {g['oracle_iso']['alpha_B']:.3f} "
          f"vs fit B {g['oracle_fit']['alpha_B']:.3f})")
    print(f"P4 two-factor on pendulum (EXPLORATORY): {p4}  (median gap {p4_median:.3f})")
    print(f"P5 spread vs alpha: {p5}  (rho {rho:+.2f}; spreads {spreads})")
    d["gates"] = dict(P1=bool(p1), P1_pairs=int(p1_hits), P1_lineage=bool(p1_lineage),
                      P2=bool(p2), P3=bool(p3), P4=bool(p4),
                      P4_median_gap=p4_median, P5=bool(p5), P5_rho=rho)
    _save(d)


STAGES = {"newton": stage_newton, "pendulum": stage_pendulum, "gi": stage_gi,
          "mnist": stage_mnist, "report": stage_report}


def main():
    args = sys.argv[1:] or ["all"]
    if args[0] == "all":
        for name in ("newton", "pendulum", "gi", "mnist", "report"):
            STAGES[name]()
    else:
        STAGES[args[0]]()


if __name__ == "__main__":
    main()
