"""GI golden-start harness: do latent vector fields UNION, and does repeated
transmission ONION (filter memorization, preserve the general core)?

World: 8 Gaussian blobs on a ring, lifted to 16D. Teacher A knows blobs 0-3,
teacher B knows blobs 4-7 (denoising AEs, contractive regime). B_mem is a
memorization-regime teacher (no denoising) on blobs 4-7 for the contaminated arm.

Common frame = teacher A's latent chart. B's field is grafted in via an affine
conjugacy T fitted on SHARED PROBE INPUTS (both encoders applied to B-region
inputs; no shared training). The union field is a hard nearest-centroid gate
between V_A and the conjugated V_B; the union ORACLE iterates that gated
composite map.

Pre-registered gates (GOLDEN GO requires all five):
  C1 union       : union student joint basin >= 0.60 AND per-side drop vs
                   single-teacher students <= 0.15.
  K3 kill-switch : merged output-distillation within 0.05 joint basin of the
                   union-field student => field composition redundant (NO-GO).
  C3 exceedance  : union student recovers >= 7 of 8 attractor sites (no single
                   teacher holds more than 4).
  C4 lineage     : clean lineage gen-5 joint basin >= 0.85 x gen-1 (plateau,
                   not photocopy decay).
  C2 filter      : clean lineage's ORDERED graft persists (gen5 B-side >= 0.8 x
                   gen1) while the contaminated lineage's DISORDERED graft dies
                   (B_mem-side <= 0.30 by gen 3) with its A-side retained
                   (gen5 >= 0.8 x gen1).

Run:  uv run python -m m5_fielddistill.gi     (~5 min on M5)
Writes runs/gi.json + runs/gi.png
"""

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np

from .ae import AE, chamfer, field, find_attractors, fit_affine, iterate
from .train import train_teacher

RUNS = Path(__file__).resolve().parents[2] / "runs"
BASIN_EPS = 0.3
ORACLE_STEPS = 400
STUDENT_STEPS = 8000
GENS = 5


# ---------- world ----------

def make_world8(n_per=500, radius=2.2, std=0.32, d_high=16):
    rng = np.random.default_rng(0)
    angles = np.arange(8) / 8 * 2 * np.pi
    centers = radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    labels = np.repeat(np.arange(8), n_per)
    x2 = centers[labels] + std * rng.normal(size=(len(labels), 2))
    W1 = rng.normal(size=(2, d_high)) / np.sqrt(2)
    b1 = 0.1 * rng.normal(size=(d_high,))
    W2 = rng.normal(size=(d_high, d_high)) / np.sqrt(d_high)
    xh = (np.tanh(x2 @ W1 + b1) @ W2).astype(np.float32)
    return xh, labels


def np_field(model, z):
    return np.array(field(model, mx.array(z.astype(np.float32))))


def np_step(model, z):
    zm = mx.array(z.astype(np.float32))
    return np.array(model.enc(model.dec(zm)))


# ---------- union frame ----------

class UnionFrame:
    """Common frame holding A natively and B via affine conjugacy T: z_B -> frame.

    mode="fit": T fitted so grafted B lands where A's encoder puts B's inputs
                (input-consistent, but relies on A's OOD encoder behavior, which
                COMPRESSES foreign territory -> the hard mode).
    mode="iso": T is a pure translation placing B's chart, undistorted, into
                empty frame territory (the designed frame -> pure union test).
    """

    def __init__(self, A, B, x_all, labels, mode="fit"):
        self.A, self.B = A, B
        self.mode = mode
        xA, xB = x_all[labels < 4], x_all[labels >= 4]
        zA_own = np.array(A.enc(mx.array(xA)))
        zB_own = np.array(B.enc(mx.array(xB)))
        if mode == "fit":
            zA_B = np.array(A.enc(mx.array(xB)))
            self.T = fit_affine(zB_own, zA_B)      # B-chart -> A-chart (OOD-fitted)
            assert self.T is not None, "conjugacy T is singular"
            self.align_resid = float(np.sqrt(np.mean(
                np.sum((np.concatenate([zB_own, np.ones((len(zB_own), 1))], 1)
                        @ np.vstack([self.T[0], self.T[1]]) - zA_B) ** 2, axis=1))))
        else:                                      # isometric graft
            gap = 1.5
            d = np.array([zA_own[:, 0].max() - zB_own[:, 0].min() + gap,
                          zA_own[:, 1].mean() - zB_own[:, 1].mean()], dtype=np.float32)
            eye = np.eye(2, dtype=np.float32)
            self.T = (eye, d, eye)
            self.align_resid = 0.0
        W, b, _ = self.T
        graft = zB_own @ W + b
        labA, labB = labels[labels < 4], labels[labels >= 4]
        cents = [zA_own[labA == c].mean(0) for c in range(4)]
        cents += [graft[labB == c].mean(0) for c in range(4, 8)]
        self.centroids = np.stack(cents)
        self.collision = float(np.min(np.linalg.norm(
            self.centroids[4:, None] - self.centroids[None, :4], axis=-1)))
        allz = np.concatenate([zA_own, graft])
        lo, hi = allz.min(0), allz.max(0)
        span = hi - lo
        self.lo, self.hi = (lo - 0.35 * span).astype(np.float32), (hi + 0.35 * span).astype(np.float32)
        self.clip_lo, self.clip_hi = self.lo - 2 * span, self.hi + 2 * span

    def gate(self, z):
        """True where the grafted (B) side owns the point."""
        d = np.linalg.norm(z[:, None, :] - self.centroids[None], axis=-1)
        return d.argmin(axis=1) >= 4

    def field_np(self, z):
        vA = np_field(self.A, z)
        W, b, Winv = self.T
        zB = (z - b) @ Winv
        fB = np_step(self.B, zB)
        vB = (fB @ W + b) - z
        return np.where(self.gate(z)[:, None], vB, vA)

    def iterate_np(self, z, steps=ORACLE_STEPS):
        z = z.copy()
        for _ in range(steps):
            z = np.clip(z + self.field_np(z), self.clip_lo, self.clip_hi)
        return z

    def oracle(self, rng, n_probes=400, grid_n=30):
        probes = (self.lo + rng.random((n_probes, 2)).astype(np.float32) * (self.hi - self.lo))
        ep = self.iterate_np(probes)
        ep_pert = self.iterate_np(
            probes + 0.05 * rng.standard_normal(probes.shape).astype(np.float32))
        d = np.linalg.norm(ep - ep_pert, axis=1)
        side = self.gate(probes)
        ceil_joint = float((d < BASIN_EPS).mean())
        ceil_A = float((d[~side] < BASIN_EPS).mean())
        ceil_B = float((d[side] < BASIN_EPS).mean())
        xs = np.linspace(self.lo[0], self.hi[0], grid_n, dtype=np.float32)
        ys = np.linspace(self.lo[1], self.hi[1], grid_n, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)
        G = np.stack([gx.ravel(), gy.ravel()], axis=1)
        epg = self.iterate_np(G)
        vn = np.linalg.norm(self.field_np(epg), axis=1)
        attractors = find_attractors(epg, vn, vtol=5e-3, merge_r=0.15)
        return dict(probes=probes, ep=ep, side=side, ceil_joint=ceil_joint,
                    ceil_A=ceil_A, ceil_B=ceil_B, attractors=attractors, G=G)


# ---------- students ----------

def distill(field_fn, lo, hi, seed, steps=STUDENT_STEPS, bs=256, lr=1e-3, d_in=16):
    import mlx.nn as nn
    import mlx.optimizers as optim
    mx.random.seed(seed)
    model = AE(d_in=d_in)
    mx.eval(model.parameters())
    opt = optim.Adam(learning_rate=lr)
    rng = np.random.default_rng(5000 + seed)

    def loss_fn(m, zb, vt):
        return mx.mean((field(m, zb) - vt) ** 2)

    lg = nn.value_and_grad(model, loss_fn)
    for _ in range(steps):
        z = (lo + rng.random((bs, 2)).astype(np.float32) * (hi - lo))
        vt = field_fn(z)
        _, g = lg(model, mx.array(z), mx.array(vt.astype(np.float32)))
        opt.update(model, g)
        mx.eval(model.parameters(), opt.state)
    return model


def eval_student(student, frame, orc, align=None):
    """Basin agreement vs the union oracle, per side; attractor recovery."""
    if align is None:
        W = np.eye(2, dtype=np.float32); b = np.zeros(2, dtype=np.float32); Winv = W
    else:
        W, b, Winv = align
    probes_s = ((orc["probes"] - b) @ Winv).astype(np.float32)
    ep_s = np.array(iterate(student, mx.array(probes_s), ORACLE_STEPS,
                            frame.clip_lo, frame.clip_hi)) @ W + b
    d = np.linalg.norm(ep_s - orc["ep"], axis=1)
    side = orc["side"]
    basin_joint = float((d < BASIN_EPS).mean())
    basin_A = float((d[~side] < BASIN_EPS).mean())
    basin_B = float((d[side] < BASIN_EPS).mean())
    G_s = ((orc["G"] - b) @ Winv).astype(np.float32)
    epg = np.array(iterate(student, mx.array(G_s), ORACLE_STEPS,
                           frame.clip_lo, frame.clip_hi))
    vn = np.linalg.norm(np_field(student, epg), axis=1)
    attr = find_attractors(epg, vn, vtol=5e-3, merge_r=0.15)
    attr = attr @ W + b if len(attr) else attr
    tgt = orc["attractors"]
    recovered = int(sum(np.min(np.linalg.norm(tgt[i] - attr, axis=1)) < BASIN_EPS
                        for i in range(len(tgt)))) if len(attr) else 0
    return dict(basin_joint=basin_joint, basin_A=basin_A, basin_B=basin_B,
                chamfer=float(chamfer(tgt, attr)), n_attractors=int(len(attr)),
                sites_recovered=recovered, sites_total=int(len(tgt)))


# ---------- main ----------

def main():
    t0 = time.time()
    RUNS.mkdir(exist_ok=True)
    rng = np.random.default_rng(7)
    xh, labels = make_world8()
    xA, xB = xh[labels < 4], xh[labels >= 4]

    print("training teachers A (blobs 0-3), B (blobs 4-7), B_mem (no denoise) ...")
    A = train_teacher(xA, noise=0.1, seed=0)
    B = train_teacher(xB, noise=0.1, seed=0)
    B_mem = train_teacher(xB, noise=0.0, seed=0)

    # hard mode first: input-fitted graft (A's OOD encoder places B's knowledge)
    frame_fit = UnionFrame(A, B, xh, labels, mode="fit")
    orc_fit = frame_fit.oracle(rng)
    s_fit = distill(frame_fit.field_np, frame_fit.lo, frame_fit.hi, 1)
    m_fit = eval_student(s_fit, frame_fit, orc_fit)
    print(f"  FIT graft (hard mode): resid {frame_fit.align_resid:.3f}, collision "
          f"{frame_fit.collision:.2f} -> union student joint {m_fit['basin_joint']:.2f} "
          f"A {m_fit['basin_A']:.2f} B {m_fit['basin_B']:.2f}")

    # pure union: isometric graft (designed frame, undistorted B chart)
    frame = UnionFrame(A, B, xh, labels, mode="iso")
    orc = frame.oracle(rng)
    print(f"  ISO frame: collision {frame.collision:.2f}, {len(orc['attractors'])} union "
          f"attractors, ceilings joint {orc['ceil_joint']:.2f} A {orc['ceil_A']:.2f} "
          f"B {orc['ceil_B']:.2f}")

    frame_c = UnionFrame(A, B_mem, xh, labels, mode="iso")
    orc_c = frame_c.oracle(rng)
    print(f"  contaminated frame: align resid {frame_c.align_resid:.3f}, "
          f"{len(orc_c['attractors'])} attractors, ceilings joint {orc_c['ceil_joint']:.2f} "
          f"A {orc_c['ceil_A']:.2f} Bmem {orc_c['ceil_B']:.2f}")

    out = dict(fit_mode=dict(align_resid=frame_fit.align_resid,
                             collision=frame_fit.collision, student=m_fit),
               frame=dict(align_resid=frame.align_resid, collision=frame.collision,
                          n_attractors=int(len(orc["attractors"])),
                          ceil_joint=orc["ceil_joint"], ceil_A=orc["ceil_A"], ceil_B=orc["ceil_B"]),
               frame_c=dict(align_resid=frame_c.align_resid,
                            n_attractors=int(len(orc_c["attractors"])),
                            ceil_joint=orc_c["ceil_joint"], ceil_A=orc_c["ceil_A"],
                            ceil_B=orc_c["ceil_B"]))

    # --- C1/C3: union student (2 seeds) + interference references + K3 baseline ---
    union_students, union_evals = [], []
    for seed in (1, 2, 3, 4, 5):
        s = distill(frame.field_np, frame.lo, frame.hi, seed)
        m = eval_student(s, frame, orc)
        union_students.append(s); union_evals.append(m)
        print(f"  union-field seed {seed}: joint {m['basin_joint']:.2f} "
              f"A {m['basin_A']:.2f} B {m['basin_B']:.2f} sites {m['sites_recovered']}/{m['sites_total']}")
    out["union"] = union_evals

    singleA = distill(lambda z: np_field(A, z), frame.lo, frame.hi, 1)
    mA = eval_student(singleA, frame, orc)
    singleB = distill(lambda z: (np_step(B, (z - frame.T[1]) @ frame.T[2]) @ frame.T[0]
                                 + frame.T[1]) - z, frame.lo, frame.hi, 1)
    mB = eval_student(singleB, frame, orc)
    print(f"  single-A: A-side {mA['basin_A']:.2f} | single-B(conj): B-side {mB['basin_B']:.2f}")
    out["single_A"], out["single_B"] = mA, mB

    # K3: merged output distillation (sees pooled data, per-sample teacher targets)
    import mlx.nn as nn
    import mlx.optimizers as optim
    tgt = np.array(A(mx.array(xh)))
    tgt[labels >= 4] = np.array(B(mx.array(xB)))
    merged_evals = []
    for seed in (1, 2):
        mx.random.seed(300 + seed)
        ms = AE(d_in=16); mx.eval(ms.parameters())
        opt = optim.Adam(learning_rate=1e-3)
        X, Tt = mx.array(xh), mx.array(tgt)
        rng2 = np.random.default_rng(400 + seed)
        def loss_fn(m, xb, tb): return mx.mean((m(xb) - tb) ** 2)
        lg = nn.value_and_grad(ms, loss_fn)
        for _ in range(STUDENT_STEPS):
            idx = mx.array(rng2.integers(0, len(xh), 256))
            _, g = lg(ms, X[idx], Tt[idx])
            opt.update(ms, g); mx.eval(ms.parameters(), opt.state)
        al = fit_affine(np.array(ms.enc(mx.array(xh))), np.array(A.enc(mx.array(xh))))
        cands = [eval_student(ms, frame, orc)]
        if al is not None:
            cands.append(eval_student(ms, frame, orc, align=al))
        best = max(cands, key=lambda r: r["basin_joint"])
        merged_evals.append(best)
        print(f"  merged-outdistill seed {seed}: joint {best['basin_joint']:.2f} "
              f"A {best['basin_A']:.2f} B {best['basin_B']:.2f}")
    out["merged_outdistill"] = merged_evals

    # --- C4/C2: lineages (seed 1, gen1 = union student seed 1) ---
    def lineage(first_student, fr, orc_ref, name):
        gens, cur = [], first_student
        gens.append(eval_student(cur, fr, orc_ref))
        for g in range(2, GENS + 1):
            cur = distill(lambda z: np_field(cur, z), fr.lo, fr.hi, 40 + g)
            gens.append(eval_student(cur, fr, orc_ref))
            e = gens[-1]
            print(f"  {name} gen {g}: joint {e['basin_joint']:.2f} "
                  f"A {e['basin_A']:.2f} B {e['basin_B']:.2f} attr {e['n_attractors']}")
        return gens

    best_i = int(np.argmax([m["basin_joint"] for m in union_evals]))
    print(f"clean lineage (ancestor = union seed {best_i+1}, best of 5) ...")
    lin_clean = lineage(union_students[best_i], frame, orc, "clean")
    print("contaminated lineage ...")
    s_c = distill(frame_c.field_np, frame_c.lo, frame_c.hi, 1)
    e1 = eval_student(s_c, frame_c, orc_c)
    print(f"  contaminated gen 1: joint {e1['basin_joint']:.2f} A {e1['basin_A']:.2f} "
          f"Bmem {e1['basin_B']:.2f}")
    lin_cont = lineage(s_c, frame_c, orc_c, "contaminated")
    out["lineage_clean"], out["lineage_contaminated"] = lin_clean, lin_cont

    # --- gates ---
    uj = float(np.mean([m["basin_joint"] for m in union_evals]))
    uA = float(np.mean([m["basin_A"] for m in union_evals]))
    uB = float(np.mean([m["basin_B"] for m in union_evals]))
    mj = float(np.mean([m["basin_joint"] for m in merged_evals]))
    sites = max(m["sites_recovered"] for m in union_evals)
    c1 = (uj >= 0.60) and (mA["basin_A"] - uA <= 0.15) and (mB["basin_B"] - uB <= 0.15)
    k3 = mj >= uj - 0.05
    c3 = sites >= min(7, orc["attractors"].shape[0] - 1)
    c4 = lin_clean[-1]["basin_joint"] >= 0.85 * lin_clean[0]["basin_joint"]
    c2 = (lin_clean[-1]["basin_B"] >= 0.8 * lin_clean[0]["basin_B"]
          and lin_cont[2]["basin_B"] <= 0.30
          and lin_cont[-1]["basin_A"] >= 0.8 * lin_cont[0]["basin_A"])
    golden = c1 and (not k3) and c3 and c4 and c2
    verdict = ("GOLDEN GO" if golden else
               "NO-GO (K3 kill-switch: behavioral merging matches field union)" if k3 else
               f"PARTIAL (C1={c1} C2={c2} C3={c3} C4={c4}, K3 clear)")
    gates = dict(C1_union=bool(c1), K3_kill=bool(k3), C3_exceedance=bool(c3),
                 C4_lineage=bool(c4), C2_filter=bool(c2), verdict=verdict,
                 union_joint=uj, merged_joint=mj, sites=sites)
    out["gates"] = gates

    print(f"\nVERDICT: {verdict}")
    print(f"  union joint {uj:.2f} (ceiling {orc['ceil_joint']:.2f}) vs merged {mj:.2f}; "
          f"sites {sites}/{len(orc['attractors'])}; "
          f"lineage gen5/gen1 {lin_clean[-1]['basin_joint']:.2f}/{lin_clean[0]['basin_joint']:.2f}; "
          f"contaminated B-side gen1->3->5: {lin_cont[0]['basin_B']:.2f}->"
          f"{lin_cont[2]['basin_B']:.2f}->{lin_cont[-1]['basin_B']:.2f}")
    (RUNS / "gi.json").write_text(json.dumps(out, indent=2))

    # --- figure ---
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    ax = axes[0]
    G = orc["G"]; V = frame.field_np(G)
    n = int(np.sqrt(len(G))); sub = 1
    mag = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    ax.quiver(G[:, 0].reshape(n, n), G[:, 1].reshape(n, n),
              (V / mag)[:, 0].reshape(n, n), (V / mag)[:, 1].reshape(n, n),
              np.linalg.norm(V, axis=1).reshape(n, n), cmap="viridis",
              scale=45, width=0.003, alpha=0.85)
    ax.scatter(orc["attractors"][:, 0], orc["attractors"][:, 1], marker="*",
               s=200, c="black", zorder=5, label="union oracle attractors")
    su = union_evals[0]
    ep_probe = np.array(iterate(union_students[0],
                                mx.array(orc["probes"]), ORACLE_STEPS,
                                frame.clip_lo, frame.clip_hi))
    vns = np.linalg.norm(np_field(union_students[0], ep_probe), axis=1)
    attr_s = find_attractors(ep_probe, vns, vtol=5e-3, merge_r=0.15)
    if len(attr_s):
        ax.scatter(attr_s[:, 0], attr_s[:, 1], marker="o", s=110, facecolors="none",
                   edgecolors="red", linewidths=2, zorder=6, label="union student attractors")
    ax.set_title(f"Union field + student (joint {su['basin_joint']:.2f}, "
                 f"sites {su['sites_recovered']}/{su['sites_total']})")
    ax.legend(fontsize=7); ax.set_aspect("equal")

    ax = axes[1]
    gens = np.arange(1, GENS + 1)
    ax.plot(gens, [g["basin_joint"] for g in lin_clean], "o-", label="clean: joint")
    ax.plot(gens, [g["basin_A"] for g in lin_clean], "s--", label="clean: A-side")
    ax.plot(gens, [g["basin_B"] for g in lin_clean], "^--", label="clean: B-side (ordered graft)")
    ax.axhline(orc["ceil_joint"], color="gray", ls=":", label="oracle ceiling")
    ax.set_xlabel("generation"); ax.set_ylabel("basin agreement vs gen-0 oracle")
    ax.set_ylim(0, 1.05); ax.set_title("C4: clean lineage (photocopy test)")
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.plot(gens, [g["basin_A"] for g in lin_cont], "s--", color="#2ca02c",
            label="contaminated: A-side (ordered, native)")
    ax.plot(gens, [g["basin_B"] for g in lin_cont], "v-", color="#d62728",
            label="contaminated: B_mem-side (disordered graft)")
    ax.plot(gens, [g["basin_B"] for g in lin_clean], "^--", color="#1f77b4", alpha=0.6,
            label="clean: B-side (ordered graft, contrast)")
    ax.set_xlabel("generation"); ax.set_ylim(0, 1.05)
    ax.set_title("C2: the onion filter (ordered persists, disordered dies)")
    ax.legend(fontsize=7)

    fig.suptitle("Composition and inheritance of latent vector fields", fontsize=13)
    fig.tight_layout()
    fig.savefig(RUNS / "gi.png", dpi=140)
    print(f"wrote {RUNS/'gi.json'} and {RUNS/'gi.png'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
