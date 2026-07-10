"""External validation of issue #1 (inem) INSIDE our own harness.

Claims under test, now against OUR teachers / probes / metrics (his scripts
reimplemented the world; here nothing is reimplemented):

  E1 (his claim 1): a read-only lattice carrier in our gi.py union world
      removes the seam/lineage losses AND inherits the memorization graft
      (onion filter = lossy-SGD selection, not a channel property).
  E2 (his claims 2+3): on our cached MNIST teacher, a k-NN lookup carrier
      beats our field arm despite worse NMSE (cost curve, no plateau), and a
      GP/kernel-ridge carrier from 16k pairs matches multi-million lookups.
      Records BOTH sigma-selection rules (held-out NMSE vs direction/cos) to
      test his "basin-relevant model selection" point.
  E3 (new, not in his gist): two-hop GP -> SGD. Train our standard parametric
      field student against the GP carrier's field (unlimited queries, zero
      further teacher access; z deepened by iterating the GP field itself).
      Decides whether the wall was sample complexity (two-hop ~= GP) or the
      parametric carrier itself (two-hop ~= direct field arm 0.09).

Staged CLI (each stage bounded, resumable):
  uv run python -m m5_fielddistill.extval gi
  uv run python -m m5_fielddistill.extval table
  uv run python -m m5_fielddistill.extval carriers <seed>
  uv run python -m m5_fielddistill.extval twohop <seed>
  uv run python -m m5_fielddistill.extval report

Writes runs/extval_gi.json + runs/extval_mnist.json (+ runs/extval_table.npz cache).
"""

import json
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .ae import field, find_attractors, iterate
from .gi import (BASIN_EPS, GENS, ORACLE_STEPS, RUNS, UnionFrame, make_world8,
                 np_field)
from .mnist import (D_LAT, ITER_STEPS, LR, STUDENT_STEPS, evaluate,
                    load_mnist, load_teacher, make_ae)
from .train import train_teacher

GI_OUT = RUNS / "extval_gi.json"
MNIST_OUT = RUNS / "extval_mnist.json"
TABLE = RUNS / "extval_table.npz"

GRID_N = 512                 # 2D lattice resolution (262,144 field samples)
N_TABLE = 1_000_000          # MNIST-tier (z, V) table size
NN_SIZES = (16_384, 262_144, 1_000_000)
M_GP = 16_384                # GP inducing set (his headline budget)
M_SELECT, N_VAL = 4096, 4096
SIGMA_MULTS = (0.05, 0.1, 0.15, 0.25, 0.35, 0.5)
JITTER = 1e-8


def _append(path, rec):
    rows = json.loads(path.read_text()) if path.exists() else []
    rows.append(rec)
    path.write_text(json.dumps(rows, indent=2))


# ---------- E1: read carrier in OUR gi world ----------

class LatticeCarrier:
    """Piecewise-constant field snapshot on a regular 2D lattice; read = nearest cell."""

    def __init__(self, field_fn, lo, hi, n=GRID_N, chunk=8192, origin_jitter=None):
        self.lo, self.hi, self.n = lo.copy(), hi.copy(), n
        self.h = (hi - lo) / (n - 1)
        if origin_jitter is not None:
            self.lo = self.lo + origin_jitter * self.h
        xs = self.lo[0] + np.arange(n, dtype=np.float32) * self.h[0]
        ys = self.lo[1] + np.arange(n, dtype=np.float32) * self.h[1]
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
        V = np.empty_like(pts)
        for i in range(0, len(pts), chunk):
            V[i:i + chunk] = field_fn(pts[i:i + chunk])
        self.V = V.reshape(n, n, 2)
        self.n_samples = len(pts)

    def lookup(self, z):
        ij = np.clip(np.rint((z - self.lo) / self.h).astype(np.int64), 0, self.n - 1)
        return self.V[ij[:, 0], ij[:, 1]]

    def iterate(self, z, clip_lo, clip_hi, steps=ORACLE_STEPS):
        z = z.copy()
        for _ in range(steps):
            z = np.clip(z + self.lookup(z), clip_lo, clip_hi)
        return z


def eval_carrier_gi(car, frame, orc):
    ep = car.iterate(orc["probes"], frame.clip_lo, frame.clip_hi)
    d = np.linalg.norm(ep - orc["ep"], axis=1)
    side = orc["side"]
    epg = car.iterate(orc["G"], frame.clip_lo, frame.clip_hi)
    vn = np.linalg.norm(car.lookup(epg), axis=1)
    attr = find_attractors(epg, vn, vtol=5e-3, merge_r=0.15)
    tgt = orc["attractors"]
    rec = int(sum(np.min(np.linalg.norm(tgt[i] - attr, axis=1)) < BASIN_EPS
                  for i in range(len(tgt)))) if len(attr) else 0
    return dict(basin_joint=float((d < BASIN_EPS).mean()),
                basin_A=float((d[~side] < BASIN_EPS).mean()),
                basin_B=float((d[side] < BASIN_EPS).mean()),
                n_attractors=int(len(attr)), sites_recovered=rec,
                sites_total=int(len(tgt)))


def stage_gi():
    t0 = time.time()
    rng = np.random.default_rng(7)                      # same rng path as gi.main
    xh, labels = make_world8()
    xA, xB = xh[labels < 4], xh[labels >= 4]
    print("training teachers A, B, B_mem (identical protocol to gi.main) ...")
    A = train_teacher(xA, noise=0.1, seed=0)
    B = train_teacher(xB, noise=0.1, seed=0)
    B_mem = train_teacher(xB, noise=0.0, seed=0)

    out = {}
    for tag, Bt in (("clean", B), ("contaminated", B_mem)):
        frame = UnionFrame(A, Bt, xh, labels, mode="iso")
        orc = frame.oracle(rng)
        print(f"[{tag}] oracle: {len(orc['attractors'])} attractors, ceilings "
              f"joint {orc['ceil_joint']:.2f} A {orc['ceil_A']:.2f} B {orc['ceil_B']:.2f}")
        car = LatticeCarrier(frame.field_np, frame.lo, frame.hi)
        lineage = [eval_carrier_gi(car, frame, orc)]
        cur = car
        for g in range(2, GENS + 1):
            jit = rng.random(2).astype(np.float32)
            cur = LatticeCarrier(cur.lookup, frame.lo, frame.hi, origin_jitter=jit)
            lineage.append(eval_carrier_gi(cur, frame, orc))
        for g, e in enumerate(lineage, 1):
            print(f"  [{tag}] gen {g}: joint {e['basin_joint']:.2f} "
                  f"A {e['basin_A']:.2f} B {e['basin_B']:.2f} "
                  f"sites {e['sites_recovered']}/{e['sites_total']}")
        out[tag] = dict(ceilings=dict(joint=orc["ceil_joint"], A=orc["ceil_A"],
                                      B=orc["ceil_B"]),
                        n_oracle_attractors=int(len(orc["attractors"])),
                        carrier_samples=car.n_samples, lineage=lineage)
    out["sgd_reference"] = "runs/gi.json (union 0.82, lineage 0.91->0.83, B_mem 0.24->0.19)"
    GI_OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {GI_OUT} [{time.time()-t0:.0f}s]")


# ---------- E2: carriers on OUR cached MNIST teacher ----------

def build_table(teacher, meta, n=N_TABLE, seed=0, bs=8192):
    """(z, V) pairs from the SAME sampling prior as stage_arm's sample_z."""
    mu, sd = meta["mu"], meta["sd"]
    rng = np.random.default_rng(seed)
    Z = np.empty((n, D_LAT), dtype=np.float32)
    V = np.empty((n, D_LAT), dtype=np.float32)
    for i in range(0, n, bs):
        k = min(bs, n - i)
        half = k // 2
        broad = (mu + 2.0 * sd * rng.standard_normal((half, D_LAT))).astype(np.float32)
        core = mx.array((mu + 1.5 * sd * rng.standard_normal((k - half, D_LAT)))
                        .astype(np.float32))
        depth = int(rng.integers(2, 25))
        for _ in range(depth):
            core = teacher.enc(teacher.dec(core))
        zb = mx.concatenate([mx.array(broad), core], axis=0)
        vb = field(teacher, zb)
        mx.eval(zb, vb)
        Z[i:i + k], V[i:i + k] = np.array(zb), np.array(vb)
    return Z, V


def stage_table():
    t0 = time.time()
    teacher, meta = load_teacher()
    Z, V = build_table(teacher, meta)
    np.savez_compressed(TABLE, Z=Z, V=V)
    print(f"wrote {TABLE}: {len(Z)} pairs [{time.time()-t0:.0f}s]")


def knn_field(Zs, Vs, Zsq, zq):
    d2 = Zsq[None, :] - 2 * (zq @ Zs.T) + mx.sum(zq * zq, axis=1)[:, None]
    return Vs[mx.argmin(d2, axis=1)]


def carrier_metrics(field_fn, teacher, meta, label):
    """Iterate a carrier field from our probes; report our standard metrics."""
    z = mx.array(meta["probes"])
    lo, hi = mx.array(meta["clip_lo"]), mx.array(meta["clip_hi"])
    for _ in range(ITER_STEPS):
        z = mx.maximum(mx.minimum(z + field_fn(z), hi), lo)
        mx.eval(z)
    ep = np.array(z)
    dists = np.linalg.norm(ep - meta["ep_probes"], axis=1)
    basin = float((dists < float(meta["basin_eps"])).mean())
    img_s = np.array(teacher.dec(mx.array(ep.astype(np.float32))))
    img_t = np.array(teacher.dec(mx.array(meta["ep_probes"])))
    dec = float((((img_s - img_t) ** 2).mean(axis=1) < 0.01).mean())
    Vh = np.array(field_fn(mx.array(meta["zeval"])))
    Vt = meta["Vt_eval"]
    nmse = float(((Vh - Vt) ** 2).mean() / (Vt ** 2).mean())
    mag = np.linalg.norm(Vt, axis=1)
    mask = mag > np.percentile(mag, 10)
    cos = float((np.sum(Vh[mask] * Vt[mask], axis=1)
                 / (np.linalg.norm(Vh[mask], axis=1) * mag[mask] + 1e-12)).mean())
    m = dict(carrier=label, basin_agreement=basin, dec_agree=dec,
             endpoint_median=float(np.median(dists)), field_nmse=nmse, field_cos=cos)
    print(f"  {label}: basin {basin:.3f}  dec {dec:.3f}  nmse {nmse:.3f}  cos {cos:.2f}")
    return m


def rbf_solve(ZM, VM, sigma):
    Zd = ZM.astype(np.float64)
    sq = (Zd ** 2).sum(1)
    K = np.exp(-np.maximum(sq[:, None] - 2 * Zd @ Zd.T + sq[None, :], 0)
               / (2 * sigma ** 2))
    K[np.diag_indices_from(K)] += JITTER * K.shape[0]
    return np.linalg.solve(K, VM.astype(np.float64))


class GPCarrier:
    def __init__(self, ZM, VM, sigma):
        self.sigma = sigma
        alpha = rbf_solve(ZM, VM, sigma)
        self.Zm = mx.array(ZM)
        self.alpha = mx.array(alpha.astype(np.float32))
        self.Zsq = mx.sum(self.Zm * self.Zm, axis=1)
        mx.eval(self.Zm, self.alpha, self.Zsq)

    def __call__(self, zq):
        d2 = self.Zsq[None, :] - 2 * (zq @ self.Zm.T) + mx.sum(zq * zq, axis=1)[:, None]
        return mx.exp(-mx.maximum(d2, 0.0) / (2 * self.sigma ** 2)) @ self.alpha


def _gp_pick(Zn, Vn, seed):
    """sigma selection at M_SELECT on held-out table points, both rules."""
    rng = np.random.default_rng(9000 + seed)
    perm = rng.permutation(len(Zn))
    val, pool = perm[:N_VAL], perm[N_VAL:]
    Zval, Vval = Zn[val], Vn[val]
    ZM, VM = Zn[pool[:M_SELECT]], Vn[pool[:M_SELECT]]
    d_med = float(np.median(np.linalg.norm(
        ZM[rng.integers(0, M_SELECT, 2000)] - ZM[rng.integers(0, M_SELECT, 2000)], axis=1)))
    grid = []
    for mult in SIGMA_MULTS:
        gp = GPCarrier(ZM, VM, mult * d_med)
        Vh = np.array(gp(mx.array(Zval)))
        nmse = float(((Vh - Vval) ** 2).mean() / (Vval ** 2).mean())
        magv = np.linalg.norm(Vval, axis=1)
        mk = magv > np.percentile(magv, 10)
        cos = float((np.sum(Vh[mk] * Vval[mk], axis=1)
                     / (np.linalg.norm(Vh[mk], axis=1) * magv[mk] + 1e-12)).mean())
        grid.append(dict(mult=mult, nmse=nmse, cos=cos))
        print(f"  sigma {mult:.2f}x median({d_med:.2f}): held-out nmse {nmse:.4f} cos {cos:.3f}")
    by_nmse = min(grid, key=lambda g: g["nmse"])["mult"]
    by_cos = max(grid, key=lambda g: g["cos"])["mult"]
    return d_med, by_nmse, by_cos, grid, pool


def stage_carriers(seed):
    t0 = time.time()
    teacher, meta = load_teacher()
    tab = np.load(TABLE)
    Zn, Vn = tab["Z"], tab["V"]
    rng = np.random.default_rng(seed)
    recs = []

    for n in NN_SIZES:
        idx = rng.permutation(len(Zn))[:n] if n < len(Zn) else np.arange(len(Zn))
        Zs, Vs = mx.array(Zn[idx]), mx.array(Vn[idx])
        Zsq = mx.sum(Zs * Zs, axis=1)
        mx.eval(Zs, Vs, Zsq)
        recs.append(carrier_metrics(lambda z: knn_field(Zs, Vs, Zsq, z),
                                    teacher, meta, f"nn1_N{n}") | dict(seed=seed))

    d_med, by_nmse, by_cos, grid, pool = _gp_pick(Zn, Vn, seed)
    print(f"selected sigma: nmse-rule {by_nmse}x | cos-rule {by_cos}x")
    ZM, VM = Zn[pool[:M_GP]], Vn[pool[:M_GP]]
    done = {}
    for rule, mult in (("nmse", by_nmse), ("cos", by_cos)):
        if mult in done:
            recs.append(done[mult] | dict(carrier=f"gp_M{M_GP}_{rule}{mult}", seed=seed))
            continue
        gp = GPCarrier(ZM, VM, mult * d_med)
        m = carrier_metrics(gp, teacher, meta, f"gp_M{M_GP}_{rule}{mult}")
        m |= dict(seed=seed, sigma_mult=mult, selection=rule, sel_grid=grid)
        done[mult] = m
        recs.append(m)
    for r in recs:
        _append(MNIST_OUT, r)
    print(f"carriers seed {seed} done [{time.time()-t0:.0f}s]")


# ---------- E3: two-hop GP -> parametric student ----------

def stage_twohop(seed):
    """Our standard field-arm student, supervised by the GP carrier instead of
    the teacher. Teacher access: the 16k+4k table draws only — z sampling is
    deepened by iterating the GP field itself, not the teacher."""
    t0 = time.time()
    xtr, ytr, xte, yte = load_mnist()
    teacher, meta = load_teacher()
    tab = np.load(TABLE)
    Zn, Vn = tab["Z"], tab["V"]
    d_med, by_nmse, by_cos, grid, pool = _gp_pick(Zn, Vn, seed)
    gp = GPCarrier(Zn[pool[:M_GP]], Vn[pool[:M_GP]], by_cos * d_med)
    print(f"twohop seed {seed}: GP sigma {by_cos}x median (cos rule)")

    mu, sd = meta["mu"], meta["sd"]
    rng = np.random.default_rng(1000 + seed)
    student = make_ae(100 + seed)
    opt = optim.Adam(learning_rate=optim.cosine_decay(LR, STUDENT_STEPS, 1e-4))

    def sample_z(k):
        half = k // 2
        broad = mx.array((mu + 2.0 * sd * rng.standard_normal((half, D_LAT)))
                         .astype(np.float32))
        core = mx.array((mu + 1.5 * sd * rng.standard_normal((k - half, D_LAT)))
                        .astype(np.float32))
        depth = int(rng.integers(2, 25))
        for _ in range(depth):
            core = core + gp(core)                    # deepen via the GP field
        return mx.concatenate([broad, core], axis=0)

    def loss_fn(m, zb, vt):
        return mx.mean((field(m, zb) - vt) ** 2)

    lg = nn.value_and_grad(student, loss_fn)
    for i in range(STUDENT_STEPS):
        zb = sample_z(256)
        vt = gp(zb)
        mx.eval(zb, vt)
        _, g = lg(student, zb, vt)
        g, _ = optim.clip_grad_norm(g, max_norm=5.0)
        opt.update(student, g)
        mx.eval(student.parameters(), opt.state)

    m = evaluate(student, teacher, meta, xtr, xte)
    m |= dict(carrier="twohop_gp_sgd", seed=seed, sigma_mult=by_cos)
    _append(MNIST_OUT, m)
    print(f"twohop seed {seed}: basin {m['basin_agreement']:.3f} "
          f"dec {m['dec_agree']:.3f} cos {m['field_cos']:.2f} "
          f"nmse {m['field_nmse']:.3f} [{time.time()-t0:.0f}s]")


class ChunkedNN:
    """1-NN lookup field over the full table, evaluated in bounded-memory
    chunks (running min across chunks; ~134MB transient per chunk instead of
    a single 1GB distance matrix — the unchunked version thrashes 24GB RAM
    when chained inside a training graph)."""

    def __init__(self, Zn, Vn, chunk=131_072, qblock=2048):
        self.qblock = qblock
        self.parts = []
        for i in range(0, len(Zn), chunk):
            # fp16 for the distance computation (memory-bound; a rare
            # 2nd-nearest mispick is harmless for a lookup FIELD), fp32 V.
            Zh = mx.array(Zn[i:i + chunk]).astype(mx.float16)
            Vs = mx.array(Vn[i:i + chunk])
            Zsq = mx.sum(Zh.astype(mx.float32) * Zh.astype(mx.float32),
                         axis=1).astype(mx.float16)
            mx.eval(Zh, Vs, Zsq)
            self.parts.append((Zh, Vs, Zsq))

    def _block(self, zq):
        zh = zq.astype(mx.float16)
        best_d, best_v = None, None
        for Zh, Vs, Zsq in self.parts:
            # |q|^2 is constant per query row -> irrelevant to the argmin
            d2 = Zsq[None, :] - 2 * (zh @ Zh.T)
            idx = mx.argmin(d2, axis=1)
            md = mx.take_along_axis(d2, idx[:, None], axis=1)[:, 0]
            v = Vs[idx]
            if best_d is None:
                best_d, best_v = md, v
            else:
                best_v = mx.where((md < best_d)[:, None], v, best_v)
                best_d = mx.minimum(md, best_d)
            mx.eval(best_d, best_v)
        return best_v

    def __call__(self, zq):
        n = zq.shape[0]
        if n <= self.qblock:
            return self._block(zq)
        return mx.concatenate([self._block(zq[i:i + self.qblock])
                               for i in range(0, n, self.qblock)], axis=0)


def stage_twohop_nn(seed):
    """Two-hop from the BEST carrier: our standard parametric field student
    supervised by the k-NN lookup field over the full 1M table (basin 0.41).
    Same sample budget as the direct field arm (STUDENT_STEPS x 256 pairs,
    one pass), but the pool is pre-generated in bulk batches so the table
    lookups amortize (per-step lookups are sync-bound and ~100x slower).
    If this transfers, the data-free pipeline teacher -> table -> SGD student
    stands end-to-end at 16D."""
    t0 = time.time()
    xtr, ytr, xte, yte = load_mnist()
    teacher, meta = load_teacher()
    tab = np.load(TABLE)
    nn_f = ChunkedNN(tab["Z"], tab["V"])
    mu, sd = meta["mu"], meta["sd"]
    rng = np.random.default_rng(1000 + seed)

    n_pool = STUDENT_STEPS * 256
    bs = 4096
    Zp = np.empty((n_pool, D_LAT), dtype=np.float32)
    Vp = np.empty((n_pool, D_LAT), dtype=np.float32)
    for i in range(0, n_pool, bs):
        k = min(bs, n_pool - i)
        half = k // 2
        broad = mx.array((mu + 2.0 * sd * rng.standard_normal((half, D_LAT)))
                         .astype(np.float32))
        core = mx.array((mu + 1.5 * sd * rng.standard_normal((k - half, D_LAT)))
                        .astype(np.float32))
        depth = int(rng.integers(2, 25))
        for _ in range(depth):
            core = core + nn_f(core)                  # deepen via the NN field
        zb = mx.concatenate([broad, core], axis=0)
        vt = nn_f(zb)
        mx.eval(zb, vt)
        Zp[i:i + k], Vp[i:i + k] = np.array(zb), np.array(vt)
    print(f"  pool {n_pool} NN-field pairs [{time.time()-t0:.0f}s]")

    student = make_ae(100 + seed)
    opt = optim.Adam(learning_rate=optim.cosine_decay(LR, STUDENT_STEPS, 1e-4))

    def loss_fn(m, zb, vt):
        return mx.mean((field(m, zb) - vt) ** 2)

    lg = nn.value_and_grad(student, loss_fn)
    for i in range(STUDENT_STEPS):
        idx = rng.integers(0, n_pool, 256)
        zb, vt = mx.array(Zp[idx]), mx.array(Vp[idx])
        _, g = lg(student, zb, vt)
        g, _ = optim.clip_grad_norm(g, max_norm=5.0)
        opt.update(student, g)
        mx.eval(student.parameters(), opt.state)

    m = evaluate(student, teacher, meta, xtr, xte)
    m |= dict(carrier="twohop_nn1M_sgd", seed=seed)
    _append(MNIST_OUT, m)
    print(f"twohop-nn seed {seed}: basin {m['basin_agreement']:.3f} "
          f"dec {m['dec_agree']:.3f} cos {m['field_cos']:.2f} "
          f"nmse {m['field_nmse']:.3f} [{time.time()-t0:.0f}s]")


def stage_report():
    print("=== E1 (gi read carrier) ===")
    if GI_OUT.exists():
        gi = json.loads(GI_OUT.read_text())
        for tag in ("clean", "contaminated"):
            lin = gi[tag]["lineage"]
            print(f"  {tag}: ceilings {gi[tag]['ceilings']} | "
                  f"joint {[round(e['basin_joint'], 2) for e in lin]} | "
                  f"B-side {[round(e['basin_B'], 2) for e in lin]}")
    print("=== E2/E3 (mnist carriers + twohop) ===")
    if MNIST_OUT.exists():
        rows = json.loads(MNIST_OUT.read_text())
        by = {}
        for r in rows:
            key = r["carrier"].split("_seed")[0] if "twohop" in r["carrier"] else r["carrier"]
            by.setdefault(key, []).append(r["basin_agreement"])
        for k, v in sorted(by.items()):
            print(f"  {k}: basin {np.mean(v):.3f} +/- {np.std(v):.3f}  (n={len(v)}: "
                  f"{[round(x, 3) for x in v]})")
    print("reference: our field arm 0.09 | outdistill 0.19 | ceiling 0.71")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "gi":
        stage_gi()
    elif cmd == "table":
        stage_table()
    elif cmd == "carriers":
        stage_carriers(int(sys.argv[2]))
    elif cmd == "twohopnn":
        stage_twohop_nn(int(sys.argv[2]))
    elif cmd == "twohop":
        stage_twohop(int(sys.argv[2]))
    elif cmd == "report":
        stage_report()
    else:
        raise SystemExit(f"unknown stage {cmd}")


if __name__ == "__main__":
    main()
