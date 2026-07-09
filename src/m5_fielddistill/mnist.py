"""MNIST tier: does vector-field distillation survive dimensionality (latent 16)?

No grid exists at 16D — the field must be matched from SPARSE probes. Sampling
prior for field arms: Gaussian fitted to the teacher's latent moments (2x16
numbers; a calibration-level leak, documented) + short teacher trajectories.
Students otherwise see no data (except fieldplus's 5% anchor).

Staged CLI (each stage bounded, resumable):
  uv run python -m m5_fielddistill.mnist teacher          # train + cache teacher
  uv run python -m m5_fielddistill.mnist arm <name> <seed> # one student + eval
  uv run python -m m5_fielddistill.mnist report            # table, verdict, images

Arms: field | fieldplus | scratch | outdistill | latmatch  (as in train.py; fieldk
dropped — lost to 1-step twice in the 2D tier).
"""

import gzip
import json
import sys
import time
import urllib.request
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .ae import AE, chamfer, field, find_attractors, fit_affine, iterate

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
DATA = ROOT / "data" / "mnist"
MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {"train_x": "train-images-idx3-ubyte.gz", "train_y": "train-labels-idx1-ubyte.gz",
         "test_x": "t10k-images-idx3-ubyte.gz", "test_y": "t10k-labels-idx1-ubyte.gz"}

D_IN, D_LAT = 784, 16
ENC_DIMS = [784, 256, 128, 16]
DEC_DIMS = [16, 128, 256, 784]
N_TRAIN, N_TEST = 8000, 2000
STEPS, BS, LR = 3500, 256, 1e-3
STUDENT_STEPS = 12000
TRAJ_KS = (2, 8)
TEACHER_NOISE = 0.2
ITER_STEPS = 800
N_PROBES = 256

TEACHER_W = RUNS / "mnist_teacher.safetensors"
META = RUNS / "mnist_meta.npz"
RESULTS = RUNS / "mnist_results.json"


# ---------- data ----------

def load_mnist():
    DATA.mkdir(parents=True, exist_ok=True)
    arrs = {}
    for key, fname in FILES.items():
        p = DATA / fname
        if not p.exists():
            print(f"downloading {fname} ...")
            urllib.request.urlretrieve(MIRROR + fname, p)
        raw = gzip.open(p, "rb").read()
        if "x" in key:
            arrs[key] = (np.frombuffer(raw, np.uint8, offset=16)
                         .reshape(-1, 784).astype(np.float32) / 255.0)
        else:
            arrs[key] = np.frombuffer(raw, np.uint8, offset=8).astype(np.int64)
    return (arrs["train_x"][:N_TRAIN], arrs["train_y"][:N_TRAIN],
            arrs["test_x"][:N_TEST], arrs["test_y"][:N_TEST])


def make_ae(seed):
    mx.random.seed(seed)
    m = AE(enc_dims=ENC_DIMS, dec_dims=DEC_DIMS)
    mx.eval(m.parameters())
    return m


# ---------- stage: teacher ----------

def stage_teacher(noise=TEACHER_NOISE, steps=STEPS):
    t0 = time.time()
    xtr, ytr, xte, yte = load_mnist()
    model = make_ae(0)
    opt = optim.Adam(learning_rate=LR)
    X = mx.array(xtr)
    rng = np.random.default_rng(0)

    def loss_fn(m, xb, xn):
        return mx.mean((m(xn) - xb) ** 2)

    lg = nn.value_and_grad(model, loss_fn)
    for i in range(steps):
        idx = mx.array(rng.integers(0, N_TRAIN, BS))
        xb = X[idx]
        xn = xb + noise * mx.array(rng.standard_normal((BS, D_IN)).astype(np.float32))
        _, g = lg(model, xb, xn)
        opt.update(model, g)
        mx.eval(model.parameters(), opt.state)
        if (i + 1) % 1000 == 0:
            print(f"  step {i+1}/{steps} [{time.time()-t0:.0f}s]")

    z_tr = np.array(model.enc(X))
    z_te = np.array(model.enc(mx.array(xte)))
    mu, sd = z_tr.mean(0), z_tr.std(0)
    scale = float(np.linalg.norm(sd))
    clip_lo = (mu - 8 * sd).astype(np.float32)
    clip_hi = (mu + 8 * sd).astype(np.float32)

    rng = np.random.default_rng(7)
    starts = np.concatenate([
        z_te[rng.integers(0, N_TEST, 256)] + 0.5 * sd * rng.standard_normal((256, D_LAT)),
        mu + 2.0 * sd * rng.standard_normal((256, D_LAT)),
    ]).astype(np.float32)
    ep = np.array(iterate(model, mx.array(starts), ITER_STEPS, clip_lo, clip_hi))
    vn = np.linalg.norm(np.array(field(model, mx.array(ep))), axis=1)
    merge_r = 0.1 * scale
    attractors = find_attractors(ep, vn, vtol=1e-2, merge_r=merge_r)

    if len(attractors) > 1:
        dm = np.linalg.norm(attractors[:, None] - attractors[None, :], axis=-1)
        basin_eps = 0.25 * float(np.median(dm[dm > 0]))
    else:
        basin_eps = 0.25 * scale

    probes = np.concatenate([
        z_te[rng.integers(0, N_TEST, N_PROBES // 2)]
        + 0.5 * sd * rng.standard_normal((N_PROBES // 2, D_LAT)),
        mu + 2.0 * sd * rng.standard_normal((N_PROBES // 2, D_LAT)),
    ]).astype(np.float32)
    ep_probes = np.array(iterate(model, mx.array(probes), ITER_STEPS, clip_lo, clip_hi))
    ep_pert = np.array(iterate(model, mx.array(
        probes + (0.05 * scale) * rng.standard_normal(probes.shape).astype(np.float32)),
        ITER_STEPS, clip_lo, clip_hi))
    self_agree = float((np.linalg.norm(ep_pert - ep_probes, axis=1) < basin_eps).mean())
    conv_frac = float((vn < 1e-2).mean())
    # decoded-endpoint ceiling: same comparison through the decoder (pixel space)
    img_a = np.array(model.dec(mx.array(ep_probes)))
    img_b = np.array(model.dec(mx.array(ep_pert)))
    dec_ceiling = float((((img_a - img_b) ** 2).mean(axis=1) < 0.01).mean())

    # eval z-set for field metrics: near-manifold + mid + far
    zeval = np.concatenate([
        z_te[rng.integers(0, N_TEST, 512)] + 0.2 * sd * rng.standard_normal((512, D_LAT)),
        z_te[rng.integers(0, N_TEST, 512)] + 1.0 * sd * rng.standard_normal((512, D_LAT)),
        mu + 2.0 * sd * rng.standard_normal((512, D_LAT)),
    ]).astype(np.float32)
    Vt_eval = np.array(field(model, mx.array(zeval)))

    model.save_weights(str(TEACHER_W))
    np.savez(META, z_tr=z_tr, z_te=z_te, mu=mu, sd=sd, scale=scale,
             clip_lo=clip_lo, clip_hi=clip_hi, attractors=attractors,
             basin_eps=basin_eps, probes=probes, ep_probes=ep_probes,
             self_agree=self_agree, conv_frac=conv_frac, dec_ceiling=dec_ceiling,
             zeval=zeval, Vt_eval=Vt_eval)
    recon_te = float(np.array(mx.mean((model(mx.array(xte)) - mx.array(xte)) ** 2)))
    print(f"teacher: {len(attractors)} attractors, conv {conv_frac:.2f}, "
          f"ceiling {self_agree:.2f}, basin_eps {basin_eps:.3f}, scale {scale:.2f}, "
          f"test recon {recon_te:.4f}  [{time.time()-t0:.0f}s]")


# ---------- stage: one student arm ----------

def load_teacher():
    teacher = make_ae(0)
    teacher.load_weights(str(TEACHER_W))
    mx.eval(teacher.parameters())
    return teacher, np.load(META)


def stage_arm(arm, seed):
    t0 = time.time()
    xtr, ytr, xte, yte = load_mnist()
    teacher, meta = load_teacher()
    mu, sd = meta["mu"], meta["sd"]
    clip_lo, clip_hi = meta["clip_lo"], meta["clip_hi"]

    student = make_ae(100 + seed)
    opt = optim.Adam(learning_rate=optim.cosine_decay(LR, STUDENT_STEPS, 1e-4))
    X = mx.array(xtr)
    rng = np.random.default_rng(1000 + seed)
    n_anchor = N_TRAIN // 20
    Xa = X[:n_anchor]

    def sample_z(k):
        """Half broad prior; half deep teacher-trajectory points (near-attractor
        precision is what decides basins — concentrate samples there)."""
        half = k // 2
        broad = (mu + 2.0 * sd * rng.standard_normal((half, D_LAT))).astype(np.float32)
        core = mx.array((mu + 1.5 * sd * rng.standard_normal((k - half, D_LAT)))
                        .astype(np.float32))
        depth = int(rng.integers(2, 25))
        for _ in range(depth):
            core = teacher.enc(teacher.dec(core))
        return mx.concatenate([mx.array(broad), core], axis=0)

    if arm == "scratch":
        def loss_fn(m, xb, zb, vt):
            return mx.mean((m(xb) - xb) ** 2)
    elif arm == "outdistill":
        def loss_fn(m, xb, zb, vt):
            return mx.mean((m(xb) - teacher(xb)) ** 2)
    elif arm == "latmatch":
        def loss_fn(m, xb, zb, vt):
            return (mx.mean((m.enc(xb) - teacher.enc(xb)) ** 2)
                    + mx.mean((m(xb) - xb) ** 2))
    elif arm == "field":
        def loss_fn(m, xb, zb, aux):
            return mx.mean((field(m, zb) - aux[0]) ** 2)
    elif arm == "fieldplus":
        def loss_fn(m, xb, zb, aux):
            return (mx.mean((field(m, zb) - aux[0]) ** 2)
                    + 0.2 * mx.mean((m(Xa) - Xa) ** 2))
    elif arm == "fieldtraj":
        # 1-step field + k-step trajectory matching (supervises long-run flow)
        def loss_fn(m, xb, zb, aux):
            vt, targets = aux
            loss = mx.mean((field(m, zb) - vt) ** 2)
            z = zb
            i = 0
            for k in range(1, max(TRAJ_KS) + 1):
                z = m.enc(m.dec(z))
                if k in TRAJ_KS:
                    loss = loss + mx.mean((z - targets[i]) ** 2)
                    i += 1
            return loss / (1 + len(TRAJ_KS))
    else:
        raise SystemExit(f"unknown arm {arm}")

    lg = nn.value_and_grad(student, loss_fn)
    uses_z = arm in ("field", "fieldplus", "fieldtraj")
    for i in range(STUDENT_STEPS):
        idx = mx.array(rng.integers(0, N_TRAIN, BS))
        xb = X[idx]
        if uses_z:
            zb = sample_z(BS)
            vt = field(teacher, zb)
            targets = None
            if arm == "fieldtraj":
                targets = []
                zt = zb
                for k in range(1, max(TRAJ_KS) + 1):
                    zt = teacher.enc(teacher.dec(zt))
                    if k in TRAJ_KS:
                        targets.append(zt)
                mx.eval(targets)
            aux = (vt, targets)
            mx.eval(zb, vt)
        else:
            zb, aux = None, (None, None)
        _, g = lg(student, xb, zb, aux)
        g, _ = optim.clip_grad_norm(g, max_norm=5.0)
        opt.update(student, g)
        mx.eval(student.parameters(), opt.state)

    m = evaluate(student, teacher, meta, xtr, xte)
    m.update(arm=arm, seed=seed)
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else []
    results = [r for r in results if not (r["arm"] == arm and r["seed"] == seed)]
    results.append(m)
    RESULTS.write_text(json.dumps(results, indent=2))
    print(f"{arm} seed {seed}: basin {m['basin_agreement']:.2f}  cos {m['field_cos']:+.2f}  "
          f"nmse {m['field_nmse']:.3f}  chamfer/scale {m['chamfer_rel']:.3f}  "
          f"recon {m['recon_heldout']:.4f}  [{m['alignment']}] [{time.time()-t0:.0f}s]")

    if arm == "fieldplus" and seed == 1:  # money shot inputs
        save_attractor_decodes(student, teacher, meta)


def evaluate(student, teacher, meta, xtr, xte):
    zeval, Vt = meta["zeval"], meta["Vt_eval"]
    scale = float(meta["scale"])
    d = D_LAT
    ident = (np.eye(d, dtype=np.float32), np.zeros(d, dtype=np.float32),
             np.eye(d, dtype=np.float32))
    cands = [("identity", ident)]
    zs_data = np.array(student.enc(mx.array(xtr)))
    fitted = fit_affine(zs_data, meta["z_tr"])
    if fitted is not None:
        cands.append(("affine", fitted))

    def conj_field(al, zt):
        W, b, Winv = al
        zs = ((zt - b) @ Winv).astype(np.float32)
        fs = np.array(student.enc(student.dec(mx.array(zs))))
        return (fs @ W + b) - zt

    best, best_nmse, best_name = None, np.inf, None
    denom = float((Vt ** 2).mean())
    for name, al in cands:
        Vh = conj_field(al, zeval)
        if not np.all(np.isfinite(Vh)):
            continue
        nmse = float(((Vh - Vt) ** 2).mean()) / denom
        if nmse < best_nmse:
            best, best_nmse, best_name = al, nmse, name
    align = best or ident
    W, b, Winv = align

    Vh = conj_field(align, zeval)
    mag = np.linalg.norm(Vt, axis=1)
    mask = mag > np.percentile(mag, 10)
    cos = float((np.sum(Vh[mask] * Vt[mask], axis=1)
                 / (np.linalg.norm(Vh[mask], axis=1) * mag[mask] + 1e-12)).mean())

    # student-coord clip box via teacher box corners is overkill at 16D: use mapped stats
    zs_probes = ((meta["probes"] - b) @ Winv).astype(np.float32)
    s_mu, s_sd = zs_data.mean(0), zs_data.std(0)
    s_lo = (s_mu - 8 * np.maximum(s_sd, 1e-3)).astype(np.float32)
    s_hi = (s_mu + 8 * np.maximum(s_sd, 1e-3)).astype(np.float32)
    ep_s = np.array(iterate(student, mx.array(zs_probes), ITER_STEPS, s_lo, s_hi)) @ W + b
    dists = np.linalg.norm(ep_s - meta["ep_probes"], axis=1)
    basin_eps = float(meta["basin_eps"])
    basin = float((dists < basin_eps).mean())

    # decoded-endpoint agreement: compare endpoints through the TEACHER decoder
    # (the semantically meaningful observable — same decoded memory or not)
    img_s = np.array(teacher.dec(mx.array(ep_s.astype(np.float32))))
    img_t = np.array(teacher.dec(mx.array(meta["ep_probes"])))
    dec_agree = float((((img_s - img_t) ** 2).mean(axis=1) < 0.01).mean())

    starts_s = ((np.concatenate([meta["probes"], meta["zeval"][:256]]) - b) @ Winv).astype(np.float32)
    ep_g = np.array(iterate(student, mx.array(starts_s), ITER_STEPS, s_lo, s_hi))
    vn_s = np.linalg.norm(np.array(field(student, mx.array(ep_g))), axis=1)
    attr_s = find_attractors(ep_g, vn_s, vtol=0.005 * scale, merge_r=0.1 * scale)
    attr_s_t = attr_s @ W + b if len(attr_s) else attr_s
    ch = float(chamfer(meta["attractors"], attr_s_t))

    Xte = mx.array(xte)
    recon = float(np.array(mx.mean((student(Xte) - Xte) ** 2)))
    return dict(field_nmse=best_nmse, field_cos=cos, basin_agreement=basin,
                dec_agree=dec_agree, endpoint_median=float(np.median(dists)),
                chamfer_rel=ch / scale, n_attractors=int(len(attr_s_t)),
                recon_heldout=recon, alignment=best_name or "identity")


def save_attractor_decodes(student, teacher, meta):
    """Decode teacher attractors through both decoders (student attractors matched)."""
    attr_t = meta["attractors"][:12]
    imgs_t = np.array(teacher.dec(mx.array(attr_t.astype(np.float32))))
    zs_data = np.array(student.enc(mx.array(load_mnist()[0])))
    fitted = fit_affine(zs_data, meta["z_tr"])
    if fitted is None:
        return
    W, b, Winv = fitted
    imgs_s = np.array(student.dec(mx.array(((attr_t - b) @ Winv).astype(np.float32))))
    np.savez(RUNS / "mnist_attr_decodes.npz", imgs_t=imgs_t, imgs_s=imgs_s)
    print(f"saved attractor decodes for {len(attr_t)} attractors")


# ---------- stage: report ----------

ARM_ORDER = ["field", "fieldtraj", "fieldplus", "scratch", "outdistill", "latmatch"]


def stage_report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta = np.load(META)
    results = json.loads(RESULTS.read_text())
    agg = {}
    for arm in ARM_ORDER:
        rows = [r for r in results if r["arm"] == arm]
        if not rows:
            continue
        agg[arm] = {k: float(np.nanmean([r[k] for r in rows]))
                    for k in rows[0] if k not in ("arm", "seed", "alignment")}

    ceiling = float(meta["self_agree"])
    dec_ceiling = float(meta["dec_ceiling"]) if "dec_ceiling" in meta else float("nan")
    print(f"teacher: {len(meta['attractors'])} attractors, conv {float(meta['conv_frac']):.2f}, "
          f"ceiling {ceiling:.2f} (decoded {dec_ceiling:.2f}), "
          f"basin_eps {float(meta['basin_eps']):.3f}")
    print(f"{'arm':11s}{'basin':>7s}{'dec':>7s}{'cos':>7s}{'nmse':>8s}{'chamf/s':>9s}{'recon':>9s}")
    for arm, a in agg.items():
        print(f"{arm:11s}{a['basin_agreement']:7.2f}{a.get('dec_agree', float('nan')):7.2f}"
              f"{a['field_cos']:7.2f}{a['field_nmse']:8.3f}{a['chamfer_rel']:9.3f}"
              f"{a['recon_heldout']:9.4f}")

    ba = {a: agg[a]["basin_agreement"] for a in agg}
    static_best = max(ba.get("outdistill", 0), ba.get("latmatch", 0), ba.get("scratch", 0))
    bf = ba.get("field", 0.0)
    kill = static_best >= bf - 0.05
    go = (bf >= 0.6 * ceiling) and (bf - static_best >= 0.15) and (agg["field"]["field_cos"] >= 0.8)
    verdict = ("NO-GO (kill-switch: static baseline matches field arm)" if kill
               else ("GO" if go else "MARGINAL"))
    print(f"\nVERDICT (MNIST tier): {verdict}")
    print(f"  field basin {bf:.2f} (ceiling {ceiling:.2f}) vs best static {static_best:.2f}")
    out = dict(verdict=verdict, aggregate=agg, ceiling=ceiling,
               teacher_attractors=int(len(meta["attractors"])))
    (RUNS / "mnist_summary.json").write_text(json.dumps(out, indent=2))

    dec = RUNS / "mnist_attr_decodes.npz"
    if dec.exists():
        d = np.load(dec)
        n = len(d["imgs_t"])
        fig, axes = plt.subplots(2, n, figsize=(1.2 * n, 2.8))
        for j in range(n):
            axes[0, j].imshow(d["imgs_t"][j].reshape(28, 28), cmap="gray")
            axes[1, j].imshow(d["imgs_s"][j].reshape(28, 28), cmap="gray")
            for i in (0, 1):
                axes[i, j].axis("off")
        axes[0, 0].set_title("teacher attractors (decoded)", loc="left", fontsize=9)
        axes[1, 0].set_title("A+ student (5% anchor) decodes same attractors", loc="left", fontsize=9)
        fig.tight_layout()
        fig.savefig(RUNS / "mnist_attractors.png", dpi=140)
        print(f"wrote {RUNS/'mnist_attractors.png'}")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "teacher":
        noise = float(sys.argv[2]) if len(sys.argv) > 2 else TEACHER_NOISE
        steps = int(sys.argv[3]) if len(sys.argv) > 3 else STEPS
        stage_teacher(noise, steps)
    elif cmd == "arm":
        stage_arm(sys.argv[2], int(sys.argv[3]))
    elif cmd == "report":
        stage_report()
    else:
        raise SystemExit("usage: mnist teacher | arm <name> <seed> | report")


if __name__ == "__main__":
    main()
