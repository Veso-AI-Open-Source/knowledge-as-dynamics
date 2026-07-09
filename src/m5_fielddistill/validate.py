"""GO/NO-GO harness for vector-field distillation of the Enc∘Dec latent dynamics.

Run:  uv run python -m m5_fielddistill.validate
Then: uv run python -m m5_fielddistill.plot

Protocol:
  1. Train teacher AE on the blob world; characterize its latent field V_t,
     attractors, and basin map on a grid box around the data encodings.
  2. Train 5 student arms x 3 seeds (see train.ARMS).
  3. Every arm gets its BEST-CASE latent alignment to teacher coords:
     best of {identity, fitted-affine-on-data-encodings} by grid field NMSE.
     (Affine alignment is our operationalization of the "alignment of latent
     vector fields" open question in Fumero et al. sec. 6.)
  4. Metrics per arm: field NMSE + cosine, attractor Chamfer, basin agreement,
     held-out recon MSE.

GO      : field arm basin agreement >= 0.70 AND beats best static baseline by
          >= 0.20 AND field cosine >= 0.80.
NO-GO   : any static/behavioral baseline (outdistill/latmatch) reaches within
          0.05 basin agreement of the field arm — the field would then be free.
"""

import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from .ae import (IDENTITY, chamfer, conjugated_field, field, find_attractors,
                 fit_affine, iterate)
from .train import ARMS, train_student, train_teacher
from .world import make_world, split

RUNS = Path(__file__).resolve().parents[2] / "runs"
SEEDS = [1, 2, 3]
GRID_N = 40
N_PROBES = 400
ITER_STEPS = 1000
BASIN_EPS = 0.3


def grid_box(z_data, margin=0.5):
    lo, hi = z_data.min(0), z_data.max(0)
    span = hi - lo
    return (lo - margin * span).astype(np.float32), (hi + margin * span).astype(np.float32)


def make_grid(lo, hi, n=GRID_N):
    xs = np.linspace(lo[0], hi[0], n, dtype=np.float32)
    ys = np.linspace(lo[1], hi[1], n, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], axis=1)


def teacher_profile(teacher, x_train, rng):
    z_data = np.array(teacher.enc(mx.array(x_train)))
    lo, hi = grid_box(z_data)
    clip_lo, clip_hi = lo - 2 * (hi - lo), hi + 2 * (hi - lo)
    G = make_grid(lo, hi)
    Vt = np.array(field(teacher, mx.array(G)))
    ep = np.array(iterate(teacher, mx.array(G), ITER_STEPS, clip_lo, clip_hi))
    vn = np.linalg.norm(np.array(field(teacher, mx.array(ep))), axis=1)
    attractors = find_attractors(ep, vn)
    probes = (lo + rng.random((N_PROBES, 2)).astype(np.float32) * (hi - lo))
    ep_probes = np.array(iterate(teacher, mx.array(probes), ITER_STEPS, clip_lo, clip_hi))
    # self-agreement ceiling: teacher vs itself under probe perturbation 0.05
    ep_pert = np.array(iterate(teacher, mx.array(
        probes + 0.05 * rng.standard_normal(probes.shape).astype(np.float32)),
        ITER_STEPS, clip_lo, clip_hi))
    self_agree = float((np.linalg.norm(ep_pert - ep_probes, axis=1) < BASIN_EPS).mean())
    conv_frac = float((vn < 5e-3).mean())
    return dict(z_data=z_data, lo=lo, hi=hi, clip_lo=clip_lo, clip_hi=clip_hi,
                G=G, Vt=Vt, attractors=attractors, probes=probes, ep_probes=ep_probes,
                self_agree=self_agree, conv_frac=conv_frac)


def pick_alignment(student, prof, x_train):
    """Best-case alignment: identity vs fitted affine, judged by grid field NMSE."""
    cands = [("identity", IDENTITY)]
    zs = np.array(student.enc(mx.array(x_train)))
    fitted = fit_affine(zs, prof["z_data"])
    if fitted is not None:
        cands.append(("affine", fitted))
    best, best_nmse, best_name = None, np.inf, None
    denom = float((prof["Vt"] ** 2).mean())
    for name, al in cands:
        Vh = conjugated_field(student, prof["G"], al)
        if not np.all(np.isfinite(Vh)):
            continue
        nmse = float(((Vh - prof["Vt"]) ** 2).mean()) / denom
        if nmse < best_nmse:
            best, best_nmse, best_name = al, nmse, name
    return (best or IDENTITY), best_name or "identity"


def evaluate(student, prof, x_test, x_train):
    align, align_name = pick_alignment(student, prof, x_train)
    W, b, Winv = align
    G, Vt = prof["G"], prof["Vt"]

    Vh = conjugated_field(student, G, align)
    nmse = float(((Vh - Vt) ** 2).mean() / (Vt ** 2).mean())
    # cosine over grid points where the teacher field is meaningfully nonzero
    mag = np.linalg.norm(Vt, axis=1)
    mask = mag > np.percentile(mag, 10)
    cos = float((np.sum(Vh[mask] * Vt[mask], axis=1)
                 / (np.linalg.norm(Vh[mask], axis=1) * mag[mask] + 1e-12)).mean())

    # student-space clip box: map teacher clip box corners through Winv (use bbox)
    corners_t = np.array([[prof["clip_lo"][0], prof["clip_lo"][1]],
                          [prof["clip_lo"][0], prof["clip_hi"][1]],
                          [prof["clip_hi"][0], prof["clip_lo"][1]],
                          [prof["clip_hi"][0], prof["clip_hi"][1]]], dtype=np.float32)
    corners_s = (corners_t - b) @ Winv
    s_lo, s_hi = corners_s.min(0), corners_s.max(0)

    # basin agreement: iterate probes in student coords, compare endpoints in teacher coords
    probes_s = (prof["probes"] - b) @ Winv
    ep_s = np.array(iterate(student, mx.array(probes_s.astype(np.float32)),
                            ITER_STEPS, s_lo, s_hi)) @ W + b
    dists = np.linalg.norm(ep_s - prof["ep_probes"], axis=1)
    basin = float((dists < BASIN_EPS).mean())
    ep_med = float(np.median(dists))

    # attractors: iterate from grid (student coords), converged by student field norm
    G_s = ((G - b) @ Winv).astype(np.float32)
    ep_g = np.array(iterate(student, mx.array(G_s), ITER_STEPS, s_lo, s_hi))
    vn_s = np.linalg.norm(np.array(field(student, mx.array(ep_g))), axis=1)
    attr_s = find_attractors(ep_g, vn_s)
    attr_s_t = attr_s @ W + b if len(attr_s) else attr_s
    ch = float(chamfer(prof["attractors"], attr_s_t))

    Xte = mx.array(x_test)
    recon = float(np.array(mx.mean((student(Xte) - Xte) ** 2)))

    return dict(field_nmse=nmse, field_cos=cos, basin_agreement=basin,
                endpoint_median=ep_med, attractor_chamfer=ch,
                n_attractors=int(len(attr_s_t)), recon_heldout=recon,
                alignment=align_name), Vh, attr_s_t


def main():
    t0 = time.time()
    RUNS.mkdir(exist_ok=True)
    rng = np.random.default_rng(7)

    xh, x2, labels = make_world()
    x_train, x_test = split(xh)

    print("training teacher ...")
    teacher = train_teacher(x_train)
    prof = teacher_profile(teacher, x_train, rng)
    print(f"  teacher: {len(prof['attractors'])} attractors, conv_frac "
          f"{prof['conv_frac']:.2f}, self-agreement ceiling {prof['self_agree']:.2f}, "
          f"box {prof['lo'].round(2)} .. {prof['hi'].round(2)}")

    results: dict[str, list] = {a: [] for a in ARMS}
    fields_for_plot = {"G": prof["G"], "Vt": prof["Vt"], "attractors_t": prof["attractors"],
                       "z_data": prof["z_data"], "labels": labels[:len(x_train)]}

    for arm in ARMS:
        for seed in SEEDS:
            s = train_student(arm, teacher, x_train, prof["lo"], prof["hi"], seed)
            m, Vh, attr = evaluate(s, prof, x_test, x_train)
            results[arm].append(m)
            print(f"  {arm:10s} seed {seed}: basin {m['basin_agreement']:.2f}  "
                  f"cos {m['field_cos']:+.2f}  nmse {m['field_nmse']:.3f}  "
                  f"chamfer {m['attractor_chamfer']:.3f}  recon {m['recon_heldout']:.4f}  "
                  f"[{m['alignment']}]")
            if seed == SEEDS[0]:
                fields_for_plot[f"Vh_{arm}"] = Vh
                fields_for_plot[f"attr_{arm}"] = attr
                fields_for_plot[f"basin_{arm}"] = np.array(m["basin_agreement"])

    # aggregate
    agg = {}
    for arm in ARMS:
        agg[arm] = {k: dict(mean=float(np.nanmean([r[k] for r in results[arm]])),
                            std=float(np.nanstd([r[k] for r in results[arm]])))
                    for k in results[arm][0] if k != "alignment"}

    # verdict: judged on the best DATA-FREE field arm (field = 1-step, fieldk = multi-step)
    ba = {a: agg[a]["basin_agreement"]["mean"] for a in ARMS}
    best_df_arm = max(["field", "fieldk"], key=lambda a: ba[a])
    ba_df = ba[best_df_arm]
    cos_df = agg[best_df_arm]["field_cos"]["mean"]
    static_best = max(ba["outdistill"], ba["latmatch"], ba["scratch"])
    kill = static_best >= ba_df - 0.05
    go = (ba_df >= 0.70) and (ba_df - static_best >= 0.20) and (cos_df >= 0.80)
    verdict = "NO-GO (kill-switch: static baseline matches the field arm)" if kill \
        else ("GO" if go else "MARGINAL (field arm did not clear thresholds)")

    print("\n=== SUMMARY (mean over seeds) ===")
    hdr = f"{'arm':11s}{'basin':>7s}{'cos':>7s}{'nmse':>8s}{'chamfer':>9s}{'recon':>9s}"
    print(hdr)
    for arm in ARMS:
        a = agg[arm]
        print(f"{arm:11s}{a['basin_agreement']['mean']:7.2f}{a['field_cos']['mean']:7.2f}"
              f"{a['field_nmse']['mean']:8.3f}{a['attractor_chamfer']['mean']:9.3f}"
              f"{a['recon_heldout']['mean']:9.4f}")
    print(f"\nVERDICT: {verdict}")
    print(f"  best data-free arm '{best_df_arm}': basin {ba_df:.2f} vs best static "
          f"{static_best:.2f}; field cosine {cos_df:.2f}")
    print(f"total {time.time()-t0:.0f}s")

    out = dict(verdict=verdict, aggregate=agg, per_seed=results,
               teacher=dict(n_attractors=int(len(prof["attractors"])),
                            attractors=prof["attractors"].tolist(),
                            self_agree=prof["self_agree"], conv_frac=prof["conv_frac"]),
               config=dict(seeds=SEEDS, grid_n=GRID_N, n_probes=N_PROBES,
                           iter_steps=ITER_STEPS, basin_eps=BASIN_EPS))
    (RUNS / "results.json").write_text(json.dumps(out, indent=2))
    np.savez(RUNS / "fields.npz", **fields_for_plot)
    print(f"wrote {RUNS/'results.json'} and {RUNS/'fields.npz'}")


if __name__ == "__main__":
    main()
