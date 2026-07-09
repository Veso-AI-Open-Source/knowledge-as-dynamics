"""The 'law' test: is dynamics-copyability a function of teacher generalization?

Sweeps teacher denoising noise 0 -> 0.2 (memorization -> generalization regime),
field-distills a data-free student at each level, and charts:
  copyability (basin agreement, raw + ceiling-normalized) vs teacher regime.

Teacher regime measured three ways: attractor count, off-manifold robustness
(recon error on noise-perturbed held-out inputs), and generalization gap.

Run:  uv run python -m m5_fielddistill.sweep     (~10 min, 2 seeds/level)
Writes runs/sweep.json + runs/sweep.png
"""

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np

from .train import train_student, train_teacher
from .validate import RUNS, evaluate, teacher_profile
from .world import make_world, split

NOISES = [0.0, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2]
SEEDS = [1, 2]
PERTURB = 0.1  # input perturbation for off-manifold robustness


def main():
    t0 = time.time()
    RUNS.mkdir(exist_ok=True)
    xh, x2, labels = make_world()
    x_train, x_test = split(xh)
    Xtr, Xte = mx.array(x_train), mx.array(x_test)
    rows = []

    for noise in NOISES:
        rng = np.random.default_rng(7)  # same probes/grid rng per level
        teacher = train_teacher(x_train, noise=noise)
        prof = teacher_profile(teacher, x_train, rng)

        recon_tr = float(np.array(mx.mean((teacher(Xtr) - Xtr) ** 2)))
        recon_te = float(np.array(mx.mean((teacher(Xte) - Xte) ** 2)))
        pert = mx.array(PERTURB * rng.standard_normal(x_test.shape).astype(np.float32))
        robust = float(np.array(mx.mean((teacher(Xte + pert) - Xte) ** 2)))

        basins, coss, chs = [], [], []
        for seed in SEEDS:
            s = train_student("field", teacher, x_train, prof["lo"], prof["hi"], seed)
            m, _, _ = evaluate(s, prof, x_test, x_train)
            basins.append(m["basin_agreement"])
            coss.append(m["field_cos"])
            chs.append(m["attractor_chamfer"])

        row = dict(noise=noise,
                   n_attractors=int(len(prof["attractors"])),
                   conv_frac=prof["conv_frac"], self_agree=prof["self_agree"],
                   recon_train=recon_tr, recon_test=recon_te,
                   gen_gap=recon_te - recon_tr, robustness=robust,
                   basin=float(np.mean(basins)), basin_std=float(np.std(basins)),
                   basin_norm=float(np.mean(basins) / max(prof["self_agree"], 1e-6)),
                   field_cos=float(np.mean(coss)),
                   chamfer=float(np.nanmean(chs)))
        rows.append(row)
        print(f"noise={noise:<5g} attr={row['n_attractors']:>3d} conv={row['conv_frac']:.2f} "
              f"ceil={row['self_agree']:.2f}  copy={row['basin']:.2f}±{row['basin_std']:.2f} "
              f"(norm {row['basin_norm']:.2f})  robust={row['robustness']:.4f} "
              f"[{time.time()-t0:.0f}s]")

    (RUNS / "sweep.json").write_text(json.dumps(rows, indent=2))

    # ---- figure ----
    ns = [r["noise"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    ax = axes[0]
    ax.plot(ns, [r["basin"] for r in rows], "o-", color="#1f77b4", label="copyability (basin agree)")
    ax.fill_between(ns, [r["basin"] - r["basin_std"] for r in rows],
                    [r["basin"] + r["basin_std"] for r in rows], alpha=0.2, color="#1f77b4")
    ax.plot(ns, [r["self_agree"] for r in rows], "s--", color="gray", label="teacher self-agreement (ceiling)")
    ax.plot(ns, [r["conv_frac"] for r in rows], "^:", color="#2ca02c", label="teacher conv fraction")
    ax.set_xlabel("teacher denoising noise (memorization → generalization)")
    ax.set_ylabel("fraction")
    ax.set_ylim(0, 1.05)
    ax.set_title("Copyability of dynamics vs teacher regime")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(ns, [r["n_attractors"] for r in rows], "o-", color="#d62728")
    ax.set_xlabel("teacher denoising noise")
    ax.set_ylabel("attractor count")
    ax.set_title("Landscape consolidation")
    ax2 = ax.twinx()
    ax2.plot(ns, [r["robustness"] for r in rows], "s--", color="#9467bd")
    ax2.set_ylabel("noisy-input recon error (robustness, lower=better)", color="#9467bd")

    ax = axes[2]
    x = [r["robustness"] for r in rows]
    y = [r["basin_norm"] for r in rows]
    ax.scatter(x, y, c=ns, cmap="viridis", s=80, zorder=3)
    for r in rows:
        ax.annotate(f"{r['noise']:g}", (r["robustness"], r["basin_norm"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("teacher off-manifold error (noisy-input recon MSE)")
    ax.set_ylabel("normalized copyability (basin / ceiling)")
    ax.set_title("THE LAW PLOT: copyability vs generalization")
    ax.invert_xaxis()  # right = better generalization

    fig.suptitle("Is dynamics-copyability a measure of generalization? "
                 "(data-free field distillation, 2 seeds/level)", fontsize=13)
    fig.tight_layout()
    fig.savefig(RUNS / "sweep.png", dpi=140)
    print(f"\nwrote {RUNS/'sweep.json'} and {RUNS/'sweep.png'}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
