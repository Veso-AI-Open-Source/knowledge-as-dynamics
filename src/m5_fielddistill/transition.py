"""Localize the copyability phase transition + tie it to Fumero's memorization signature.

Dense noise grid across the transition (found between 0.02 and 0.05 in sweep.py),
3 student seeds per level, plus two teacher-side observables:

  mem_attach : median over attractors of distance to NEAREST TRAINING ENCODING.
               ~0 means attractors sit on memorized training points (memorization
               signature per Fumero et al. sec. 5).
  proto_dist : median over attractors of distance to nearest BLOB-PROTOTYPE encoding
               (prototype = encoded per-class mean input). Small means attractors
               represent class prototypes (generalization regime).

If the copyability jump co-locates with mem_attach lifting off zero, the order
parameter is anchored to the memorization->generalization transition.

Run:  uv run python -m m5_fielddistill.transition     (~4 min)
Writes runs/transition.json + runs/transition.png
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

NOISES = [0.0, 0.01, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06, 0.08]
SEEDS = [1, 2, 3]


def attractor_signatures(prof, teacher, x_train, labels_train):
    """Distance from attractors to nearest training encoding / nearest prototype encoding."""
    attr = prof["attractors"]
    if len(attr) == 0:
        return float("nan"), float("nan")
    z_data = prof["z_data"]
    d_train = np.linalg.norm(attr[:, None, :] - z_data[None, :, :], axis=-1).min(axis=1)
    protos = np.stack([x_train[labels_train == c].mean(axis=0)
                       for c in np.unique(labels_train)])
    z_proto = np.array(teacher.enc(mx.array(protos.astype(np.float32))))
    d_proto = np.linalg.norm(attr[:, None, :] - z_proto[None, :, :], axis=-1).min(axis=1)
    return float(np.median(d_train)), float(np.median(d_proto))


def main():
    t0 = time.time()
    RUNS.mkdir(exist_ok=True)
    xh, x2, labels = make_world()
    x_train, x_test = split(xh)
    labels_train = labels[: len(x_train)]
    rows = []

    for noise in NOISES:
        rng = np.random.default_rng(7)
        teacher = train_teacher(x_train, noise=noise)
        prof = teacher_profile(teacher, x_train, rng)
        mem_attach, proto_dist = attractor_signatures(prof, teacher, x_train, labels_train)

        copies = []
        for seed in SEEDS:
            s = train_student("field", teacher, x_train, prof["lo"], prof["hi"], seed)
            m, _, _ = evaluate(s, prof, x_test, x_train)
            copies.append(m["basin_agreement"])

        row = dict(noise=noise, n_attractors=int(len(prof["attractors"])),
                   conv_frac=prof["conv_frac"], self_agree=prof["self_agree"],
                   mem_attach=mem_attach, proto_dist=proto_dist,
                   copy_seeds=[float(c) for c in copies],
                   copy_mean=float(np.mean(copies)), copy_std=float(np.std(copies)))
        rows.append(row)
        print(f"noise={noise:<6g} attr={row['n_attractors']:>3d} conv={row['conv_frac']:.2f} "
              f"attach={mem_attach:.3f} proto={proto_dist:.3f}  "
              f"copy={row['copy_mean']:.2f}±{row['copy_std']:.2f} {row['copy_seeds']}  "
              f"[{time.time()-t0:.0f}s]")

    (RUNS / "transition.json").write_text(json.dumps(rows, indent=2))

    # ---- figure ----
    ns = [r["noise"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    for r in rows:
        ax.scatter([r["noise"]] * len(r["copy_seeds"]), r["copy_seeds"],
                   color="#1f77b4", alpha=0.5, s=30, zorder=3)
    ax.plot(ns, [r["copy_mean"] for r in rows], "o-", color="#1f77b4",
            label="copyability (mean, 3 seeds)")
    ax.plot(ns, [r["conv_frac"] for r in rows], "^:", color="#2ca02c",
            label="teacher conv fraction")
    ax.set_xlabel("teacher denoising noise")
    ax.set_ylabel("fraction")
    ax.set_ylim(0, 1.05)
    ax.set_title("Transition localization (per-seed points shown)")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(ns, [r["mem_attach"] for r in rows], "o-", color="#d62728",
            label="attractor dist to nearest TRAIN encoding (memorization ~0)")
    ax.plot(ns, [r["proto_dist"] for r in rows], "s--", color="#9467bd",
            label="attractor dist to nearest PROTOTYPE encoding")
    ax.set_xlabel("teacher denoising noise")
    ax.set_ylabel("latent distance")
    ax.set_title("Fumero memorization signature: attractor detachment")
    ax.legend(fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(ns, [r["copy_mean"] for r in rows], "-", color="#1f77b4", alpha=0.4)
    ax2.set_ylabel("copyability", color="#1f77b4")
    ax2.set_ylim(0, 1.05)

    fig.suptitle("Copyability phase transition vs attractor detachment from training points",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(RUNS / "transition.png", dpi=140)
    print(f"\nwrote {RUNS/'transition.json'} and {RUNS/'transition.png'} "
          f"({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
