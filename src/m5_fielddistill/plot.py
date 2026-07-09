"""Quiver-panel figure: teacher field vs the five student arms (seed 1).

Run after validate:  uv run python -m m5_fielddistill.plot
Writes runs/field_panels.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = Path(__file__).resolve().parents[2] / "runs"
ARM_TITLES = {
    "field": "A1 field distill, 1-step (data-free)",
    "fieldk": "A2 field distill, k-step (data-free)",
    "fieldplus": "A+ k-step + 5% data anchor",
    "scratch": "B0 scratch on data",
    "outdistill": "B1 output distillation",
    "latmatch": "B2 latent matching (kill-switch)",
}


def _quiver(ax, G, V, color, sub=2):
    n = int(np.sqrt(len(G)))
    gx = G[:, 0].reshape(n, n)[::sub, ::sub]
    gy = G[:, 1].reshape(n, n)[::sub, ::sub]
    vx = V[:, 0].reshape(n, n)[::sub, ::sub]
    vy = V[:, 1].reshape(n, n)[::sub, ::sub]
    mag = np.sqrt(vx**2 + vy**2) + 1e-12
    ax.quiver(gx, gy, vx / mag, vy / mag, mag, cmap=color,
              scale=40, width=0.004, alpha=0.9)


def main():
    d = np.load(RUNS / "fields.npz")
    G, Vt, attr_t = d["G"], d["Vt"], d["attractors_t"]

    fig, axes = plt.subplots(2, 4, figsize=(21, 10))
    panels = [("teacher", None)] + [(a, ARM_TITLES[a]) for a in ARM_TITLES]
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")

    for ax, (arm, title) in zip(axes.ravel(), panels):
        if arm == "teacher":
            _quiver(ax, G, Vt, "viridis")
            ax.scatter(d["z_data"][:, 0], d["z_data"][:, 1], s=2, c="lightgray",
                       alpha=0.4, zorder=1, label="data encodings")
            ax.set_title("TEACHER latent field V(z) = Enc(Dec(z)) − z")
        else:
            _quiver(ax, G, d[f"Vh_{arm}"], "magma")
            attr_s = d[f"attr_{arm}"]
            if len(attr_s):
                ax.scatter(attr_s[:, 0], attr_s[:, 1], marker="o", s=90,
                           facecolors="none", edgecolors="red", linewidths=2,
                           zorder=6, label="student attractors")
            ax.set_title(f"{title} — basin {float(d[f'basin_{arm}']):.2f}")
        ax.scatter(attr_t[:, 0], attr_t[:, 1], marker="*", s=180, c="black",
                   zorder=5, label="teacher attractors")
        ax.set_xlim(G[:, 0].min(), G[:, 0].max())
        ax.set_ylim(G[:, 1].min(), G[:, 1].max())
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=7)

    fig.suptitle("Vector-field distillation go/no-go — student fields in teacher coordinates (seed 1)",
                 fontsize=14)
    fig.tight_layout()
    out = RUNS / "field_panels.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
