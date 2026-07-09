"""Synthetic world: k Gaussian blobs on a ring in 2D, lifted nonlinearly to d_high dims.

The 2D structure gives the teacher AE (latent dim 2) a clean, plottable latent
vector field with point attractors near the blob prototypes. The nonlinear lift
makes the encode/decode job non-trivial (a linear AE cannot solve it), which is
what makes attractors form at all.

World is FIXED (seed 0) across all runs; only model inits vary by seed.
"""

import numpy as np

WORLD_SEED = 0


def make_world(n=2400, k=4, radius=2.0, std=0.35, d_high=16):
    rng = np.random.default_rng(WORLD_SEED)
    angles = np.arange(k) / k * 2 * np.pi
    centers = radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    labels = rng.integers(0, k, size=n)
    x2 = centers[labels] + std * rng.normal(size=(n, 2))

    # Fixed nonlinear lift 2D -> d_high (tanh feature map then linear mix).
    W1 = rng.normal(size=(2, d_high)) / np.sqrt(2)
    b1 = 0.1 * rng.normal(size=(d_high,))
    W2 = rng.normal(size=(d_high, d_high)) / np.sqrt(d_high)
    xh = np.tanh(x2 @ W1 + b1) @ W2

    return xh.astype(np.float32), x2.astype(np.float32), labels


def split(xh, n_train=2000):
    return xh[:n_train], xh[n_train:]
