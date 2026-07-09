"""Autoencoder, latent vector field, attractor finding, and latent-space alignment.

Definitions follow Fumero et al. (arXiv 2505.22785):
    f(z)  = Enc(Dec(z))          the composite map on latent space
    V(z)  = f(z) - z             the latent vector field / displacement field
    attractors = fixed points reached by iterating f from a grid of inits
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class MLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = [nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])]

    def __call__(self, x):
        n = len(self.layers)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < n - 1:
                x = nn.gelu(x)
        return x


class AE(nn.Module):
    def __init__(self, d_in=16, d_lat=2, width=64, enc_dims=None, dec_dims=None):
        super().__init__()
        self.enc = MLP(enc_dims or [d_in, width, width, d_lat])
        self.dec = MLP(dec_dims or [d_lat, width, width, d_in])

    def __call__(self, x):
        return self.dec(self.enc(x))


def field(model, z):
    """V(z) = Enc(Dec(z)) - z"""
    return model.enc(model.dec(z)) - z


def iterate(model, z, steps, clip_lo, clip_hi):
    """Iterate z <- f(z), clipped to a sanity box so divergent students don't inf out."""
    lo = mx.array(clip_lo)
    hi = mx.array(clip_hi)
    for i in range(steps):
        z = model.enc(model.dec(z))
        z = mx.maximum(mx.minimum(z, hi), lo)
        if i % 20 == 0:
            mx.eval(z)
    mx.eval(z)
    return z


def find_attractors(endpoints, vnorms, vtol=5e-3, merge_r=0.15):
    """Dedupe converged endpoints into attractor centers (greedy radius merge)."""
    pts = endpoints[vnorms < vtol]
    centers: list[np.ndarray] = []
    for p in pts:
        if all(np.linalg.norm(p - c) > merge_r for c in centers):
            centers.append(p)
    return np.array(centers) if centers else np.zeros((0, endpoints.shape[1]))


def chamfer(a, b):
    """Symmetric mean nearest-neighbour distance between two point sets."""
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return 0.5 * (d.min(axis=1).mean() + d.min(axis=0).mean())


def fit_affine(z_src, z_dst):
    """Least-squares affine map z_dst ~= z_src @ W + b. Returns (W, b) or None if singular."""
    A = np.concatenate([z_src, np.ones((len(z_src), 1))], axis=1)
    M, *_ = np.linalg.lstsq(A, z_dst, rcond=None)
    W, b = M[:-1], M[-1]
    try:
        Winv = np.linalg.inv(W)
    except np.linalg.LinAlgError:
        return None
    if not (np.all(np.isfinite(W)) and np.all(np.isfinite(Winv))):
        return None
    return W.astype(np.float32), b.astype(np.float32), Winv.astype(np.float32)


IDENTITY = (np.eye(2, dtype=np.float32), np.zeros(2, dtype=np.float32), np.eye(2, dtype=np.float32))


def conjugated_field(student, grid_t, align):
    """Student's field expressed in TEACHER latent coordinates.

    align = (W, b, Winv) mapping student coords -> teacher coords: A(z) = z @ W + b.
    V_hat(z) = A(f_s(A^-1(z))) - z
    """
    W, b, Winv = align
    zs = (grid_t - b) @ Winv                      # teacher coords -> student coords
    zs_m = mx.array(zs.astype(np.float32))
    fs = np.array(student.enc(student.dec(zs_m)))  # f_s in student coords
    return (fs @ W + b) - grid_t                   # back to teacher coords, displacement
