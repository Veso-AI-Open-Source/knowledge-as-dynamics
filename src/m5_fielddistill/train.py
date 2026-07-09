"""Teacher training + the six student arms.

Arms:
  field      A1  : match V_s(z) to V_t(z) on z ~ Uniform(box). DATA-FREE, one-step.
                   (Run 1 showed: transfers the field a.e. (cos 0.97) but per-step error
                   compounds over iteration -> basins drift. Kept as the key ablation.)
  fieldk     A2  : match k-step composites f_s^k(z) to f_t^k(z), k in KS. DATA-FREE.
                   The horizon-compounded fix: supervises long-run dynamics directly.
  fieldplus  A+  : fieldk loss + small recon anchor on 5% of data (fixes data-space chart).
  scratch    B0  : plain recon on data (control: is the field seed-stable for free?).
  outdistill B1  : classic output distillation, student(x) -> teacher(x) targets on data.
  latmatch   B2  : static feature matching, Enc_s(x) -> Enc_t(x) + recon. THE KILL-SWITCH:
                   if this transfers the off-manifold field too, field distillation adds nothing.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .ae import AE, field

ARMS = ["field", "fieldk", "fieldplus", "scratch", "outdistill", "latmatch"]
KS = (1, 2, 4, 8, 16)


def _make_model(seed, d_in):
    mx.random.seed(seed)
    model = AE(d_in=d_in)
    mx.eval(model.parameters())
    return model


def train_teacher(x_train, steps=4000, bs=256, lr=1e-3, seed=0, noise=0.1):
    """Denoising AE teacher. noise=0.1 puts it in the contractive/generalization
    regime (Fumero et al. sec. 5): crisp point attractors, conv_frac 1.0,
    self-agreement ~0.98. noise=0 gives the memorization regime (26 fragmented
    attractors + slow filaments) where endpoint metrics are ill-posed."""
    model = _make_model(seed, x_train.shape[1])
    opt = optim.Adam(learning_rate=lr)
    X = mx.array(x_train)
    rng = np.random.default_rng(seed)
    d = x_train.shape[1]

    def loss_fn(m, xb, xn):
        return mx.mean((m(xn) - xb) ** 2)

    lg = nn.value_and_grad(model, loss_fn)
    for _ in range(steps):
        idx = mx.array(rng.integers(0, len(x_train), bs))
        xb = X[idx]
        xn = xb + noise * mx.array(rng.standard_normal((bs, d)).astype(np.float32))
        _, grads = lg(model, xb, xn)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
    return model


def train_student(arm, teacher, x_train, box_lo, box_hi, seed, steps=4000, bs=256, lr=1e-3):
    assert arm in ARMS
    model = _make_model(seed, x_train.shape[1])
    opt = optim.Adam(learning_rate=lr)
    X = mx.array(x_train)
    rng = np.random.default_rng(1000 + seed)
    lo = np.asarray(box_lo, dtype=np.float32)
    hi = np.asarray(box_hi, dtype=np.float32)
    n_anchor = max(1, len(x_train) // 20)  # 5% anchor for fieldplus
    Xa = X[: n_anchor]

    def sample_z(k):
        u = rng.random((k, lo.shape[0])).astype(np.float32)
        return mx.array(lo + u * (hi - lo))

    def teacher_traj(zb):
        """Teacher iterates f_t^k(zb) for k in KS (no grads; teacher is not in model params)."""
        targets = []
        z = zb
        for k in range(1, max(KS) + 1):
            z = teacher.enc(teacher.dec(z))
            if k in KS:
                targets.append(z)
        mx.eval(targets)
        return targets

    def multistep_loss(m, zb, targets):
        z = zb
        loss = 0.0
        i = 0
        for k in range(1, max(KS) + 1):
            z = m.enc(m.dec(z))
            if k in KS:
                loss = loss + mx.mean((z - targets[i]) ** 2)
                i += 1
        return loss / len(KS)

    if arm == "scratch":
        def loss_fn(m, xb, _zb, _aux):
            return mx.mean((m(xb) - xb) ** 2)
    elif arm == "outdistill":
        def loss_fn(m, xb, _zb, _aux):
            return mx.mean((m(xb) - teacher(xb)) ** 2)
    elif arm == "latmatch":
        def loss_fn(m, xb, _zb, _aux):
            return (mx.mean((m.enc(xb) - teacher.enc(xb)) ** 2)
                    + mx.mean((m(xb) - xb) ** 2))
    elif arm == "field":
        def loss_fn(m, _xb, zb, aux):
            return mx.mean((field(m, zb) - aux) ** 2)
    elif arm == "fieldk":
        def loss_fn(m, _xb, zb, aux):
            return multistep_loss(m, zb, aux)
    elif arm == "fieldplus":
        def loss_fn(m, _xb, zb, aux):
            return multistep_loss(m, zb, aux) + 0.2 * mx.mean((m(Xa) - Xa) ** 2)

    lg = nn.value_and_grad(model, loss_fn)
    uses_z = arm in ("field", "fieldk", "fieldplus")
    for _ in range(steps):
        idx = mx.array(rng.integers(0, len(x_train), bs))
        xb = X[idx]
        if uses_z:
            zb = sample_z(bs)
            if arm == "field":
                aux = field(teacher, zb)
                mx.eval(aux)
            else:
                aux = teacher_traj(zb)
        else:
            zb, aux = None, None
        _, grads = lg(model, xb, zb, aux)
        grads, _ = optim.clip_grad_norm(grads, max_norm=1.0)  # unrolled arms can spike early
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
    return model
