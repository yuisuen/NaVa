"""Validity constraint, the closed-form feasible set, and the perturbation (Sections 3.3-3.4).

For a series ``x_i`` with reference class ``y_i`` (its label if labeled, otherwise the
prediction) and the ``min{M, K-1}`` most probable other classes ``c``, the class margin
``m_ic = f_{y_i}(x_i) - f_c(x_i)`` and ``g_ic = grad_x m_ic / max{m_ic, m0}`` give the
validity constraint ``V_i = { delta : sum_c (g_ic^T delta)^2 <= gamma^2 }``.

With ``G~_i = C_i^{1/2} G_i`` and the eigendecomposition of the small matrix
``G~_i^T G~_i``, the maximum-volume ellipsoid inside ``N_i`` and ``V_i`` is
``{ L_i u : ||u|| <= 1 }`` with

    L_i = C_i^{1/2} [ eps I + Q_i (diag(a_i) - eps I) Q_i^T ],   a_ij = min{eps, gamma / sigma_ij},

i.e. it is shrunk only along the (at most |K_i|) axes ``Q_i`` that cross the class
boundary. Unlabeled series whose largest predicted probability is below ``tau`` keep
``L_i = eps C_i^{1/2}`` (no validity constraint). The perturbation is the result of one
power iteration inside the unit ball, as in VAT.
"""

import contextlib

import torch
import torch.nn.functional as F

__all__ = ["bn_batch_view", "feasible_set", "worst_perturbation", "kl_consistency"]


@contextlib.contextmanager
def bn_batch_view(model):
    """Dropout off; BatchNorm normalises with the statistics of the current batch without
    updating its running buffers. Used for every forward of the consistency term."""
    was = model.training
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    model.eval()
    saved = [(m, m.momentum) for m in bns]
    for m in bns:
        m.momentum = 0.0
        m.train()
    try:
        yield
    finally:
        for m, mom in saved:
            m.momentum = mom
        if was:
            model.train()


@contextlib.contextmanager
def _bn_detached_stats(model):
    """BatchNorm with the batch mean / variance treated as constants, so that the input
    gradient of each series does not flow through the other series of the batch."""
    patched = []
    for mod in model.modules():
        if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
            def fwd(x, _m=mod):
                dims = [0] + list(range(2, x.dim()))
                return F.batch_norm(x, x.mean(dim=dims).detach(), x.var(dim=dims, unbiased=False).detach(),
                                    _m.weight, _m.bias, False, 0.0, _m.eps)
            mod.forward = fwd
            patched.append(mod)
    try:
        yield
    finally:
        for mod in patched:
            del mod.forward


def feasible_set(model, x, idx, p_clean, is_lab, y_lab, nat, gamma, M, tau, m0):
    """Returns ``to_input``: unit-ball coordinates (B, D) -> perturbations (B, T, C)."""
    mul = nat.mul(idx)
    eps = nat.eps
    xr = x.detach().requires_grad_(True)
    with _bn_detached_stats(model):
        z = model(xr)
    ref = torch.where(is_lab, y_lab, p_clean.argmax(1))            # reference class
    others = z.detach().clone()
    others.scatter_(1, ref[:, None], float("-inf"))
    kk = max(1, min(M, z.shape[1] - 1))
    comp = others.topk(kk, dim=1).indices                          # competing classes K_i
    marg = z.gather(1, ref[:, None]) - z.gather(1, comp)           # class margins (B, kk)
    gs = []
    for j in range(kk):
        gj = torch.autograd.grad(marg[:, j].sum(), xr, retain_graph=j < kk - 1)[0].flatten(1)
        gs.append(gj / marg[:, j].detach().clamp_min(m0)[:, None])
    g_t = mul(torch.stack(gs, 2)) / eps                            # G~_i = C_i^{1/2} G_i  (B, D, kk)
    ev, w = torch.linalg.eigh((g_t.transpose(1, 2) @ g_t).double().cpu())
    ev, w = ev.to(g_t.device, g_t.dtype), w.to(g_t.device, g_t.dtype)
    sig = ev.clamp_min(0.0).sqrt()
    q = (g_t @ w) / sig.clamp_min(1e-12)[:, None, :]              # axes crossing the boundary (B, D, kk)
    q = q * (sig > 1e-6 * sig.amax(dim=1, keepdim=True).clamp_min(1e-30))[:, None, :].to(q.dtype)
    a = (gamma / sig.clamp_min(1e-12)).clamp(max=eps) / eps        # a_ij / eps in (0, 1]
    if tau > 0:
        trusted = is_lab | (p_clean.max(1).values >= tau)
        a = torch.where(trusted[:, None], a, torch.ones_like(a))
    q, a = q.detach(), a.detach()
    shape = x.shape[1:]

    def to_input(u):
        qu = torch.einsum("bdk,bd->bk", q, u)
        w_ = u - torch.einsum("bdk,bk->bd", q, qu) + torch.einsum("bdk,bk->bd", q, a * qu)
        return mul(w_).view(-1, *shape)
    return to_input


def kl_consistency(p_clean, logits):
    """KL( p(x) || p(x + delta) ), averaged over the batch."""
    return (p_clean * (p_clean.clamp_min(1e-12).log() - F.log_softmax(logits, dim=1))).sum(dim=1).mean()


def worst_perturbation(model, x, p_clean, to_input, dim, xi):
    """One power iteration from a random direction of the unit ball (VAT)."""
    u = torch.randn(x.shape[0], dim, device=x.device, dtype=x.dtype)
    u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-12)
    probe = (xi * u).detach().requires_grad_(True)
    g = torch.nan_to_num(torch.autograd.grad(kl_consistency(p_clean, model(x + to_input(probe))), probe)[0])
    gn = g.norm(dim=1, keepdim=True)
    u = torch.where(gn > 1e-12, g / gn.clamp_min(1e-12), u).detach()
    return to_input(u)
