"""Naturalness constraint: local variations of real series (Section 3.2 of the paper).

PCA on all training series gives the subspace ``B`` (D x p). For each series ``x_i``, the
displacements to its ``k`` nearest neighbours in the PCA coordinate give the local
variation ``C_hat_i = V_i diag(rho_i) V_i^T`` (largest eigenvalue normalised to 1), and

    C_i = I_D + alpha * B C_hat_i B^T,      N_i = { delta : delta^T C_i^{-1} delta <= eps^2 }.

Only ``V_i`` (p x r) and ``c_i = sqrt(1 + alpha * rho_i) - 1`` are stored, so that

    C_i^{1/2} w = w + B V_i (c_i * (V_i^T B^T w))

is applied without forming any D x D matrix. Everything here is computed once before
training and does not use labels.
"""

import torch

__all__ = ["NaturalFactors", "build_naturalness"]


def fit_pca(flat, p, min_dim=8):
    """Mean, orthonormal basis (D, m) and variances of the leading principal components;
    m = min(p, D, N - 1, rank) and at least min(min_dim, ...)."""
    n, d = flat.shape
    mu = flat.mean(dim=0)
    _, s, vh = torch.linalg.svd(flat - mu, full_matrices=False)
    lam = s ** 2 / max(n - 1, 1)
    ratio = torch.cumsum(lam / lam.sum().clamp_min(1e-300), dim=0)
    rank = int((s > s[0] * 1e-12).sum()) if s.numel() and s[0] > 0 else 1
    lo, hi = min(min_dim, d, max(n - 1, 1), rank), min(p, d, max(n - 1, 1), rank)
    m = int(torch.searchsorted(ratio, torch.tensor(1.0, dtype=ratio.dtype)).item()) + 1
    m = max(lo, min(m, hi))
    return mu, vh[:m].T.contiguous()


def _svd(d):
    """Thin SVD (U, S) of a batch; rows on which LAPACK does not converge fall back to the
    eigendecomposition of d d^T, which spans the same ellipsoid."""
    try:
        u, s, _ = torch.linalg.svd(d, full_matrices=False)
        return u, s
    except torch.linalg.LinAlgError:
        pass
    r = min(d.shape[-2], d.shape[-1])
    us, ss = [], []
    for di in d:
        try:
            ui, si, _ = torch.linalg.svd(di, full_matrices=False)
        except torch.linalg.LinAlgError:
            ev, ui = torch.linalg.eigh(di @ di.T)
            ev, ui = ev.flip(-1)[:r].clamp_min(0), ui.flip(-1)[:, :r]
            si = torch.where(ev > ev[:1] * 1e-14, ev, torch.zeros_like(ev)).sqrt()
        us.append(ui)
        ss.append(si)
    return torch.stack(us), torch.stack(ss)


class NaturalFactors:
    """Per-series factors of ``L_i = eps * C_i^{1/2}``."""

    def __init__(self, basis, v, coef, eps):
        self.basis, self.v, self.coef, self.eps = basis, v, coef, float(eps)

    @property
    def dim(self):
        return self.basis.shape[0]

    def to(self, device, dtype=torch.float32):
        return NaturalFactors(self.basis.to(device, dtype), self.v.to(device, dtype), self.coef.to(device, dtype), self.eps)

    def mul(self, idx):
        """``w -> eps * C_i^{1/2} w`` for the series ``idx``; ``w`` is (B, D) or (B, D, q)."""
        b, v, c = self.basis, self.v[idx], self.coef[idx]

        def f(w):
            if w.dim() == 2:
                y = torch.einsum("bmr,br->bm", v, c * torch.einsum("bmr,bm->br", v, w @ b))
                return self.eps * (w + y @ b.T)
            bw = torch.einsum("dm,bdq->bmq", b, w)
            y = torch.einsum("bmr,brq->bmq", v, c[:, :, None] * torch.einsum("bmr,bmq->brq", v, bw))
            return self.eps * (w + torch.einsum("dm,bmq->bdq", b, y))
        return f


def build_naturalness(x_train, eps, alpha, k, p, chunk=256, tol=1e-12):
    """``x_train``: all training series (N, T, C), labeled and unlabeled."""
    x = torch.as_tensor(x_train, dtype=torch.float32)
    n = x.shape[0]
    flat = x.reshape(n, -1).double()
    mu, basis = fit_pca(flat, p)
    z = (flat - mu) @ basis                                      # PCA coordinate (N, m)
    dist = torch.cdist(z, z)
    dist.fill_diagonal_(float("inf"))
    nbr = dist.topk(min(k, n - 1), dim=1, largest=False).indices
    vs, cs = [], []
    for i0 in range(0, n, chunk):
        sl = slice(i0, min(n, i0 + chunk))
        d = (z[nbr[sl]] - z[sl][:, None, :]).transpose(1, 2) / (nbr.shape[1] ** 0.5)   # (b, m, k)
        u, s = _svd(d)
        s = torch.where(s > tol, s, torch.zeros_like(s))
        s = s / s.amax(dim=1, keepdim=True).clamp_min(1e-300)    # rho_i1 = 1
        vs.append(u)
        cs.append((1.0 + alpha * s * s).clamp_min(0).sqrt() - 1.0)
    return NaturalFactors(basis, torch.cat(vs), torch.cat(cs), eps)
