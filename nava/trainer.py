"""Training NaVa on one fold, with validation-based model selection.

Every ``eval_every`` steps the model is evaluated on the validation and test sets. The
reported test accuracy is the one at the evaluation with the highest validation accuracy
(ties: lower validation loss, then the earlier evaluation), and training stops early when
the validation accuracy has not improved for ``patience`` epochs. The test set is never
used for any decision.
"""

import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .backbones import build_backbone
from .naturalness import build_naturalness
from .perturbation import bn_batch_view, feasible_set, kl_consistency, worst_perturbation

__all__ = ["train", "DEFAULTS"]

DEFAULTS = dict(
    k=100, M=5,                                   # the same for all datasets
    eps=1.0, gamma=0.5, beta=1.0, tau=0.9, alpha=1.0, p=64,
    m0=1e-3, xi=1e-6,
    epochs=1000, steps_per_epoch=100, batch_size=64, unlabeled_batch_size=256, lr=1e-3,
    eval_every=50, patience=100, seed=42,
)


@torch.no_grad()
def evaluate(model, loader, device):
    """-> (accuracy %, mean cross-entropy)."""
    model.eval()
    correct = total = 0
    loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return 100.0 * correct / max(total, 1), loss / max(total, 1)


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(fold_data, backbone, cfg, device=None, log=print):
    """``fold_data``: output of ``nava.data.load_fold``; ``cfg``: DEFAULTS updated with the
    hyperparameters of this run. Returns a dict with the selected test accuracy and the
    evaluation curve."""
    cfg = {**DEFAULTS, **cfg}
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _seed(cfg["seed"])
    d = fold_data
    xtr, ytr, lab = d["x_train"], d["y_train"], d["lab"]
    n_class, (t_len, n_ch) = d["n_class"], d["x_train"].shape[1:]

    t0 = time.time()
    nat = build_naturalness(xtr, cfg["eps"], cfg["alpha"], cfg["k"], cfg["p"]).to(device)
    t_nat = time.time() - t0

    torch.manual_seed(cfg["seed"] + 500_000)
    torch.cuda.manual_seed_all(cfg["seed"] + 500_000)
    model = build_backbone(backbone, n_ch, n_class, t_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    tt = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt)
    lab_t = torch.as_tensor(lab, dtype=torch.bool, device=device)
    ytr_t = torch.as_tensor(ytr, dtype=torch.long, device=device)
    g = torch.Generator().manual_seed(cfg["seed"])
    lab_ld = DataLoader(TensorDataset(tt(xtr[lab]), tt(ytr[lab], torch.long)), batch_size=cfg["batch_size"],
                        shuffle=True, generator=g)
    all_ld = DataLoader(TensorDataset(tt(xtr), torch.arange(len(ytr))), batch_size=cfg["unlabeled_batch_size"],
                        shuffle=True, generator=g)
    val_ld = DataLoader(TensorDataset(tt(d["x_val"]), tt(d["y_val"], torch.long)), batch_size=256)
    test_ld = DataLoader(TensorDataset(tt(d["x_test"]), tt(d["y_test"], torch.long)), batch_size=256)

    curve = []                                     # [step, val_acc, val_loss, test_acc]
    best = None                                    # (val_acc, -val_loss, -step), test_acc, step
    last_improve_step, step = 0, 0
    patience_steps = cfg["patience"] * cfg["steps_per_epoch"]
    epoch = 0
    for epoch in range(cfg["epochs"]):
        model.train()
        lab_it, all_it = iter(lab_ld), iter(all_ld)
        for _ in range(cfg["steps_per_epoch"]):
            try:
                xl, yl = next(lab_it)
            except StopIteration:
                lab_it = iter(lab_ld)
                xl, yl = next(lab_it)
            try:
                xu, iu = next(all_it)
            except StopIteration:
                all_it = iter(all_ld)
                xu, iu = next(all_it)
            xl, yl, xu, iu = xl.to(device), yl.to(device), xu.to(device), iu.to(device)

            sup = F.cross_entropy(model(xl), yl)
            with bn_batch_view(model):
                with torch.no_grad():
                    p_clean = F.softmax(model(xu), dim=1)
                to_input = feasible_set(model, xu, iu, p_clean, lab_t[iu], ytr_t[iu], nat,
                                        cfg["gamma"], cfg["M"], cfg["tau"], cfg["m0"])
                delta = worst_perturbation(model, xu, p_clean, to_input, nat.dim, cfg["xi"])
                cons = kl_consistency(p_clean, model(xu + delta))
            loss = sup + cfg["beta"] * cons
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1

            if step % cfg["eval_every"] == 0:
                va, vl = evaluate(model, val_ld, device)
                ta, _ = evaluate(model, test_ld, device)
                curve.append([step, va, vl, ta])
                key = (va, -vl, -step)
                if best is None or key > best[0]:
                    if best is None or va > best[0][0]:
                        last_improve_step = step
                    best = (key, ta, step)
                model.train()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            log(f"epoch {epoch + 1:4d}  sup {sup.item():.4f}  cons {cons.item():.5f}  "
                f"best val {best[0][0]:.2f} @ step {best[2]}  ({time.time() - t0:.0f}s)")
        if step - last_improve_step >= patience_steps:
            log(f"no better validation accuracy for {cfg['patience']} epochs; stopping at epoch {epoch + 1}")
            break

    (va, nvl, _), ta, at = best
    return dict(val_acc=va, val_loss=-nvl, test_acc=ta, best_step=at, epochs_run=epoch + 1,
                seconds=round(time.time() - t0, 1), naturalness_seconds=round(t_nat, 1),
                pca_dim=int(nat.basis.shape[1]), curve=curve, config=cfg)
