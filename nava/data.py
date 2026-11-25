"""Data loading and the fixed 5-fold evaluation protocol.

Every dataset is stored once per fold as ``<root>/<dataset>/fold<k>.npz`` with

    x_train, y_train, x_val, y_val, x_test, y_test   (N, T, C) float32 / (N,) int64
    lab_<ratio>                                       (N_train,) bool, True = labeled
    n_class, fold, seed

Three ways of building these files are provided (see ``scripts/make_splits.py``):

* ``export_kfold``      -- UCR and UEA archives: pool TRAIN and TEST, stratified
                           5-fold cross-validation, each test fold 20%, the rest split
                           3:1 into training (60%) and validation (20%); per-series
                           z-normalisation; stratified labeled subsets that are nested
                           across label ratios.
* ``export_unequal``    -- SpokenArabicDigits (unequal lengths): NaN-padded to the
                           maximum length, filled with the per-position mean of the
                           training fold, then the same protocol as above.
* ``export_catcc_pt``   -- HAR / Epilepsy / sleepEDF: the fixed train / val / test
                           splits released with CA-TCC; fold ``k`` uses the labeled
                           subset drawn with seed ``42 + k``.
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np

__all__ = ["load_fold", "export_kfold", "export_unequal", "export_catcc_pt", "ratio_key"]

# Folder name of the equal-length replacements shipped with the UCR 2018 archive
# for the datasets with missing values or variable lengths.
ADJUSTED_DIRS = ("Missing_value_and_variable_length_datasets_adjusted",
                 "_Missing_value_and_variable_length_datasets_adjusted")


def ratio_key(ratio: float) -> str:
    return f"lab_{int(round(ratio * 100)):03d}"


# ------------------------------------------------------------------ readers

def _read_tsv(path):
    a = np.loadtxt(path, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    x = np.nan_to_num(a[:, 1:])[:, :, None]
    return x, np.array([str(v) for v in a[:, 0].astype(object)])


def _ts_rows(path):
    rows, in_data = [], False
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if in_data:
                rows.append(line)
            elif line.lower().startswith("@data"):
                in_data = True
    return rows


def _parse_dim(field):
    return np.array([float(v) for v in field.split(",")], dtype=np.float64)


def _read_ts(path):
    xs, ys = [], []
    for r in _ts_rows(path):
        fields = r.split(":")
        ys.append(fields[-1].strip())
        xs.append(np.stack([_parse_dim(d) for d in fields[:-1]], axis=-1))
    return np.stack(xs, axis=0), np.array(ys)


def _read_ts_unequal(path):
    xs, ys = [], []
    for r in _ts_rows(path):
        fields = r.split(":")
        ys.append(fields[-1].strip())
        dims = [_parse_dim(d) for d in fields[:-1]]
        if len({len(d) for d in dims}) != 1:
            raise ValueError(f"{path}: a series has channels of different lengths")
        xs.append(np.stack(dims, axis=-1))
    return xs, np.array(ys)


def _split_file(root, name, split, prefer_adjusted=True):
    bases = [os.path.join(root, d, name, f"{name}_{split}") for d in ADJUSTED_DIRS] if prefer_adjusted else []
    bases.append(os.path.join(root, name, f"{name}_{split}"))
    for base in bases:
        for ext in (".tsv", ".ts"):
            if os.path.exists(base + ext):
                return base + ext
    raise FileNotFoundError(f"{bases[-1]}.tsv / .ts not found")


def _encode_labels(y_raw):
    """Class indices in [0, K): numeric labels in numeric order, then the others."""
    def key(s):
        try:
            return (0, float(s), "")
        except ValueError:
            return (1, 0.0, s)
    classes = sorted(set(y_raw.tolist()), key=key)
    lut = {c: i for i, c in enumerate(classes)}
    return np.array([lut[v] for v in y_raw], dtype=np.int64), len(classes)


def load_pooled(root, name, prefer_adjusted=True):
    """TRAIN and TEST of a UCR (.tsv) or UEA (.ts) dataset pooled: (N, T, C) float32, labels, K."""
    parts = []
    for split in ("TRAIN", "TEST"):
        path = _split_file(root, name, split, prefer_adjusted)
        parts.append(_read_tsv(path) if path.endswith(".tsv") else _read_ts(path))
    x = np.concatenate([parts[0][0], parts[1][0]], axis=0).astype(np.float32)
    y, n_class = _encode_labels(np.concatenate([parts[0][1], parts[1][1]], axis=0))
    return x, y, n_class


# --------------------------------------------------------------- protocol

def split_indices(y, seed, fold, n_folds=5):
    """(train, val, test) indices: fold ``fold`` of a stratified k-fold split is the test
    set, and the remaining folds are split 3:1 into training and validation."""
    from sklearn.model_selection import StratifiedKFold
    outer = list(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))
    rest, test = outer[fold % n_folds]
    inner = StratifiedKFold(n_folds - 1, shuffle=True, random_state=seed)
    tr, va = next(iter(inner.split(np.zeros(len(rest)), y[rest])))
    return rest[tr], rest[va], test


def znorm_series(v):
    """Per-series, per-channel z-normalisation along time; constant series become 0."""
    mu = v.mean(axis=1, keepdims=True)
    sd = v.std(axis=1, keepdims=True)
    flat = v.max(axis=1, keepdims=True) == v.min(axis=1, keepdims=True)
    sd = np.where(flat, 1.0, sd)
    return np.where(flat, 0.0, (v - mu) / sd)


def label_mask(y, ratio, n_class, seed):
    """Stratified labeled subset of the training split. Masks drawn with the same seed
    are nested across ratios (the same per-class permutation, a longer prefix)."""
    rng = np.random.RandomState(seed + 10_000)
    mask = np.zeros(len(y), dtype=bool)
    for c in range(n_class):
        pos = np.where(y == c)[0]
        k = max(1, int(round(ratio * len(pos))))
        mask[rng.permutation(pos)[:k]] = True
    return mask


def _write_fold(dest, fold, seed, n_class, splits, masks):
    payload = dict(n_class=np.int64(n_class), fold=np.int64(fold), seed=np.int64(seed))
    for part, (x, y) in splits.items():
        if not np.isfinite(x).all():
            raise ValueError(f"{dest} fold{fold}: x_{part} is not finite")
        payload[f"x_{part}"] = x.astype(np.float32)
        payload[f"y_{part}"] = y
    payload.update(masks)
    np.savez(os.path.join(dest, f"fold{fold}.npz"), **payload)


def _check_masks(name, fold, y, n_class, masks, ratios):
    ordered = [masks[ratio_key(r)] for r in sorted(ratios)]
    for lo, hi in zip(ordered, ordered[1:]):
        if not np.all(hi[lo]):
            raise ValueError(f"{name} fold{fold}: labeled subsets are not nested")
    if len(np.unique(y[ordered[0]])) != n_class:
        raise ValueError(f"{name} fold{fold}: the smallest labeled subset misses a class")


def export_kfold(name, root, out, ratios, seed=2001, n_folds=5):
    """UCR / UEA dataset with equal-length series -> ``out/name/fold*.npz``."""
    x, y, n_class = load_pooled(root, name)
    dest = os.path.join(out, name)
    os.makedirs(dest, exist_ok=True)
    for fold in range(n_folds):
        tr, va, te = split_indices(y, seed, fold, n_folds)
        splits = {p: (znorm_series(x[i]).astype(np.float32), y[i]) for p, i in (("train", tr), ("val", va), ("test", te))}
        masks = {ratio_key(r): label_mask(y[tr], r, n_class, seed) for r in ratios}
        _check_masks(name, fold, y[tr], n_class, masks, ratios)
        _write_fold(dest, fold, seed, n_class, splits, masks)
    return dict(dataset=name, n=len(y), length=int(x.shape[1]), channels=int(x.shape[2]), n_class=n_class)


def export_unequal(name, root, out, ratios, seed=2001, n_folds=5):
    """UEA dataset with unequal lengths (SpokenArabicDigits): NaN-padded to the maximum
    length, NaNs filled with the training fold's per-position mean, then z-normalised."""
    series, labels = [], []
    for split in ("TRAIN", "TEST"):
        s, l = _read_ts_unequal(_split_file(root, name, split, prefer_adjusted=False))
        series += s
        labels.append(l)
    t_max, n_ch = max(len(s) for s in series), series[0].shape[1]
    x = np.full((len(series), t_max, n_ch), np.nan, dtype=np.float32)
    for j, s in enumerate(series):
        x[j, :len(s)] = s
    y_str = np.concatenate(labels)
    classes = np.unique(y_str)
    y, n_class = np.searchsorted(classes, y_str).astype(np.int64), len(classes)
    dest = os.path.join(out, name)
    os.makedirs(dest, exist_ok=True)
    for fold in range(n_folds):
        tr, va, te = split_indices(y, seed, fold, n_folds)
        parts = {"train": x[tr].copy(), "val": x[va].copy(), "test": x[te].copy()}
        with warnings.catch_warnings():                          # positions that are NaN in every series
            warnings.simplefilter("ignore", RuntimeWarning)
            fill = np.nanmean(parts["train"], axis=0)
        fill[np.isnan(fill)] = 1e-6
        for a in parts.values():
            idx = np.where(np.isnan(a))
            a[idx] = fill[idx[1], idx[2]]
        splits = {p: (znorm_series(parts[p]).astype(np.float32), y[i]) for p, i in (("train", tr), ("val", va), ("test", te))}
        masks = {ratio_key(r): label_mask(y[tr], r, n_class, seed) for r in ratios}
        _check_masks(name, fold, y[tr], n_class, masks, ratios)
        _write_fold(dest, fold, seed, n_class, splits, masks)
    return dict(dataset=name, n=len(y), length=t_max, channels=n_ch, n_class=n_class)


def export_catcc_pt(name, src, out, ratios, seeds=(42, 43, 44, 45, 46)):
    """HAR / Epilepsy / sleepEDF from the CA-TCC release (``src/name/{train,val,test}.pt``,
    already preprocessed). The splits are fixed; fold ``k`` differs only in the labeled
    subset, drawn with ``train_test_split(random_state=seeds[k])`` as in CA-TCC."""
    import torch
    from sklearn.model_selection import train_test_split

    def load(part):
        d = torch.load(os.path.join(src, name, part + ".pt"), map_location="cpu", weights_only=False)
        return d["samples"].numpy().astype(np.float32).transpose(0, 2, 1), d["labels"].numpy().astype(np.int64)

    (xtr, ytr), (xva, yva), (xte, yte) = load("train"), load("val"), load("test")
    lut = {v: i for i, v in enumerate(np.unique(ytr))}
    ytr, yva, yte = (np.array([lut[v] for v in a], dtype=np.int64) for a in (ytr, yva, yte))
    n_class = len(lut)
    dest = os.path.join(out, name)
    os.makedirs(dest, exist_ok=True)
    for fold, seed in enumerate(seeds):
        masks = {}
        for r in ratios:
            lab_idx, _ = train_test_split(np.arange(len(ytr)), test_size=1 - r, random_state=seed)
            m = np.zeros(len(ytr), dtype=bool)
            m[lab_idx] = True
            masks[ratio_key(r)] = m
        _write_fold(dest, fold, seed, n_class, {"train": (xtr, ytr), "val": (xva, yva), "test": (xte, yte)}, masks)
    return dict(dataset=name, n=len(ytr) + len(yva) + len(yte), length=int(xtr.shape[1]),
                channels=int(xtr.shape[2]), n_class=n_class)


def write_manifest(out, records, protocol):
    with open(os.path.join(out, "manifest.json"), "w", newline="\n") as f:
        json.dump({"protocol": protocol, "datasets": {r["dataset"]: r for r in records}}, f, indent=1)


def load_fold(root, dataset, fold, ratio):
    """One fold of the protocol, with the labeled mask of ``ratio``."""
    z = np.load(os.path.join(root, dataset, f"fold{fold}.npz"))
    key = ratio_key(ratio)
    if key not in z.files:
        raise KeyError(f"{dataset} fold{fold}: no labeled subset for ratio {ratio}")
    out = {k: z[k] for k in ("x_train", "y_train", "x_val", "y_val", "x_test", "y_test")}
    out["lab"] = z[key]
    out["n_class"] = int(z["n_class"])
    return out
