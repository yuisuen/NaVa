"""Train NaVa on one (dataset, label ratio, fold).

    python scripts/run.py --config configs/ucr.yaml --dataset GunPoint --ratio 0.1 --fold 0 --out results/gunpoint_f0.json

The hyperparameters of the dataset are read from ``nava/hparams/<benchmark>_<ratio>.csv``;
any of them can be overridden on the command line, e.g. ``--eps 2 --gamma 1``.
"""

import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import yaml  # noqa: E402

from nava.data import load_fold  # noqa: E402
from nava.trainer import train  # noqa: E402

PER_DATASET = ("eps", "gamma", "beta", "tau", "alpha", "p")


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(HERE, path)


def table_row(path, dataset):
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["dataset"] == dataset:
                return {k: (int(float(r[k])) if k == "p" else float(r[k])) for k in PER_DATASET}
    raise KeyError(f"{dataset} is not in {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ratio", type=float, required=True)
    ap.add_argument("--fold", type=int, required=True)
    for k in PER_DATASET:
        ap.add_argument("--" + k, type=int if k == "p" else float, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    with open(a.config) as f:
        conf = yaml.safe_load(f)
    given = {k: getattr(a, k) for k in PER_DATASET if getattr(a, k) is not None}
    hp = {}
    if len(given) < len(PER_DATASET):
        hp = table_row(resolve(f"nava/hparams/{conf['benchmark']}_{a.ratio:g}.csv"), a.dataset)
    hp.update(given)

    data = load_fold(resolve(conf["data_root"]), a.dataset, a.fold, a.ratio)
    res = train(data, conf["backbone"], {**conf["nava"], **conf["training"], **hp})
    res.update(dataset=a.dataset, ratio=a.ratio, fold=a.fold)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f)
    print(f"{a.dataset} ratio {a.ratio:g} fold {a.fold}: val {res['val_acc']:.2f}  test {res['test_acc']:.2f} "
          f"(step {res['best_step']})")


if __name__ == "__main__":
    main()
