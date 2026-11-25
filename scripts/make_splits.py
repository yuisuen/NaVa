"""Build the fixed 5-fold splits of one benchmark.

    python scripts/make_splits.py --benchmark ucr       --root <UCR archive>  --out data/ucr
    python scripts/make_splits.py --benchmark uea       --root <UEA archive>  --out data/uea
    python scripts/make_splits.py --benchmark realworld --root <UCR archive>  --catcc_root <CA-TCC data> --out data/realworld
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nava.data import export_catcc_pt, export_kfold, export_unequal, write_manifest  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 2001
CATCC_PT = ("HAR", "Epilepsy", "sleepEDF")
UNEQUAL = ("SpokenArabicDigits",)


def names(fname):
    return [l.strip() for l in open(os.path.join(HERE, "datasets", fname)) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=["ucr", "uea", "realworld"])
    ap.add_argument("--root", required=True, help="UCR / UEA archive with <name>/<name>_TRAIN.(tsv|ts)")
    ap.add_argument("--catcc_root", default=None, help="CA-TCC data with <name>/{train,val,test}.pt (realworld)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", nargs="*", default=None, help="only these datasets of the benchmark")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.benchmark == "ucr":
        todo, ratios = names("ucr106.txt"), [0.1, 0.2, 0.4]
    elif a.benchmark == "uea":
        todo, ratios = names("uea5.txt"), [0.1]
    else:
        todo, ratios = names("realworld10.txt"), [0.01, 0.05]
        if a.catcc_root is None:
            ap.error("--catcc_root is required for the realworld benchmark")

    if a.datasets:
        unknown = set(a.datasets) - set(todo)
        if unknown:
            ap.error(f"not in the {a.benchmark} list: {sorted(unknown)}")
        todo = [n for n in todo if n in a.datasets]
    records = []
    for i, name in enumerate(todo, 1):
        if name in CATCC_PT:
            rec = export_catcc_pt(name, a.catcc_root, a.out, ratios)
        elif name in UNEQUAL:
            rec = export_unequal(name, a.root, a.out, ratios, seed=SEED)
        else:
            rec = export_kfold(name, a.root, a.out, ratios, seed=SEED)
        records.append(rec)
        print(f"[{i:3d}/{len(todo)}] {name:<32} N={rec['n']:>6} T={rec['length']:>5} C={rec['channels']:>3} "
              f"K={rec['n_class']:>3}", flush=True)
    write_manifest(a.out, records, dict(seed=SEED, folds=5, ratios=ratios))


if __name__ == "__main__":
    main()
