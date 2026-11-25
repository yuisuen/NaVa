"""Evaluate the released weights (UCR, 10% labeled series) on the test sets of the 5 folds.

    python scripts/evaluate.py --dataset GunPoint                       # all 5 folds, weights from the Hugging Face Hub
    python scripts/evaluate.py --dataset GunPoint --fold 0
    python scripts/evaluate.py --dataset GunPoint --weights /path/to/weights   # a local copy (<dataset>/fold<k>.pt)

The splits must be built first (``scripts/make_splits.py --benchmark ucr``).
"""

import argparse
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from nava.backbones import FCN  # noqa: E402
from nava.data import load_fold  # noqa: E402
from nava.trainer import evaluate  # noqa: E402

REPO = "isyuisuen/NaVa-UCR-10pct"


def weight_file(args, dataset, fold):
    name = f"{dataset}/fold{fold}.pt"
    if args.weights:
        return os.path.join(args.weights, name)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=args.repo, filename=name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--fold", type=int, default=None, help="one fold (default: all 5)")
    ap.add_argument("--config", default=os.path.join(HERE, "configs", "ucr.yaml"))
    ap.add_argument("--weights", default=None, help="local directory with <dataset>/fold<k>.pt")
    ap.add_argument("--repo", default=REPO, help="Hugging Face repository of the weights")
    a = ap.parse_args()

    with open(a.config) as f:
        conf = yaml.safe_load(f)
    root = conf["data_root"] if os.path.isabs(conf["data_root"]) else os.path.join(HERE, conf["data_root"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accs = []
    for fold in ([a.fold] if a.fold is not None else range(conf["folds"])):
        d = load_fold(root, a.dataset, fold, 0.1)
        model = FCN(d["x_test"].shape[2], d["n_class"])
        model.load_state_dict(torch.load(weight_file(a, a.dataset, fold), map_location="cpu", weights_only=True))
        model.to(device)
        test = DataLoader(TensorDataset(torch.as_tensor(d["x_test"]), torch.as_tensor(d["y_test"], dtype=torch.long)),
                          batch_size=256)
        acc, _ = evaluate(model, test, device)
        accs.append(acc)
        print(f"{a.dataset} fold {fold}: test accuracy {acc:.2f}")
    if len(accs) > 1:
        print(f"{a.dataset}: {st.mean(accs):.2f} +- {st.stdev(accs):.2f} over {len(accs)} folds")


if __name__ == "__main__":
    main()
