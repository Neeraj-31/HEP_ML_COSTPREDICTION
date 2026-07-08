"""
train_elasticnet.py
=====================
ElasticNet regression on dam_ml_ready.csv, target = cost_overrun_pct.

ElasticNet blends Ridge's L2 penalty and Lasso's L1 penalty
(l1_ratio=0.5 by default = equal mix). Included as a middle-ground
check between train_ridge.py and train_lasso.py.

Usage:
    python train_elasticnet.py dam_ml_ready.csv
    python train_elasticnet.py dam_ml_ready.csv --k 8 --alpha 1.0 --l1-ratio 0.5 -o elasticnet_outputs
"""

import argparse
from sklearn.linear_model import ElasticNet

from metrics_utils import run_full_evaluation

MODEL_NAME = "ElasticNet"
DEFAULT_K = 8
DEFAULT_ALPHA = 1.0
DEFAULT_L1_RATIO = 0.5


def main():
    parser = argparse.ArgumentParser(description="ElasticNet regression for dam cost overrun.")
    parser.add_argument("input_csv", nargs="?", default=r"dam_ml_ready (1).csv")
    parser.add_argument("-o", "--out-dir", default="elasticnet_outputs")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Number of features to select (SelectKBest).")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Overall regularization strength.")
    parser.add_argument("--l1-ratio", type=float, default=DEFAULT_L1_RATIO, help="0=pure Ridge, 1=pure Lasso.")
    args = parser.parse_args()

    model = ElasticNet(alpha=args.alpha, l1_ratio=args.l1_ratio, max_iter=20000)
    run_full_evaluation(
        model, f"{MODEL_NAME} (alpha={args.alpha}, l1_ratio={args.l1_ratio})",
        args.k, args.input_csv, args.out_dir,
    )


if __name__ == "__main__":
    main()