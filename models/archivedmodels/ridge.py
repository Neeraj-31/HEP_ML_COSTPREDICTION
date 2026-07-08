import argparse
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.compose import TransformedTargetRegressor

from metrics_utils import run_full_evaluation

MODEL_NAME = "Ridge"
DEFAULT_K = 8
DEFAULT_ALPHA = 2.0


def main():
    parser = argparse.ArgumentParser(description="Ridge regression for dam cost overrun.")
    parser.add_argument("input_csv", nargs="?", default=r"dam_ml_ready (1).csv")
    parser.add_argument("-o", "--out-dir", default="ridge_outputs")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Number of features to select (SelectKBest).")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Ridge L2 regularization strength.")
    args = parser.parse_args()

    base_model = Ridge(alpha=args.alpha)
    
    # Wrap with arcsinh to protect the linear model weights from outlier pulling
    model = TransformedTargetRegressor(
        regressor=base_model,
        func=np.arcsinh,
        inverse_func=np.sinh
    )

    run_full_evaluation(model, f"{MODEL_NAME} (alpha={args.alpha})", args.k, args.input_csv, args.out_dir)


if __name__ == "__main__":
    main()