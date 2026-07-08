import argparse
import os

from catboost import CatBoostRegressor
from metrics_utils import run_full_evaluation

MODEL_NAME = "CatBoost"
DEFAULT_K = 10


def main():

    parser = argparse.ArgumentParser(description="CatBoost regression for dam cost overrun.")

    parser.add_argument(
        "input_csv",
        nargs="?",
        default="dam_ml_ready (1).csv"
    )

    parser.add_argument(
        "-o",
        "--out-dir",
        default="catboost_outputs"
    )

    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help="Number of features selected."
    )

    args = parser.parse_args()

    # Create a dedicated temp folder (safe on Windows)
    os.makedirs("catboost_tmp", exist_ok=True)

    model = CatBoostRegressor(

        iterations=1000,
        learning_rate=0.03,
        depth=4,

        loss_function="RMSE",
        eval_metric="RMSE",

        l2_leaf_reg=5,
        random_strength=1,

        bootstrap_type="Bayesian",

        random_seed=42,

        verbose=False,

        train_dir="catboost_tmp",

        allow_writing_files=False

    )

    run_full_evaluation(
        model=model,
        model_name=MODEL_NAME,
        k=args.k,
        input_csv=args.input_csv,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()