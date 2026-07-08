import argparse

from sklearn.ensemble import HistGradientBoostingRegressor

from metrics_utils import run_full_evaluation

MODEL_NAME="HistGradientBoosting"

def main():

    parser=argparse.ArgumentParser()

    parser.add_argument("input_csv",nargs="?",default="dam_ml_ready (1).csv")

    parser.add_argument("-o","--out-dir",default="histgb_outputs")

    parser.add_argument("--k",type=int,default=10)

    args=parser.parse_args()

    model=HistGradientBoostingRegressor(

        learning_rate=0.03,

        max_depth=4,

        max_iter=300,

        min_samples_leaf=3,

        l2_regularization=1,

        random_state=42

    )

    run_full_evaluation(

        model,

        MODEL_NAME,

        args.k,

        args.input_csv,

        args.out_dir

    )

if __name__=="__main__":
    main()