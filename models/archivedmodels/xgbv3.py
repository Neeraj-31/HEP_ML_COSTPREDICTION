import argparse

from xgboost import XGBRegressor

from metrics_utils import run_full_evaluation

MODEL_NAME="XGBoost"

def main():

    parser=argparse.ArgumentParser()

    parser.add_argument("input_csv",nargs="?",default="dam_ml_ready (1).csv")

    parser.add_argument("-o","--out-dir",default="xgb_outputs")

    parser.add_argument("--k",type=int,default=10)

    args=parser.parse_args()

    model=XGBRegressor(

        n_estimators=500,

        learning_rate=0.03,

        max_depth=4,

        subsample=0.8,

        colsample_bytree=0.8,

        reg_alpha=1,

        reg_lambda=2,

        objective="reg:squarederror",

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