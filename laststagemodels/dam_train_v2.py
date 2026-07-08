import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression

import dam_common as C

OUT_MODEL_PATH = "dam_best_rf_v2.pkl"

FEATURES = [
    "cost_per_mw",
    "initial_cost",
    "R30mm_total",
    "dist_to_nearest_pilgrim_km",
    "R100mm_total",
    "glof_composite_score",
    "steep_frac_25",
    "CWD_mean",                    
    "landowner_displaced",        
    "dist_to_nearest_resort_km",   
    "PRCPTOT_max",                 
]
RF_PARAMS = dict(
    n_estimators=500,
    max_depth=4,          
    min_samples_leaf=3,
    random_state=C.RANDOM_SEED,
)

def make_y_shift(y):
    return 1.0 - y.min()


def log_transform(y, shift):
    return np.log(y + shift)


def inv_log(y_log, shift):
    return np.exp(y_log) - shift


def loocv_predict(X, y_log, rf_params):
    loo = LeaveOneOut()
    preds_log = np.zeros(len(y_log))
    for tr_idx, te_idx in loo.split(X):
        scaler = StandardScaler().fit(X[tr_idx])
        Xtr, Xte = scaler.transform(X[tr_idx]), scaler.transform(X[te_idx])
        m = RandomForestRegressor(**rf_params)
        m.fit(Xtr, y_log[tr_idx])
        preds_log[te_idx] = m.predict(Xte)
    return preds_log


def evaluate(preds, y, label):
    resid = preds - y
    rmse = np.sqrt(np.mean(resid ** 2))
    mae = np.mean(np.abs(resid))
    bias = np.mean(resid)
    pbias = 100 * np.sum(resid) / np.sum(y)
    ss_res, ss_tot = np.sum(resid ** 2), np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    pear = np.corrcoef(preds, y)[0, 1]
    print(f"  [{label}] R2={r2:.3f}  RMSE={rmse:.1f}  MAE={mae:.1f}  "
          f"Bias={bias:+.1f}  PBIAS={pbias:+.1f}%  Pearson r={pear:.3f}")
    return dict(rmse=rmse, mae=mae, bias=bias, pbias=pbias, r2=r2, pearson=pear)


def main():
    df = C.load_cleaned()
    y = df["cost_overrun_pct"].values.astype(float)
    X = df[FEATURES].values.astype(float)
    y_shift = make_y_shift(y)
    y_log = log_transform(y, y_shift)

    print(f"Training on {len(df)} historical projects, {len(FEATURES)} features ")

    print("Leave-One-Out CV :")
    oof_log = loocv_predict(X, y_log, RF_PARAMS)
    oof_pred = inv_log(oof_log, y_shift)
    evaluate(oof_pred, y, "RAW, out-of-fold")

    calibrator = LinearRegression().fit(oof_pred.reshape(-1, 1), y - oof_pred)
    oof_pred_calibrated = oof_pred + calibrator.predict(oof_pred.reshape(-1, 1))
    evaluate(oof_pred_calibrated, y, "BIAS-CORRECTED, out-of-fold")

    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)
    final_model = RandomForestRegressor(**RF_PARAMS)
    final_model.fit(X_s, y_log)

    tree_preds = np.stack([t.predict(X_s) for t in final_model.estimators_], axis=1)
    tree_log_std = tree_preds.std(axis=1)
    avg_tree_log_std = float(tree_log_std.mean())
    print(f"Average across-tree log-space std at training points: "
          f"{avg_tree_log_std:.3f} ")

    bundle = {
        "model": final_model,
        "scaler": scaler,
        "y_shift": y_shift,
        "features": FEATURES,
        "bias_correction": {
            "intercept": float(calibrator.intercept_),
            "coef": float(calibrator.coef_[0]),
        },
        "avg_tree_log_std": avg_tree_log_std,
        "loocv_metrics_raw": evaluate(oof_pred, y, "recheck raw"),
        "loocv_metrics_calibrated": evaluate(oof_pred_calibrated, y, "recheck calibrated"),
        "n_train": len(df),
    }
    with open(OUT_MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)


if __name__ == "__main__":
    main()
