"""
data_prep.py
============
Shared data loading/prep logic used by every model script, factored
out so the rules (leakage guard, target transform, climate PCA) live
in exactly one place instead of being copy-pasted six times.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

LEAKAGE_COLS = ["actual_dur", "schedule_overrun_pct"]
ID_COLS = ["project_name"]
TARGET = "cost_overrun_pct"
CLIMATE_BLOCK = [
    "PRCPTOT_mean", "PRCPTOT_max", "R10mm_mean", "RX1day_max",
    "RX5day_max", "RX7day_mean", "RX7day_max", "R95p_mean", "R99p_mean",
]


def load_and_prepare(path):
    df = pd.read_csv(path)
    y_raw = df[TARGET].copy()

    drop_cols = ID_COLS + LEAKAGE_COLS + [TARGET]
    X = df.drop(columns=drop_cols, errors="ignore")

    non_climate = [c for c in X.columns if c not in CLIMATE_BLOCK]
    imputer = SimpleImputer(strategy="median")
    X[non_climate] = imputer.fit_transform(X[non_climate])

    clim_imputer = SimpleImputer(strategy="median")
    clim_vals = clim_imputer.fit_transform(X[CLIMATE_BLOCK])
    clim_scaled = StandardScaler().fit_transform(clim_vals)
    pca = PCA(n_components=2, random_state=42)
    clim_pcs = pca.fit_transform(clim_scaled)

    X = X.drop(columns=CLIMATE_BLOCK)
    X["climate_PC1"] = clim_pcs[:, 0]
    X["climate_PC2"] = clim_pcs[:, 1]

    shift = abs(min(y_raw.min(), 0)) + 1
    y_log = np.log1p(y_raw + shift)
    return X, y_log, y_raw, shift


def inverse_target(y_log, shift):
    return np.expm1(y_log) - shift
