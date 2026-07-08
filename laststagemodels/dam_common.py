"""
dam_common.py
-------------
Shared utilities for the dam cost-overrun model: data loading, the
log-shift target transform, and a couple of small numeric helpers.
Kept deliberately tiny so both the training script and the Monte Carlo
simulator import the exact same logic (no drift between train/predict).
"""
import numpy as np
import pandas as pd

RANDOM_SEED = 42
CSV_PATH = "dam_ml_ready_cleaned.csv"


def load_cleaned(path: str = CSV_PATH) -> pd.DataFrame:
    """Load the cleaned, ML-ready historical project table."""
    return pd.read_csv(path)


def make_y_shift(y: np.ndarray) -> float:
    """
    Smallest additive shift that makes every target value strictly
    positive before taking logs (targets here are % cost overruns and
    can be negative, e.g. a project that finished under budget).
    """
    return float(-y.min() + 1.0)


def log_transform(y: np.ndarray, y_shift: float) -> np.ndarray:
    return np.log(y + y_shift)


def inv_log(y_log: np.ndarray, y_shift: float) -> np.ndarray:
    return np.exp(y_log) - y_shift
