import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load data
# Replace with your actual path if running locally
df = pd.read_csv('dam_ml_ready (1).csv')

# Drop non-numeric identifiers (like project_name) for ML training
X = df.drop(columns=['project_name', 'cost_overrun_pct']) 
y = df['cost_overrun_pct']

# Handle any unexpected NaNs just in case
X = X.fillna(X.median())

# Initialize the robust Random Forest Regressor
# Tuning hyperparameters like min_samples_leaf helps prevent overfitting
rf_model = RandomForestRegressor(
    n_estimators=200, 
    max_depth=10, 
    min_samples_leaf=2, 
    random_state=42
)

# ==========================================
# SPLIT 1: 80/20 Train-Test Split
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
rf_model.fit(X_train, y_train)

# Predictions
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

# Calculations
r2_train_8020 = r2_score(y_train, y_train_pred)
r2_test_8020 = r2_score(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

# ==========================================
# SPLIT 2: Leave-One-Out Cross-Validation (LOOCV)
# ==========================================
loo = LeaveOneOut()

# For LOOCV R², we collect predictions for each held-out sample
y_preds_loo = []
y_true_loo = []

# LOOCV loop
for train_index, test_index in loo.split(X):
    X_tr, X_te = X.iloc[train_index], X.iloc[test_index]
    y_tr, y_te = y.iloc[train_index], y.iloc[test_index]
    
    # Fit on N-1 samples
    rf_model.fit(X_tr, y_tr)
    
    # Predict on the 1 held out sample
    pred = rf_model.predict(X_te)
    
    y_preds_loo.append(pred[0])
    y_true_loo.append(y_te.values[0])

# LOOCV R² calculation
r2_loocv = r2_score(y_true_loo, y_preds_loo)

# ==========================================
# OUTPUT RESULTS & PERFORMANCE BENCHMARKS
# ==========================================
print("## --- ML Model Statistical Summary --- ##\n")

print(f"### 80/20 Split Results:")
print(f" - Train R² Score: {r2_train_8020:.4f}")
print(f" - Test R² Score:  {r2_test_8020:.4f}")
print(f" - Test MAE:       {mae_test:.2f}")
print(f" - Test RMSE:      {rmse_test:.2f}\n")

print(f"### LOOCV Results:")
print(f" - Overall LOOCV R² Score: {r2_loocv:.4f}\n")

print("### Performance Interpretation:")

def evaluate_r2(r2_train, r2_test):
    gap = r2_train - r2_test
    if r2_test > 0.75 and gap < 0.15:
        return "**Excellent!** High predictive accuracy and generalized perfectly."
    elif r2_test > 0.50 and gap < 0.20:
        return "**Decent.** Good real-world capability, though there's some minor noise."
    elif r2_train > 0.85 and r2_test < 0.40:
        return "**Bad (Overfitting).** The model memorized the training data but failed on unseen data."
    elif r2_test < 0.30:
        return "**Bad.** The model is struggling to find a meaningful signal in the data."
    else:
        return "**Decent to Mediocre.** Yields some predictive power but could be optimized."

print(f" -> 80/20 Evaluation: {evaluate_r2(r2_train_8020, r2_test_8020)}")