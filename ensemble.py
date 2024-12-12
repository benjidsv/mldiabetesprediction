import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
from data import get_data

class ThresholdRandomForest:
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        # Use probabilities and apply the threshold
        probabilities = self.model.predict_proba(X)[:, 1]
        return (probabilities >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def set_threshold(self, threshold):
        self.threshold = threshold

    def feature_importances(self):
        return self.model.feature_importances_


# Load data
X_train, X_test, y_train, y_test = get_data()
print(f"Training dataset: {X_train.shape}, Testing dataset: {X_test.shape}")

print("Training RandomForestClassifier...")
# Random Forest Model
rf_model = RandomForestClassifier(
    random_state=42,
    n_estimators=1500,
    max_depth=10,
    min_samples_leaf=2,
    min_samples_split=15,
    class_weight='balanced_subsample',
)
rf_model.fit(X_train, y_train)
rf_probs = rf_model.predict_proba(X_test)[:, 1]

print("Training XGBClassifier...")
# XGBoost Model
xgb_model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    enable_categorical=True
)
xgb_model.fit(X_train, y_train)
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

# Ensemble Predictions (Averaging)
rf_weight = 0.3
xgb_weight = 0.7
ensemble_probs = (rf_probs * rf_weight + xgb_probs * xgb_weight)
ensemble_preds = (ensemble_probs >= 0.45).astype(int)

# Evaluate Ensemble Model
print("Training Accuracy (Random Forest):", accuracy_score(y_train, rf_model.predict(X_train)))
print("Training Accuracy (XGBoost):", accuracy_score(y_train, xgb_model.predict(X_train)))
print("Testing Accuracy (Ensemble):", accuracy_score(y_test, ensemble_preds))
print("Classification Report (Ensemble):\n", classification_report(y_test, ensemble_preds))
auc_score = roc_auc_score(y_test, ensemble_probs)
print(f"AUC Score (Ensemble): {auc_score:.4f}")
rf_auc = roc_auc_score(y_test, rf_probs)  # Random Forest
xgb_auc = roc_auc_score(y_test, xgb_probs)  # XGBoost

print(f"Random Forest AUC: {rf_auc:.4f}")
print(f"XGBoost AUC: {xgb_auc:.4f}")

with open("rf_model.pkl", "wb") as rf_file:
    pickle.dump(rf_model, rf_file)

with open("xgb_model.pkl", "wb") as xgb_file:
    pickle.dump(xgb_model, xgb_file)
