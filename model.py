import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve, make_scorer, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
import shap

from data import get_data

X_train, X_test, y_train, y_test = get_data()

print(f"Training dataset: {X_train.shape}, Testing dataset: {X_test.shape}")


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


rf_model = ThresholdRandomForest(RandomForestClassifier(
    random_state=42,
    verbose=1,
    class_weight='balanced_subsample',
    max_depth=10,
    min_samples_leaf=2,
    min_samples_split=15,
    n_estimators=1500
), 0.55)


def cross_validation_scores(model, X, y):
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Use accuracy as the evaluation metric
    scoring = make_scorer(accuracy_score)
    cv_scores = cross_val_score(model, X, y, cv=kfold, scoring=scoring)

    # Print results
    print(f"Cross-Validation Accuracy Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f}")
    print(f"Standard Deviation of CV Accuracy: {np.std(cv_scores):.4f}")


#cross_validation_scores(rf_model.model, X_train, y_train)

print("Training model...")
rf_model.fit(X_train, y_train)

y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

# Evaluate performance on training and testing sets
print("Training Accuracy:", accuracy_score(y_train, y_pred_train))
print("Testing Accuracy:", accuracy_score(y_test, y_pred_test))

# Classification report for test set
print("Classification Report:\n", classification_report(y_test, y_pred_test))

features_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_model.feature_importances()
})

# Sort the features by importance in descending order
features_df = features_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
print(features_df)

#import matplotlib.pyplot as plt

#plt.figure(figsize=(10, 6))
#plt.barh(features_df['Feature'], features_df['Importance'], align='center')
#plt.gca().invert_yaxis()  # Invert y-axis for better readability
#plt.xlabel('Importance')
#plt.title('Feature Importances')
#plt.show()

y_probs = rf_model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
auc_score = roc_auc_score(y_test, y_probs)
print(f"AUC Score: {auc_score:.4f}")

#f1_scores = (2 * precision * recall) / (precision + recall + 1e-10)
#best_threshold = thresholds[np.argmax(f1_scores)]
#print(f"Best Threshold: {best_threshold}")

#perm_importance = permutation_importance(rf_model.model, X_test, y_test, n_repeats=10, random_state=42)
#sorted_idx = perm_importance.importances_mean.argsort()

#rfe = RFE(estimator=RandomForestClassifier(random_state=42), n_features_to_select=10)
#rfe.fit(X_train, y_train)
#selected_features = X_train.columns[rfe.support_]
#print(f"Selected Features: {list(selected_features)}")

#plt.figure(figsize=(10, 6))
#plt.barh(X_train.columns[sorted_idx], perm_importance.importances_mean[sorted_idx], align="center")
#plt.xlabel("Permutation Importance")
#plt.title("Feature Importance (Permutation)")
#plt.show()

# Plot precision-recall curve
#import matplotlib.pyplot as plt

#plt.plot(thresholds, precision[:-1], label="Precision")
#plt.plot(thresholds, recall[:-1], label="Recall")
#plt.xlabel("Threshold")
#plt.ylabel("Score")
#plt.legend()
#plt.show()
