import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve
from sklearn.model_selection import GridSearchCV
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
        # Delegate to the underlying model
        return self.model.predict_proba(X)

    def set_threshold(self, threshold):
        # Update the threshold
        self.threshold = threshold

    def feature_importances(self):
        return self.model.feature_importances_


# Grid Search for RandomForest
#param_grid_1 = {
#    'n_estimators': [1750, 2000, 2250, 2500],
#}
#grid_search_1 = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_1, scoring='f1', cv=3, verbose=2)
#grid_search_1.fit(X_train, y_train)
best_n_estimators = 2000#grid_search_1.best_params_['n_estimators']

# Step 2: Optimize max_depth
#param_grid_2 = {
#    'max_depth': [8, 10, 12],
#}
#grid_search_2 = GridSearchCV(RandomForestClassifier(random_state=42, n_estimators=best_n_estimators), param_grid_2, scoring='f1', cv=3, verbose=2)
#grid_search_2.fit(X_train, y_train)
best_max_depth = 10#grid_search_2.best_params_['max_depth']
#print("Best max_depth:", best_max_depth)

# Step 3: Optimize min_samples_split and min_samples_leaf
param_grid_3 = {
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 4, 6, 8, 10],
}
#grid_search_3 = GridSearchCV(RandomForestClassifier(random_state=42, n_estimators=best_n_estimators, max_depth=best_max_depth), param_grid_3, scoring='accuracy', cv=3, verbose=2)
#grid_search_3.fit(X_train, y_train)
best_min_samples_split = 15#grid_search_3.best_params_['min_samples_split']
best_min_samples_leaf = 2#grid_search_3.best_params_['min_samples_leaf']
#print("Best min_samples_split:", best_min_samples_split)
#print("Best min_samples_leaf:", best_min_samples_leaf)

# Step 4: Optimize class_weight
param_grid_4 = {
    'class_weight': ['balanced', 'balanced_subsample'],
}
#grid_search_4 = GridSearchCV(RandomForestClassifier(
#    random_state=42,
#    n_estimators=best_n_estimators,
#    max_depth=best_max_depth,
#    min_samples_split=best_min_samples_split,
#    min_samples_leaf=best_min_samples_leaf), param_grid_4, scoring='accuracy', cv=3, verbose=2)
#grid_search_4.fit(X_train, y_train)
best_class_weight = 'balanced_subsample'#grid_search_4.best_params_['class_weight']
#print("Best class_weight:", best_class_weight)

best_params = {
    'n_estimators': best_n_estimators,
    'max_depth': best_max_depth,
    'min_samples_split': best_min_samples_split,
    'min_samples_leaf': best_min_samples_leaf,
    'class_weight': best_class_weight,
}

print("Combined Best Parameters:", best_params)


# Train the model with best parameters
rf_model = ThresholdRandomForest(RandomForestClassifier(**best_params, random_state=42), 0.5)
rf_model.fit(X_train, y_train)

y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

# Evaluate performance on training and testing sets
print("Training Accuracy:", accuracy_score(y_train, y_pred_train))
print("Testing Accuracy:", accuracy_score(y_test, y_pred_test))

# Classification report for test set
print("Classification Report:\n", classification_report(y_test, y_pred_test))

# Feature importance
features_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_model.feature_importances()
})

# Sort the features by importance in descending order
features_df = features_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
print(features_df)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.barh(features_df['Feature'], features_df['Importance'], align='center')
plt.gca().invert_yaxis()  # Invert y-axis for better readability
plt.xlabel('Importance')
plt.title('Feature Importances')
plt.show()

# Precision-recall curve
y_probs = rf_model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

plt.plot(thresholds, precision[:-1], label="Precision")
plt.plot(thresholds, recall[:-1], label="Recall")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.legend()
plt.show()
