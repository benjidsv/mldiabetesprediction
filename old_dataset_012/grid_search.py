from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, recall_score
from data import get_data

X_train, X_test, y_train, y_test = get_data()

print(f"Training dataset: {X_train.shape}, Testing dataset: {X_test.shape}")

from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import joblib

# Define the parameter grid for BalancedRandomForestClassifier
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [10, 20, None],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4],  # Minimum samples required at a leaf node
    'class_weight': ['balanced', 'balanced_subsample'],  # Handling imbalance
}


# Wrap GridSearchCV with tqdm progress bar
class TqdmGridSearchCV(GridSearchCV):
    def fit(self, X, y=None, **fit_params):
        # Total combinations of parameters
        total_fits = len(self.param_grid['n_estimators']) * \
                     len(self.param_grid['max_depth']) * \
                     len(self.param_grid['min_samples_split']) * \
                     len(self.param_grid['min_samples_leaf']) * \
                     len(self.param_grid['class_weight'])

        # Create tqdm progress bar
        with tqdm(total=total_fits, desc="Grid Search Progress") as pbar:
            original_fit = super().fit

            # Custom function to increment progress bar
            def fit_and_progress(*args, **kwargs):
                result = original_fit(*args, **kwargs)
                pbar.update(1)
                return result

            # Use joblib to replace fit with progress bar wrapper
            with joblib.parallel_backend('loky', n_jobs=-1):
                self._fit = fit_and_progress
                return self._fit(X, y, **fit_params)

rf_model = BalancedRandomForestClassifier(
    random_state=42,
    verbose=2,
    replacement=False,          # Explicitly set to the current default value
    bootstrap=True,             # Explicitly set to the current default value
    sampling_strategy='auto'    # Explicitly set to the current default value
)

# Instantiate TqdmGridSearchCV object
grid_search = TqdmGridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    scoring='f1_macro',
    cv=3,
    verbose=2,
    n_jobs=-1
)

# Run grid search with progress bar
grid_search.fit(X_train, y_train)

# Best parameters from grid search
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Evaluate the best model
best_rf_model = grid_search.best_estimator_

y_pred_train = best_rf_model.predict(X_train)
y_pred_test = best_rf_model.predict(X_test)

# Evaluate performance on training and testing sets
print("Training Accuracy:", accuracy_score(y_train, y_pred_train))
print("Testing Accuracy:", accuracy_score(y_test, y_pred_test))

# Classification report for test set
print("Classification Report:\n", classification_report(y_test, y_pred_test))
