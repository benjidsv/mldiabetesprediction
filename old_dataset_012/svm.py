import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, make_scorer, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import time

from tqdm import tqdm

from data import get_data
X_train, X_test, y_train, y_test = get_data()

# Set up the pipeline with scaling and SVM
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # SVM benefits from standardized data
    ('svm', SVC(probability=True))  # Use SVM as the classifier
])
print("Succesfully set up training pipeline")

# Define hyperparameter grid for SVM
param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
    'svm__kernel': ['linear', 'rbf', 'poly'],
    'svm__class_weight': [None, 'balanced']  # Automatic balancing of class weights
}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='f1_macro',
    cv=3,
    verbose=3,
    n_jobs=-1
)

# Fit GridSearchCV
#grid_search.fit(X_train, y_train)

print("Training model...")
start = time.time()

with tqdm(total=len(param_grid['svm__C']) * len(param_grid['svm__gamma']) * len(param_grid['svm__kernel']) * len(param_grid['svm__class_weight'])) as pbar:
    for _ in pipeline.fit(X_train, y_train):
        pbar.update(1)

end = time.time()

best_svm = pipeline#grid_search.best_estimator_

print(f"Training took {end - start} seconds")
# Evaluate on training data
y_pred_train = best_svm.predict(X_train)
print("Training Accuracy:", accuracy_score(y_train, y_pred_train))

# Evaluate on test data
y_pred_test = best_svm.predict(X_test)
print("Testing Accuracy:", accuracy_score(y_test, y_pred_test))
print("Classification Report:\n", classification_report(y_test, y_pred_test))
