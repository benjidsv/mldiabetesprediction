import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, recall_score, make_scorer, f1_score
from sklearn.utils import compute_sample_weight
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

data_012 = pd.read_csv('../dataset/diabetes_012_health_indicators_BRFSS2015.csv')
data_012 = data_012.drop(columns=['CholCheck', 'AnyHealthcare', 'HvyAlcoholConsump', 'Stroke'])
data_012['Socioeconomics'] = data_012['Income'] * data_012['Education']
data_012['LifestyleIndex'] = data_012['Age'] * data_012['BMI']

from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X = data_012.drop(columns=['Diabetes_012'])
y = data_012['Diabetes_012']

print("Succesfully imported dataset")

X_train_og, X_test, y_train_og, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Succesfully split data into train and test set")

from imblearn.combine import SMOTEENN
smoteenn = SMOTEENN(random_state=42)
X_train, y_train = smoteenn.fit_resample(X_train_og, y_train_og)
print("Succesfully resampled data using SMOTEENN")

# Define hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7, 10],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.3],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'scale_pos_weight': [1, len(y_train) / sum(y_train == 1), len(y_train) / sum(y_train == 2)]
}

# Initialize XGBoost model
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')

# Use GridSearchCV for tuning
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='f1_macro',
    cv=3,
    verbose=3,
    n_jobs=-1
)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

best_xgb = grid_search.best_estimator_

# Evaluate on training data
y_pred_train = best_xgb.predict(X_train)
print("Training Accuracy:", accuracy_score(y_train, y_pred_train))

# Evaluate on test data
y_pred_test = best_xgb.predict(X_test)
print("Testing Accuracy:", accuracy_score(y_test, y_pred_test))
print("Classification Report:\n", classification_report(y_test, y_pred_test))
xgb_model.plot_importance(best_xgb, max_num_features=10)
plt.show()