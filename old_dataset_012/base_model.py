from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from data import get_data

X_train, X_test, y_train, y_test = get_data()

print(f"Training dataset: {X_train.shape}, Testing dataset: {X_test.shape}")

rf_model = BalancedRandomForestClassifier(
    random_state=42,
    verbose=2,
    replacement=False,          # Explicitly set to the current default value
    bootstrap=True,             # Explicitly set to the current default value
    sampling_strategy='auto',    # Explicitly set to the current default value
    class_weight='balanced_subsample',
    max_depth=None,
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=300
)

rf_model.fit(X_train, y_train)

y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

# Evaluate performance on training and testing sets
print("Training Accuracy:", accuracy_score(y_train, y_pred_train))
print("Testing Accuracy:", accuracy_score(y_test, y_pred_test))

# Classification report for test set
print("Classification Report:\n", classification_report(y_test, y_pred_test))
