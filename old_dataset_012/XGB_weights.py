from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, recall_score, make_scorer, f1_score
from sklearn.utils import compute_sample_weight
from xgboost import XGBClassifier


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

# Class weight initialization
class_counts = y_train.value_counts()
total_samples = len(y_train)
initial_weights = {cls: total_samples / count for cls, count in class_counts.items()}


# Function to test scale_pos_weight values
def try_weights(scale_weights):
    # Map weights to the sample weight array
    sample_weights = y_train.map({cls: scale_weights[cls] for cls in y_train.unique()})

    # Train model with current weights
    model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train, sample_weight=sample_weights)

    # Predict and evaluate
    y_pred_test = model.predict(X_test)
    report = classification_report(y_test, y_pred_test, output_dict=True)
    print(f"Scale Weights: {scale_weights}")
    print(f"Testing Accuracy: {accuracy_score(y_test, y_pred_test)}")
    print(f"Class 1 Recall: {report['1.0']['recall']}")
    print(f"Class 2 Recall: {report['2.0']['recall']}")
    print()


# Test different weight configurations
weights = [
    {cls: weight * factor for cls, weight in initial_weights.items()}
    for factor in np.arange(1, 5, 0.5)  # Scale weights by 1x, 1.5x, ..., 4.5x
]

total = len(weights)
i = 1
for weight in weights:
    print(f"Iter {i}/{total}")
    try_weights(weight)
    i+=1