from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, recall_score
from xgboost import XGBClassifier

data_012 = pd.read_csv('../dataset/diabetes_012_health_indicators_BRFSS2015.csv')
data_012 = data_012.drop(columns=['CholCheck', 'AnyHealthcare', 'HvyAlcoholConsump', 'Stroke'])
data_012['Socioeconomics'] = data_012['Income'] * data_012['Education']
data_012['LifestyleIndex'] = data_012['Age'] * data_012['BMI']

from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X = data_012.drop(columns=['Diabetes_012'])
y = data_012['Diabetes_012']

X_train_og, X_test, y_train_og, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from imblearn.combine import SMOTEENN
smoteenn = SMOTEENN(random_state=42)
X_train, y_train = smoteenn.fit_resample(X_train_og, y_train_og)

print(f"Training dataset: {X_train.shape}, Testing dataset: {X_test.shape}")

lgbm = LGBMClassifier(is_unbalance=True, random_state=42)
lgbm.fit(X_train, y_train)

# Evaluate the model
y_pred_train = lgbm.predict(X_train)
y_pred_test = lgbm.predict(X_test)

# Evaluate performance on training and testing sets
print("Training Accuracy:", accuracy_score(y_train, y_pred_train))
print("Testing Accuracy:", accuracy_score(y_test, y_pred_test))

# Classification report for test set
print("Classification Report:\n", classification_report(y_test, y_pred_test, zero_division=1))