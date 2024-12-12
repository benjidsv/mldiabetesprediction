from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

data_012 = pd.read_csv('../dataset/diabetes_012_health_indicators_BRFSS2015.csv')
data_012 = data_012.drop(columns=['CholCheck', 'AnyHealthcare', 'HvyAlcoholConsump', 'Stroke'])
data_012['Socioeconomics'] = data_012['Income'] * data_012['Education']
data_012['LifestyleIndex'] = data_012['Age'] * data_012['BMI']

X = data_012.drop(columns=['Diabetes_012'])
y = data_012['Diabetes_012']

X_train_og, X_test, y_train_og, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from imblearn.combine import SMOTEENN
smoteenn = SMOTEENN(random_state=42)
X_train, y_train = smoteenn.fit_resample(X_train_og, y_train_og)

# Train KNN model
knn_model = KNeighborsClassifier(n_neighbors=5, weights='distance')  # Adjust n_neighbors as needed
knn_model.fit(X_train, y_train)

# Evaluate on the test set
y_pred_test = knn_model.predict(X_test)
y_pred_train = knn_model.predict(X_train)
print("Training Accuracy:", accuracy_score(y_train, y_pred_train))
print("Testing Accuracy:", accuracy_score(y_test, y_pred_test))
print("Classification Report:\n", classification_report(y_test, y_pred_test))