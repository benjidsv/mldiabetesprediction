import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split

data_012 = pd.read_csv('../../dataset/diabetes_012_health_indicators_BRFSS2015.csv')
data_012 = data_012.drop(columns=['CholCheck', 'AnyHealthcare', 'HvyAlcoholConsump', 'Stroke'])
data_012['Socioeconomics'] = data_012['Income'] * data_012['Education']
data_012['LifestyleIndex'] = data_012['Age'] * data_012['BMI']

data_012 = data_012[data_012['Diabetes_012'] != 1.0]

print("Successfully imported dataset")

# Split data into training and testing sets
X = data_012.drop(columns=['Diabetes_012'])
y = data_012['Diabetes_012']
X_train_og, X_test, y_train_og, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Successfully split data into train and test set")

print("Resampling data... (2mn)")
# Resample using SMOTEENN
smoteenn = SMOTEENN(random_state=42)
X_train, y_train = smoteenn.fit_resample(X_train_og, y_train_og)
print("Successfully resampled data using SMOTEENN")


def get_data():
    return X_train, X_test, y_train, y_test

