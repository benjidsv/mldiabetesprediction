import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('../../dataset/diabetes_binary_5050split_health_indicators_BRFSS2015.csv')

# Normalize Age and BMI
scaler = StandardScaler()
data['Normalized_BMI'] = scaler.fit_transform(data[['BMI']])
data['Normalized_Age'] = scaler.fit_transform(data[['Age']])

# Create BMI categories
data['BMI_Category'] = pd.cut(data['BMI'], bins=[0, 18.5, 24.9, 29.9, 100], labels=["Underweight", "Normal", "Overweight", "Obese"])

# Log-transform MentHlth and PhysHlth
data['Log_MentHlth'] = np.log1p(data['MentHlth'])
data['Log_PhysHlth'] = np.log1p(data['PhysHlth'])

# Combine features
data['HighBP_HeartDisease'] = data['HighBP'] * data['HeartDiseaseorAttack']
data['PhysActivity_BMI'] = data['PhysActivity'] * data['BMI']
data['Socioeconomics_NoDoc'] = data['Income'] * data['Education'] * data['NoDocbcCost']

# Drop less relevant features
columns_to_drop = ['BMI', 'Age', 'MentHlth', 'PhysHlth', 'Education', 'Income', 'NoDocbcCost', 'AnyHealthcare', 'Fruits']
data = data.drop(columns=columns_to_drop)

# 4. Scatter Plots for Key Features
sns.pairplot(data, vars=['HighBP'], hue='Diabetes_binary', diag_kind='kde')
plt.show()

# 5. Boxplots for Outliers
for column in ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Veggies', 'HvyAlcoholConsump', 'GenHlth', 'DiffWalk', 'Sex', 'Normalized_BMI', 'Normalized_Age', 'BMI_Category', 'Log_MentHlth', 'Log_PhysHlth', 'HighBP_HeartDisease', 'PhysActivity_BMI', 'Socioeconomics_NoDoc']:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='Diabetes_binary', y=column, data=data)
    plt.title(f'{column} vs Diabetes_binary')
    plt.xlabel('Diabetes Binary (0: No, 1: Yes)')
    plt.ylabel(column)
    plt.show()

# 6. Overlap Analysis
from scipy.stats import gaussian_kde

def calculate_overlap(X_class0, X_class1):
    kde_class0 = gaussian_kde(X_class0)
    kde_class1 = gaussian_kde(X_class1)
    x_range = np.linspace(min(min(X_class0), min(X_class1)), max(max(X_class0), max(X_class1)), 1000)
    pdf_class0 = kde_class0(x_range)
    pdf_class1 = kde_class1(x_range)
    overlap_area = np.trapz(np.minimum(pdf_class0, pdf_class1), x_range)
    return x_range, pdf_class0, pdf_class1, overlap_area

X_class0 = data[data['Diabetes_binary'] == 0]
X_class1 = data[data['Diabetes_binary'] == 1]

for feature_name in ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Veggies', 'HvyAlcoholConsump', 'GenHlth', 'DiffWalk', 'Sex', 'Normalized_BMI', 'Normalized_Age', 'BMI_Category', 'Log_MentHlth', 'Log_PhysHlth', 'HighBP_HeartDisease', 'PhysActivity_BMI', 'Socioeconomics_NoDoc']:
    x_range, pdf_class0, pdf_class1, overlap_area = calculate_overlap(X_class0[feature_name], X_class1[feature_name])
    plt.figure(figsize=(8, 4))
    plt.plot(x_range, pdf_class0, label='Class 0')
    plt.plot(x_range, pdf_class1, label='Class 1')
    plt.fill_between(x_range, np.minimum(pdf_class0, pdf_class1), alpha=0.5, color='gray', label='Overlap')
    plt.title(f'Density Overlap for {feature_name} (Overlap Area = {overlap_area:.2f})')
    plt.xlabel(feature_name)
    plt.ylabel('Density')
    plt.legend()
    plt.show()
