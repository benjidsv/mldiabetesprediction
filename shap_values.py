import shap
from sklearn.ensemble import RandomForestClassifier
from data import get_data

X_train, X_test, y_train, y_test = get_data()


simplified_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
simplified_model.fit(X_train, y_train)
explainer = shap.TreeExplainer(simplified_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values[:, :, 1], X_test)