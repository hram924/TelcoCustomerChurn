import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import joblib
from imblearn.over_sampling import SMOTE
import shap
from lime.lime_tabular import LimeTabularExplainer

# Load the dataset
churn_data = pd.read_csv("Telco-Customer-Churn.csv")

# Data preprocessing
churn_data['TotalCharges'] = pd.to_numeric(churn_data['TotalCharges'], errors='coerce')
churn_data['TotalCharges'].fillna(churn_data['TotalCharges'].median(), inplace=True)

# Simulate rate change by increasing MonthlyCharges by 10%
churn_data['AdjustedMonthlyCharges'] = churn_data['MonthlyCharges'] * 1.1

# Simulate external economic factor
churn_data['EconomicFactor'] = np.random.choice(['Low', 'Medium', 'High'], size=len(churn_data))

# Feature engineering: Interaction feature and Polynomial features
churn_data['ChargesPerTenure'] = churn_data['MonthlyCharges'] / churn_data['Tenure']
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(churn_data.drop('Churn', axis=1))

# Convert categorical variables to numeric
le = LabelEncoder()
for column in churn_data.columns:
    if churn_data[column].dtype == 'object':
        churn_data[column] = le.fit_transform(churn_data[column])

# Feature and target separation
X = churn_data.drop('Churn', axis=1)
y = churn_data['Churn']

# Handle imbalanced data using SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Advanced Model Tuning with Optuna
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 3, 15)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
best_params = study.best_trial.params
print(f"Best Parameters from Optuna: {best_params}")

# Model training with Stacking and GridSearchCV for RandomForest
models = {
    'Random Forest': RandomForestClassifier(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'], random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

stacking_model = StackingClassifier(estimators=[('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                                                ('gb', GradientBoostingClassifier(random_state=42)),
                                                ('lr', LogisticRegression(random_state=42))],
                                    final_estimator=LogisticRegression())

# Fit the models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    results[name] = {
        'Confusion Matrix': confusion_matrix(y_test, predictions),
        'Classification Report': classification_report(y_test, predictions)
    }

stacking_model.fit(X_train, y_train)
stacking_predictions = stacking_model.predict(X_test)
results['Stacking'] = {
    'Confusion Matrix': confusion_matrix(y_test, stacking_predictions),
    'Classification Report': classification_report(y_test, stacking_predictions)
}

# Model Evaluation
best_rf_model = RandomForestClassifier(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'], random_state=42)
best_rf_model.fit(X_train, y_train)
best_rf_predictions = best_rf_model.predict(X_test)

# Confusion Matrix for Random Forest
plt.figure(figsize=(10, 7))
best_rf_cm = confusion_matrix(y_test, best_rf_predictions)
sns.heatmap(best_rf_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Churned', 'Churned'], yticklabels=['Not Churned', 'Churned'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for Tuned Random Forest')
plt.show()

# Print Classification Report for Random Forest
print("Best Random Forest Model Performance:\n", classification_report(y_test, best_rf_predictions))

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, best_rf_predictions)
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, best_rf_predictions)
auc = roc_auc_score(y_test, best_rf_predictions)
plt.plot(fpr, tpr, label=f'AUC = {auc}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.show()

# Save the best model using joblib
joblib.dump(best_rf_model, 'best_random_forest_model.pkl')

# SHAP explanation
explainer = shap.TreeExplainer(best_rf_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values[1], X_test, plot_type="bar")

# LIME explanation for a single prediction
explainer = LimeTabularExplainer(X_train, training_labels=y_train, mode='classification')
explanation = explainer.explain_instance(X_test[0], best_rf_model.predict_proba)
explanation.show_in_notebook()

# Feature importance for the best model
feature_importances = pd.Series(best_rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Feature importances:\n", feature_importances)
