import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
churn_data = pd.read_csv("Telco-Customer-Churn.csv")

# Data preprocessing
churn_data['TotalCharges'] = pd.to_numeric(churn_data['TotalCharges'], errors='coerce')
churn_data['TotalCharges'].fillna(churn_data['TotalCharges'].median(), inplace=True)

# Simulate rate change by increasing MonthlyCharges by 10%
churn_data['AdjustedMonthlyCharges'] = churn_data['MonthlyCharges'] * 1.1

# Simulate external economic factor
churn_data['EconomicFactor'] = np.random.choice(['Low', 'Medium', 'High'], size=len(churn_data))

# Convert categorical variables to numeric
le = LabelEncoder()
for column in churn_data.columns:
    if churn_data[column].dtype == 'object':
        churn_data[column] = le.fit_transform(churn_data[column])

# Feature and target separation
X = churn_data.drop('Churn', axis=1)
y = churn_data['Churn']

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model training
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    results[name] = {
        'Confusion Matrix': confusion_matrix(y_test, predictions),
        'Classification Report': classification_report(y_test, predictions)
    }

# Parameter Tuning for Random Forest
parameters = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15, 20]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), parameters, cv=5)
grid_search.fit(X_train, y_train)
best_rf_model = grid_search.best_estimator_

# Model Evaluation
best_rf_predictions = best_rf_model.predict(X_test)
best_rf_cm = confusion_matrix(y_test, best_rf_predictions)
best_rf_cr = classification_report(y_test, best_rf_predictions)

# Plotting confusion matrix for the best model
plt.figure(figsize=(10, 7))
sns.heatmap(best_rf_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Churned', 'Churned'], yticklabels=['Not Churned', 'Churned'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for Tuned Random Forest')
plt.show()

# Print reports and display feature importance for the best model
print("Best Random Forest Model Performance:\n", best_rf_cr)
feature_importances = pd.Series(best_rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Feature importances:\n", feature_importances)
