import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
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

# Additional feature engineering: Create tenure categories
churn_data['TenureCategory'] = pd.cut(churn_data['tenure'], bins=[0, 12, 24, 48, 72, np.inf], 
                                       labels=['0-12 months', '13-24 months', '25-48 months', '49-72 months', '73+ months'])

# Convert 'TenureCategory' to numeric
churn_data['TenureCategory'] = churn_data['TenureCategory'].astype('category').cat.codes

# Feature and target separation
X = churn_data.drop('Churn', axis=1)
y = churn_data['Churn']

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model training with Random Forest
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    results[name] = {
        'Confusion Matrix': confusion_matrix(y_test, predictions),
        'Classification Report': classification_report(y_test, predictions),
        'ROC AUC': roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    }

# Parameter Tuning for Random Forest using GridSearchCV
parameters = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), parameters, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rf_model = grid_search.best_estimator_

# Model Evaluation for Best Random Forest Model
best_rf_predictions = best_rf_model.predict(X_test)
best_rf_cm = confusion_matrix(y_test, best_rf_predictions)
best_rf_cr = classification_report(y_test, best_rf_predictions)
best_rf_roc_auc = roc_auc_score(y_test, best_rf_model.predict_proba(X_test)[:, 1])

# Plotting confusion matrix for the best model
plt.figure(figsize=(10, 7))
sns.heatmap(best_rf_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Churned', 'Churned'], yticklabels=['Not Churned', 'Churned'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for Tuned Random Forest')
plt.show()

# Feature Importance Visualization
feature_importances = pd.Series(best_rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
feature_importances.plot(kind='bar', title='Feature Importance for Tuned Random Forest')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()

# Print final evaluation reports
print("Best Random Forest Model Performance:\n", best_rf_cr)
print("\nBest Random Forest ROC AUC:", round(best_rf_roc_auc, 4))

# Display results
print("\nTop features influencing churn prediction:")
print(feature_importances.head(10))
