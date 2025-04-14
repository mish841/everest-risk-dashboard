# 3_model_logistic_regression.ipynb
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load processed datasets
X_train = pd.read_csv('X_train_processed.csv')
X_test = pd.read_csv('X_test_processed.csv')
y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')

# Function to bin risk scores into categories
def bin_risk_score(score):
    if score <= 19:
        return 0  # Very Low
    elif score <= 39:
        return 1  # Low
    elif score <= 59:
        return 2  # Medium
    elif score <= 79:
        return 3  # High
    else:
        return 4  # Very High

# Apply binning
y_train_binned = y_train['risk_score'].apply(bin_risk_score)
y_test_binned = y_test['risk_score'].apply(bin_risk_score)

# Fit Logistic Regression (multinomial)
log_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train_binned)

# Predict on test set
y_pred = log_reg.predict(X_test)

# Evaluate
print("Accuracy Score:", accuracy_score(y_test_binned, y_pred))
print("\nClassification Report:\n", classification_report(y_test_binned, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test_binned, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
            yticklabels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
