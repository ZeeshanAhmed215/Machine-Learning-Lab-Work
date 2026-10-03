# ============================================
# LAB: Random Forest Classification
# Dataset: Breast Cancer Dataset
# ============================================

# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================
# 2. Load Dataset
# ============================================

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

print("Dataset Shape:", X.shape)
print("\nFirst 5 rows:")
print(X.head())

print("\nTarget Names:")
print(data.target_names)


# ============================================
# 3. Check Missing Values
# ============================================

print("\nMissing Values:")
print(X.isnull().sum().sum())


# ============================================
# 4. Data Preprocessing
# ============================================

# Since this dataset contains numerical features,
# no categorical encoding is required.

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ============================================
# 5. Split Dataset
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# ============================================
# 6. Train Random Forest Classifier
# ============================================

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)


# ============================================
# 7. Make Predictions
# ============================================

y_pred_rf = rf_model.predict(X_test)


# ============================================
# 8. Evaluate Random Forest
# ============================================

accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print("\n===== Random Forest Results =====")

print("Accuracy :", accuracy_rf)
print("Precision:", precision_rf)
print("Recall   :", recall_rf)
print("F1-Score :", f1_rf)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred_rf,
    target_names=data.target_names
))


# ============================================
# 9. Confusion Matrix
# ============================================

cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=data.target_names,
    yticklabels=data.target_names
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ============================================
# 10. Train Single Decision Tree
# ============================================

dt_model = DecisionTreeClassifier(
    random_state=42
)

dt_model.fit(X_train, y_train)


# ============================================
# 11. Make Predictions using Decision Tree
# ============================================

y_pred_dt = dt_model.predict(X_test)


# ============================================
# 12. Evaluate Decision Tree
# ============================================

accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)

print("\n===== Decision Tree Results =====")

print("Accuracy :", accuracy_dt)
print("Precision:", precision_dt)
print("Recall   :", recall_dt)
print("F1-Score :", f1_dt)


# ============================================
# 13. Compare Both Models
# ============================================

results = pd.DataFrame({
    "Model": ["Random Forest", "Decision Tree"],
    "Accuracy": [accuracy_rf, accuracy_dt],
    "Precision": [precision_rf, precision_dt],
    "Recall": [recall_rf, recall_dt],
    "F1-Score": [f1_rf, f1_dt]
})

print("\n===== Model Comparison =====")
print(results)


# ============================================
# 14. Visualize Model Comparison
# ============================================

results.set_index("Model").plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Random Forest vs Decision Tree")
plt.ylabel("Score")
plt.ylim(0, 1.1)
plt.xticks(rotation=0)
plt.legend()
plt.grid(axis="y")
plt.show()