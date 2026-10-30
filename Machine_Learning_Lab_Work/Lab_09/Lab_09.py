
# ============================================================
# LAB TASK: Naive Bayes Classification Using Iris Dataset
# Algorithm: Gaussian Naive Bayes
# Dataset: Iris
# ============================================================

# 1. Import required libraries
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ------------------------------------------------------------
# 2. Load the Iris dataset
# ------------------------------------------------------------
iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = pd.Series(iris.target, name="target")

# Display dataset information
print("=" * 60)
print("IRIS DATASET")
print("=" * 60)

print("\nFirst five records:")
print(X.head())

print("\nDataset shape:", X.shape)
print("\nMissing values in each feature:")
print(X.isnull().sum())

print("\nTarget classes:", iris.target_names)

# ------------------------------------------------------------
# 3. Split dataset into training and testing sets
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ------------------------------------------------------------
# 4. Preprocess data and train Gaussian Naive Bayes
# ------------------------------------------------------------
model = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("classifier", GaussianNB())
])

model.fit(X_train, y_train)

print("\nGaussian Naive Bayes model trained successfully.")

# ------------------------------------------------------------
# 5. Make predictions on the test dataset
# ------------------------------------------------------------
y_pred = model.predict(X_test)

results = pd.DataFrame({
    "Actual Species": iris.target_names[y_test.to_numpy()],
    "Predicted Species": iris.target_names[y_pred]
})

print("\nTest Set Predictions:")
print(results.to_string(index=False))

# ------------------------------------------------------------
# 6. Evaluate model performance
# ------------------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy Percentage: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names,
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ------------------------------------------------------------
# 7. Test the model on new, unseen data
# ------------------------------------------------------------
# Feature order:
# Sepal length, Sepal width, Petal length, Petal width
new_flower = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=iris.feature_names
)

prediction = model.predict(new_flower)
probabilities = model.predict_proba(new_flower)

predicted_species = iris.target_names[prediction[0]]

print("\n" + "=" * 60)
print("PREDICTION FOR NEW FLOWER")
print("=" * 60)

print("\nNew flower measurements:")
print(new_flower.to_string(index=False))

print("\nPredicted Species:", predicted_species)

print("\nPrediction Probabilities:")
for species, probability in zip(
    iris.target_names,
    probabilities[0]
):
    print(f"{species}: {probability * 100:.2f}%")

print("\nLab task completed successfully.")
