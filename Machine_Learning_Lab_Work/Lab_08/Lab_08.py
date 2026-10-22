
# LAB TASK: K-Nearest Neighbors (KNN) Classification

# Step 1: Import required libraries
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Step 2: Load the dataset
dataset = load_breast_cancer()

X = pd.DataFrame(dataset.data, columns=dataset.feature_names)

# Convert target numbers into categorical class names
y = dataset.target_names[dataset.target]

# Encode categorical target labels into numbers
encoder = LabelEncoder()
y = encoder.fit_transform(y)

print("Dataset loaded successfully!")
print("Dataset shape:", X.shape)
print("\nFirst five rows:")
print(X.head())

print("\nClass labels:", encoder.classes_)
print("\nMissing values before preprocessing:", X.isnull().sum().sum())

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Step 4: Train KNN with different values of K
k_values = [3, 5, 7]
results = []

for k in k_values:

    # Preprocess data and train the KNN classifier
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])

    model.fit(X_train, y_train)

    # Step 5: Make predictions
    y_pred = model.predict(X_test)

    # Step 6: Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average="weighted", zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, average="weighted", zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, average="weighted", zero_division=0
    )

    results.append({
        "K": k,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-Score": round(f1, 4)
    })

    print(f"\nResults for K = {k}")
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-Score:", round(f1, 4))

    if k == 5:
        print("\nClassification report for K = 5:")
        print(classification_report(
            y_test,
            y_pred,
            target_names=encoder.classes_,
            zero_division=0
        ))

# Step 7: Compare accuracy for different K values
results_df = pd.DataFrame(results)

print("\n--- Performance Comparison ---")
print(results_df.to_string(index=False))

best_row = results_df.loc[results_df["Accuracy"].idxmax()]
print(
    f"\nBest K based on test accuracy: {int(best_row['K'])}"
)
print(f"Best accuracy: {best_row['Accuracy']:.4f}")
