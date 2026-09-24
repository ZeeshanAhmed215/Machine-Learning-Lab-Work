# ------------------------------------------------------------
# TASK 01: LOAD DATASET
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)
df["target"] = iris.target
df["species"] = df["target"].map({
    0: "setosa",
    1: "versicolor",
    2: "virginica"
})


# ------------------------------------------------------------
# TASK 2: EXPLORATORY DATA ANALYSIS (EDA)
# ------------------------------------------------------------


print("\n========== FIRST 5 ROWS ==========")
print(df.head())
print("\n========== DATASET INFORMATION ==========")
df.info()
print("\n========== DATASET SHAPE ==========")
print(df.shape)
print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())
df[iris.feature_names].hist(
    figsize=(10, 8),
    bins=10
)

plt.suptitle("Distribution of Iris Features")
plt.tight_layout()
plt.show()

sns.pairplot(
    df,
    hue="species",
    diag_kind="hist"
)

plt.show()
correlation = df[iris.feature_names].corr()

print("\n========== CORRELATION MATRIX ==========")
print(correlation)
plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation Matrix")
plt.show()
X = df[iris.feature_names]


y = df["target"]


# Split into training and testing data
# 70% Training
# 30% Testing

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\n========== DATA SPLIT ==========")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ------------------------------------------------------------
# TASK 4: BUILD DECISION TREE CLASSIFIER
# ------------------------------------------------------------

model = DecisionTreeClassifier(
    criterion="gini",
    random_state=42
)


model.fit(X_train, y_train)
plt.figure(figsize=(15, 10))

plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True
)

plt.title("Decision Tree Classifier")
plt.show()

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL ACCURACY ==========")
print("Accuracy:", accuracy)
print("Accuracy (%):", accuracy * 100)
cm = confusion_matrix(y_test, y_pred)

print("\n========== CONFUSION MATRIX ==========")
print(cm)


plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()

print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)
model_gini = DecisionTreeClassifier(
    criterion="gini",
    random_state=42
)

model_gini.fit(X_train, y_train)

pred_gini = model_gini.predict(X_test)

accuracy_gini = accuracy_score(
    y_test,
    pred_gini
)


model_entropy = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)

model_entropy.fit(X_train, y_train)

pred_entropy = model_entropy.predict(X_test)

accuracy_entropy = accuracy_score(
    y_test,
    pred_entropy
)

print("\n========== GINI VS ENTROPY ==========")
print("Gini Accuracy:   ", accuracy_gini)
print("Entropy Accuracy:", accuracy_entropy)


depths = [1, 2, 3, 4, 5, 6, 10, None]

depth_results = []

for depth in depths:

    model_depth = DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        random_state=42
    )

    model_depth.fit(X_train, y_train)

    train_prediction = model_depth.predict(X_train)
    test_prediction = model_depth.predict(X_test)

    train_accuracy = accuracy_score(
        y_train,
        train_prediction
    )

    test_accuracy = accuracy_score(
        y_test,
        test_prediction
    )

    depth_results.append({
        "Max Depth": depth,
        "Training Accuracy": train_accuracy,
        "Testing Accuracy": test_accuracy
    })


depth_df = pd.DataFrame(depth_results)

print("\n========== MAX DEPTH EXPERIMENT ==========")
print(depth_df)
plt.figure(figsize=(10, 6))

plt.plot(
    range(len(depths)),
    depth_df["Training Accuracy"],
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    range(len(depths)),
    depth_df["Testing Accuracy"],
    marker="o",
    label="Testing Accuracy"
)

plt.xticks(
    range(len(depths)),
    [str(x) for x in depths]
)

plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.title("Training vs Testing Accuracy for Different Tree Depths")
plt.legend()
plt.grid()

plt.show()
split_values = [2, 5, 10, 20, 30]

split_results = []

for split in split_values:

    model_split = DecisionTreeClassifier(
        criterion="gini",
        min_samples_split=split,
        random_state=42
    )

    model_split.fit(X_train, y_train)

    train_prediction = model_split.predict(X_train)
    test_prediction = model_split.predict(X_test)

    train_accuracy = accuracy_score(
        y_train,
        train_prediction
    )

    test_accuracy = accuracy_score(
        y_test,
        test_prediction
    )

    split_results.append({
        "Min Samples Split": split,
        "Training Accuracy": train_accuracy,
        "Testing Accuracy": test_accuracy
    })

split_df = pd.DataFrame(split_results)

print("\n========== MIN_SAMPLES_SPLIT EXPERIMENT ==========")
print(split_df)
experiments = [
    {
        "Model": "Gini Default",
        "Criterion": "gini",
        "Max Depth": None,
        "Min Samples Split": 2
    },
    {
        "Model": "Entropy Default",
        "Criterion": "entropy",
        "Max Depth": None,
        "Min Samples Split": 2
    },
    {
        "Model": "Gini Depth 2",
        "Criterion": "gini",
        "Max Depth": 2,
        "Min Samples Split": 2
    },
    {
        "Model": "Gini Depth 5",
        "Criterion": "gini",
        "Max Depth": 5,
        "Min Samples Split": 2
    },
    {
        "Model": "Gini Split 10",
        "Criterion": "gini",
        "Max Depth": None,
        "Min Samples Split": 10
    },
    {
        "Model": "Gini Split 20",
        "Criterion": "gini",
        "Max Depth": None,
        "Min Samples Split": 20
    }
]
final_results = []

for experiment in experiments:

    model_exp = DecisionTreeClassifier(
        criterion=experiment["Criterion"],
        max_depth=experiment["Max Depth"],
        min_samples_split=experiment["Min Samples Split"],
        random_state=42
    )

    model_exp.fit(X_train, y_train)

    train_pred = model_exp.predict(X_train)
    test_pred = model_exp.predict(X_test)

    train_accuracy = accuracy_score(
        y_train,
        train_pred
    )

    test_accuracy = accuracy_score(
        y_test,
        test_pred
    )

    final_results.append({
        "Model": experiment["Model"],
        "Criterion": experiment["Criterion"],
        "Max Depth": experiment["Max Depth"],
        "Min Samples Split": experiment["Min Samples Split"],
        "Training Accuracy": train_accuracy,
        "Testing Accuracy": test_accuracy
    })

results_df = pd.DataFrame(final_results)
print("\n====================================================")
print("              FINAL EXPERIMENT RESULTS")
print("====================================================")
print(results_df.round(3))
















