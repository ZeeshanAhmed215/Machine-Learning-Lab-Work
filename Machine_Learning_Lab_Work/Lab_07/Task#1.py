# ============================================================
# OPEN ENDED LAB
# Project: House Price & Category Prediction
# Task 1: House Price Prediction
# Model: Linear Regression
# Name: Zeeshan Ahmed
# Roll No: 24F-AI-215
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATA
# ============================================================

# The file train.csv must be in the same folder as this .py file.
df = pd.read_csv(r"E:\Online Courses\ML_Lab_Work\Machine_Learning_Lab_Work\Lab_07\train.csv")


print("\n========== DATASET ==========")
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())




# ============================================================
# 2. EXPLORE DATA
# ============================================================

print("\n========== DATA INFORMATION ==========")
df.info()

print("\nMissing values:")
print(df.isnull().sum().sort_values(ascending=False).head(20))

print("\nDuplicate rows:", df.duplicated().sum())

print("\nTarget (SalePrice) statistics:")
print(df["SalePrice"].describe())


# ============================================================
# 3. REMOVE DUPLICATES
# ============================================================

df = df.drop_duplicates().copy()


# ============================================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# Id is only an identifier, so it is not used for prediction.
if "Id" in X.columns:
    X = X.drop("Id", axis=1)


# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

# We split before outlier removal to avoid using information from
# the test set while preparing the training data.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n========== DATA SPLIT ==========")
print("Training rows:", len(X_train))
print("Testing rows :", len(X_test))


# ============================================================
# 6. OUTLIER REMOVAL
# ============================================================

# IQR = Q3 - Q1
# Values outside Q1 - 1.5*IQR and Q3 + 1.5*IQR are treated
# as outliers.
#
# We use important house-size/quality variables where extreme
# values can strongly affect Linear Regression.

outlier_columns = [
    col for col in [
        "GrLivArea",
        "TotalBsmtSF",
        "GarageArea",
        "1stFlrSF",
        "LotArea"
    ]
    if col in X_train.columns
]

train_clean = X_train.copy()
y_train_clean = y_train.copy()

for col in outlier_columns:
    q1 = train_clean[col].quantile(0.25)
    q3 = train_clean[col].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    mask = train_clean[col].isna() | train_clean[col].between(lower, upper)
    train_clean = train_clean.loc[mask]
    y_train_clean = y_train_clean.loc[train_clean.index]

print("\nRows after outlier removal:", len(train_clean))


# ============================================================
# 7. IDENTIFY COLUMN TYPES
# ============================================================

numeric_features = train_clean.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = train_clean.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumeric features:", len(numeric_features))
print("Categorical features:", len(categorical_features))


# ============================================================
# 8. DATA PREPROCESSING
# ============================================================

# Numeric data:
#   1. Fill missing values with median
#   2. Scale features using StandardScaler
#
# Categorical data:
#   1. Fill missing values with most frequent value
#   2. Convert categories into numbers using One-Hot Encoding

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# ============================================================
# 9. BUILD LINEAR REGRESSION MODEL
# ============================================================

model = Pipeline([
    ("preprocessing", preprocessor),
    ("regression", LinearRegression())
])


# ============================================================
# 10. TRAIN MODEL
# ============================================================

print("\n========== TRAINING ==========")
model.fit(train_clean, y_train_clean)

print("Linear Regression model trained successfully.")


# ============================================================
# 11. TEST MODEL
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 12. MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n========== LINEAR REGRESSION PERFORMANCE ==========")
print(f"MAE  : {mae:,.2f}")
print(f"RMSE : {rmse:,.2f}")
print(f"R²   : {r2:.4f}")


# ============================================================
# 13. VISUALIZATION
# ============================================================

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Linear Regression - Actual vs Predicted Price")
plt.tight_layout()
plt.show()


# ============================================================
# 14. FINAL PREDICTION USING PROVIDED test.csv
# ============================================================

try:
    external_test = pd.read_csv(r"E:\Online Courses\ML_Lab_Work\Machine_Learning_Lab_Work\Lab_07\test.csv")

    ids = external_test["Id"].copy()

    if "Id" in external_test.columns:
        external_features = external_test.drop("Id", axis=1)
    else:
        external_features = external_test

    final_predictions = model.predict(external_features)

    submission = pd.DataFrame({
        "Id": ids,
        "SalePrice": final_predictions
    })

    submission.to_csv(
        "linear_regression_house_price_predictions.csv",
        index=False
    )

    print("\nFinal prediction file created:")
    print("linear_regression_house_price_predictions.csv")

except FileNotFoundError:
    print("\ntest.csv was not found. Final prediction file skipped.")
