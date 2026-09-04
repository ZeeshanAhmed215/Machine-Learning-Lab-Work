import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
<<<<<<< HEAD
# Task 1 – Dataset Cleaning 
# • Load the Titanic dataset (train.csv). 
df = pd.read_csv("E:/Online Courses/Lab Work/Machine_Learning_Lab_Work/Lab_01/Titanic-Dataset.csv")
=======
## Task 1 – Dataset Cleaning 
# • Load the Titanic dataset (train.csv). 
df = pd.read_csv("E:/Online Courses/Lab Work/Machine_Learning_Lab_Work/Lab#1/Titanic-Dataset.csv")
>>>>>>> e57dee11268c9dd08425fb5d8a780c421de322e7
# • Display the first 10 rows. 
print(df.head(10))
# • Check for missing values in each column. 
print(df.isnull().sum())
# • Fill missing values in the "Age" column with the mean age. 
df["Age"] = df["Age"].fillna(df["Age"].mean())
# • Drop rows where the "Embarked" column is missing. 
df = df.dropna(subset=["Embarked"])
print(df.isnull().sum()) 

# =======================================================
# Task 2 – Encoding Categorical Data 
# • Convert the "Sex" column into numeric (0 = Male, 1 = Female). 
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
# • Apply One-Hot Encoding on the "Embarked" column.
df = pd.get_dummies(df, columns=["Embarked"], prefix="Embarked")

# ==========================================================
# Task 3 – Feature Scaling & Splitting 

# ==========================================================
## Task 3 – Feature Scaling & Splitting 

# • Select features: Age, Fare, Sex, Pclass. 
features = ["Age", "Fare", "Sex", "Pclass"]
# • Apply StandardScaler to normalize them. 
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])
# • Split data into 80% training and 20% testing.
X = df[features]
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

