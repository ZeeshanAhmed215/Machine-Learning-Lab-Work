
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)





cancer = load_breast_cancer()


df = pd.DataFrame(cancer.data, columns=cancer.feature_names)


df['target'] = cancer.target

print("Dataset loaded successfully!")
print("Shape of dataset:", df.shape)





print("\nFirst 5 rows of the dataset:")
print(df.head())


print("\nMissing values in each column:")
print(df.isnull().sum())


print("\nDataset Information:")
print(df.info())


print("\nStatistical Summary:")
print(df.describe())



print("\nClass Distribution:")
print(df['target'].value_counts())

class_labels = {
    0: 'Malignant',
    1: 'Benign'
}

df['diagnosis'] = df['target'].map(class_labels)

print("\nClass Distribution with Labels:")
print(df['diagnosis'].value_counts())



plt.figure(figsize=(6, 4))
sns.countplot(x='diagnosis', data=df)
plt.title('Class Distribution: Malignant vs Benign')
plt.xlabel('Diagnosis')
plt.ylabel('Number of Samples')
plt.show()





df[cancer.feature_names[:10]].hist(
    figsize=(15, 10),
    bins=20
)

plt.suptitle('Feature Distributions', fontsize=16)
plt.tight_layout()
plt.show()



X = cancer.data
y = cancer.target

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)




X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)



scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("\nFeatures standardized successfully.")



model = LogisticRegression(max_iter=1000)


model.fit(X_train_scaled, y_train)

print("Logistic Regression model trained successfully.")


y_pred = model.predict(X_test_scaled)

print("\nPredictions:")
print(y_pred)



accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy Score:")
print(f"{accuracy:.4f}")




cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Malignant', 'Benign'],
    yticklabels=['Malignant', 'Benign']
)

plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.show()




print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=['Malignant', 'Benign']
    )
)