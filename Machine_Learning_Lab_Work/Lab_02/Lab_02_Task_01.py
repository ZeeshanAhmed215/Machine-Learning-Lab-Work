import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# Task 1: Simple Linear Regression:- 

# • Load dataset: Use the Diabetes dataset from sklearn.datasets.
#  Select one feature (bmi) to predict 
# the target (disease progression).
data=load_diabetes(as_frame=True)
df=data.frame
X=df[['bmi']]
y=df['target']

# • Perform Exploratory Data Analysis (EDA): 
# o Plot scatter plot of BMI vs. Disease Progression. 
# o Check correlation. 
plt.scatter(X, y)
plt.xlabel('BMI')
plt.ylabel('Disease Progression')
plt.title('Scatter Plot of BMI vs Disease Progression')
plt.show()


correlation = df['bmi'].corr(df['target'])
print(f'Correlation between BMI and Disease Progression: {correlation}')

# • Implement Simple Linear Regression using sklearn.linear_model.LinearRegression: 
# o Split data into training and testing sets. 
# o Fit the model and predict disease progression. 
# o Plot the regression line on the plt.scatter plot.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.scatter(X, y, color='blue', alpha=0.5)
plt.xlabel('BMI')
plt.ylabel('Disease Progression')
plt.title('Simple Linear Regression')
plt.show()

# • Evaluate the model using: 
# o Mean Squared Error (MSE) 
# o R² score

from sklearn.metrics import mean_squared_error, r2_score    

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')


