import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


# Instructions ================================================================================
# Task 2: Multivariate Linear Regression: - 
# • Load dataset: Use the same Diabetes dataset, but include all 10 features to predict disease 
# progression. 
# • Perform EDA: 
# o Generate a correlation heatmap between features and the target. 
# o Create pair plots for selected features vs. target. 
# • Implement Multivariate Linear Regression: 
# o Use all independent variables to predict the target. 
# o Fit and predict using the model. 
# • Compare actual vs. predicted values using: 
# o Scatter plot (predicted vs. actual). 
# o Residual plot. 
# • Evaluate model performance with: 
# o MSE 
# o RMSE 
# o R² score 
# =====================================================================



data=load_diabetes(as_frame=True)
df=data.frame
X=df[['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']]
y=df[["target"]]



plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(  correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap: Features and Disease Progression')
plt.show()



sns.pairplot(df, x_vars=X, y_vars=y, height=4)
plt.show()


X_train ,X_test , y_train  ,y_test=train_test_split(X,y,test_size=0.20, random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)

predicted_value=model.predict(X_test)

plt.scatter(y_test, predicted_value)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.show()

residuals = y_test - predicted_value



# MSE
mse_multi = mean_squared_error(y_test, predicted_value)

# RMSE
rmse_multi = np.sqrt(mse_multi)

# R^2 Score
r2_multi = r2_score(y_test, predicted_value)

print("MSE:", mse_multi)
print("RMSE:", rmse_multi)
print("R^2 Score:", r2_multi)


