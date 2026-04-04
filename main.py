##############################################
# Linear Regression Model for Student Exams
# Author : Hamza 
##############################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt                                          # Used to embed the visualizations within Jupyter notebook
import seaborn as sns                                                    # Make the datasets look stunning. Like a boss.
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression                        # This is the model we are using to predict the exam score. 


raw_data = pd.read_csv('StudentPerformanceFactors2.csv')                  

raw_data.info()                                                           # This will show the information about the dataset.
sns.pairplot(raw_data[['Hours_Studied', 'Sleep_Hours', 'Exam_Score']])    # This will show the relationship between the selectedvariables in the dataset.

index = raw_data.columns
X = raw_data.drop('Exam_Score', axis=1)
y = raw_data['Exam_Score']

X = pd.get_dummies(X)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)



###############


print(model.coef_) # Examining the coefficients of the model.
# The large coefficients are the ones that are most important for predicting the exam score. 
# For example, the coefficient of Hours_Studied is 0.80, which means that for every hour studied, the exam score increases by 0.80.

print(model.coef_)

coef_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficients'])
print(coef_df)

# Predictions
predictions = model.predict(X_test)

# Visualization
plt.scatter(y_test, predictions)
plt.xlabel("Actual Exam Score")
plt.ylabel("Predicted Exam Score")
plt.title("Actual vs Predicted")
plt.show()

# Residuals
plt.hist(y_test - predictions)
plt.title("Residual Distribution")
plt.show()