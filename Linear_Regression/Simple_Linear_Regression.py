                                                                          #Without PreProcessing

import numpy as np
from sklearn.linear_model import LinearRegression

# Data
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
Y = np.array([30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000])

# Linear Regression Model
model_without_preprocessing = LinearRegression()
model_without_preprocessing.fit(X, Y)

# Predict
new_years = np.array([[11], [12]])
predictions_without_preprocessing = model_without_preprocessing.predict(new_years)

print("Predictions without preprocessing:", predictions_without_preprocessing)



                                                                  #With PreProcessing

import numpy as np
from sklearn.linear_model import LinearRegression

# Data
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
Y = np.array([30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000])

# Linear Regression Model
model_without_preprocessing = LinearRegression()
model_without_preprocessing.fit(X, Y)

# Predict
new_years = np.array([[11], [12]])
predictions_without_preprocessing = model_without_preprocessing.predict(new_years)

print("Predictions without preprocessing:", predictions_without_preprocessing)

