 #Regression Task 
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the California Housing dataset
california = fetch_california_housing()
X = california.data
y = california.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

def build_regression_model(optimizer='adam', neurons=32):
    model = Sequential()
    model.add(Dense(neurons, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss='mse')
    return model

# Create the KerasRegressor
regressor = KerasRegressor(build_fn=build_regression_model, verbose=0)

from sklearn.model_selection import GridSearchCV

# Define the hyperparameters grid
param_grid = {
    'batch_size': [10, 20, 40],
    'epochs': [50, 100, 150],
    'optimizer': ['adam', 'rmsprop'],
    'neurons': [16, 32, 64]
}

# Perform grid search
grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3)
grid_result = grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best parameters found: ", grid_result.best_params_)


# Evaluate the best model on the test set
best_model = grid_result.best_estimator_
test_mse = best_model.score(X_test, y_test)
print("Test MSE: ", -test_mse)



#Classification Task
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelBinarizer

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Binarize the labels for classification
lb = LabelBinarizer()
y = lb.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def build_classification_model(optimizer='adam', neurons=32):
    model = Sequential()
    model.add(Dense(neurons, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create the KerasClassifier
classifier = KerasClassifier(build_fn=build_classification_model, verbose=0)

# Define the hyperparameters grid
param_grid = {
    'batch_size': [10, 20, 40],
    'epochs': [50, 100, 150],
    'optimizer': ['adam', 'rmsprop'],
    'neurons': [16, 32, 64]
}

# Perform grid search
grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, scoring='accuracy', cv=3)
grid_result = grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best parameters found: ", grid_result.best_params_)
# Evaluate the best model on the test set
best_model = grid_result.best_estimator_
test_accuracy = best_model.score(X_test, y_test)
print("Test Accuracy: ", test_accuracy)