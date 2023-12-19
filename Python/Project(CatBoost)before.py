import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier as abc
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier as cbr

# Load data
data = pd.read_csv('./Data/WineQT.csv')
X = data.iloc[:, 0:11]
y = data.iloc[:, 11] 
print(y.head())

# # Determine the number of unique classes
# num_classes = y.nunique()
# print("Number of unique classes:", num_classes)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

###
# Convert the data to CatBoost dataset format
# train_data = cbr.Dataset(X_train, label=y_train)

# Define the parameter grid to search
param_grid = {
    'max_depth': [4, 6]
}

# Create the CatBoost model
cbr_model = cbr(verbose=0, objective='MultiClass')

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(estimator=cbr_model, param_grid=param_grid, scoring='accuracy', cv=5)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Train a new model with the best hyperparameters
best_model = cbr(**grid_search.best_params_, verbose=0, objective='MultiClass')
best_model.fit(X_train, y_train)
best_model.get_params()
# Make predictions on the test set
y_pred = best_model.predict(X_test)
y_pred_train = best_model.predict(X_train)

# Evaluate the model
test_accuracy = accuracy_score(y_test, y_pred)
train_accuracy = accuracy_score(y_train, y_pred_train)
print(f'Test Accuracy with best hyperparameters: {test_accuracy}')
print(f'Training Accuracy with best hyperparameters: {train_accuracy}')