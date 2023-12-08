import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier as abc
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Load data
data = pd.read_csv('./Data/WineQT.csv')
X = data.iloc[:, 0:11]
y = data.iloc[:, 11] 
print(y.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

####
# Choose a weak learner (e.g., decision tree)
weak_learner = DecisionTreeClassifier(max_depth=1)

# Create an AdaBoostClassifier
ada_boost = abc(base_estimator=weak_learner, n_estimators=50)

# Train the AdaBoost model
ada_boost.fit(X_train, y_train)

# Make predictions
predictions = ada_boost.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")

###
# Define the parameter grid to search
param_grid = {
    'learning_rate': [0.01, 0.1, 1]
}

# Create the AdaBoost model
# abc_model = abc(verbose=-1,num_classes=6, objective='multiclass')
abc_model = abc(base_estimator=DecisionTreeClassifier(), random_state=42)

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(estimator=abc_model, param_grid=param_grid, scoring='accuracy', cv=5)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Train a new model with the best hyperparameters
best_model = abc(**grid_search.best_params_)
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

