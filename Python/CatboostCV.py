import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score
import catboost as cat

# Load data
data = pd.read_csv('./Data/WineQT.csv')
X = data.iloc[:, 0:11]
y = data.iloc[:, 11]
print("Label Counts:")
print(y.value_counts())
y = pd.Series(LabelEncoder().fit_transform(y))

# Print the label counts
data.info()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=58)
print("Label Train Counts:")
print(y_train.value_counts())
print("Label Test Counts:")
print(y_test.value_counts())

# Define the parameter grid
param_grid = {
    'learning_rate': [0.1],
    'max_depth': [3, 5, 7],# [3, 5, 7],
    'l2_leaf_reg':[1,3,6,9], # [1,3,6,9],
    'leaf_estimation_iterations':[1, 10], #[1, 10],
}

# Create the CatBoost model
cbr_model = cat.CatBoostClassifier(verbose=0, objective='MultiClass', iterations=400)

# Use StratifiedKFold for cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=32)

# Perform grid search with cross-validation
grid_search = GridSearchCV(cbr_model, param_grid, cv=cv, scoring='f1_weighted', verbose=3)
grid_search.fit(X_train, y_train, early_stopping_rounds=50, verbose=0)

# Display the best parameters and corresponding F1 score
print("Best Parameters:", grid_search.best_params_)
print("Best Weighted F1 Score:", grid_search.best_score_)

# Display the table of results
results = pd.DataFrame(grid_search.cv_results_)
columns_of_interest = ['param_learning_rate', 'param_max_depth', 'param_l2_leaf_reg','param_leaf_estimation_iterations', 'mean_test_score', 'std_test_score']
results_table = results[columns_of_interest].sort_values(by='mean_test_score', ascending=False)
print("\nResults Table:")
print(results_table)
# Save the results table to a CSV file
results_table.to_csv('grid_search_results_CAT.csv', index=False)