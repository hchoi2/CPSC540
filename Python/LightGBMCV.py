import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb

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

# Define the parameter grid for LightGBM
param_grid = {
    'learning_rate': [0.025, 0.05, 0.1, 0.2, 0.3],
    'max_depth': [-1],
    'num_leaves': [3, 7, 15, 31, 127, 1024],
    'feature_fraction_bynode': [ 0.25, 1.0], #0.31, 0.3,
}

# Use StratifiedKFold for cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=32)

# Initialize the LightGBM model
lgb_model = lgb.LGBMClassifier(verbose=-1,objective='multiclass', metric='multi_logloss', n_iterations=400)

# Perform grid search with cross-validation
grid_search = GridSearchCV(lgb_model, param_grid, cv=cv, scoring='f1_weighted', verbose=3)

# Add early stopping callback
callbacks = [lgb.early_stopping(75)]

grid_search.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=callbacks)

# Display the best parameters and corresponding F1 score
print("Best Parameters:", grid_search.best_params_)
print("Best Weighted F1 Score:", grid_search.best_score_)

# Display the table of results
results = pd.DataFrame(grid_search.cv_results_)
columns_of_interest = ['param_learning_rate', 'param_max_depth', 'param_feature_fraction_bynode', 'param_num_leaves', 'mean_test_score', 'std_test_score']
results_table = results[columns_of_interest].sort_values(by='mean_test_score', ascending=False)
print("\nResults Table:")
print(results_table)

# Save the results table to a CSV file
results_table.to_csv('grid_search_results_LGBM_75.csv', index=False)