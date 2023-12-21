import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
import xgboost as xgb

# Load data
data = pd.read_csv('./Data/WineQT.csv')
X = data.iloc[:, 0:11]
y = data.iloc[:, 11]
print("Label Counts:")
print(y.value_counts())
y = pd.Series(LabelEncoder().fit_transform(y))

# Print the label counts
data.info()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=58)
print("Label Train Counts:")
print(y_train.value_counts())
print("Label Test Counts:")
print(y_test.value_counts())

# Define the parameter grid for XGBoost
param_grid = {
    'learning_rate': [0.1], #default [0.1]  [0.025 , 0.05 , 0.1 , 0.2 , 0.3]
    'max_depth': [2, 3, 7, 10, 20], #default [3]  [2, 3, 5, 7, 10, 100]
    'gamma': [0,0.1,0.2,0.4,1.0,2.0],  #default [0]   [0,0.1,0.2,0.3,0.4,1.0,1.5,2.0]
    'colsample_bylevel':[0.25, 1.0],  #default [1]  [log2,sqrt , 0.25, 1.0]
    'subsample':[0.15 , 0.5, 1.0],  #default [1]  [0.15 , 0.5 , 0.75 , 1.0]
    #'min_child_weight':[1],
}

# Use StratifiedKFold for cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=32)

# Initialize the XGBoost model
xgb_model = xgb.XGBClassifier(objective='MultiClass',n_estimators=150, early_stopping_rounds=40)

# Perform grid search with cross-validation
grid_search = GridSearchCV(xgb_model, param_grid, cv=cv, scoring='f1_weighted', verbose=3)

grid_search.fit(X_train, y_train, eval_set=[(X_test, y_test)],verbose=0)

# Display the best parameters and corresponding F1 score
print("Best Parameters:", grid_search.best_params_)
print("Best Weighted F1 Score:", grid_search.best_score_)

# Display the table of results
results = pd.DataFrame(grid_search.cv_results_)
columns_of_interest = ['param_learning_rate', 'param_max_depth', 'param_subsample', 'param_colsample_bylevel', 'param_gamma', 'mean_test_score', 'std_test_score']
results_table = results[columns_of_interest].sort_values(by='mean_test_score', ascending=False)
print("\nResults Table:")
print(results_table)

results_table.to_csv('grid_search_results_XGB_75.csv', index=False)
