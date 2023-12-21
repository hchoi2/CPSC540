import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

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

# Define the parameter grid for Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 150],  # Number of trees in the forest
    'max_depth': [3, 5, 7],           # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],    # Minimum number of samples required to be at a leaf node
}

# Create the Random Forest model
rf_model = RandomForestClassifier(random_state=58)

# Use StratifiedKFold for cross-validation
cv_rf = StratifiedKFold(n_splits=5, shuffle=True, random_state=32)

# Perform grid search with cross-validation
grid_search_rf = GridSearchCV(rf_model, param_grid_rf, cv=cv_rf, scoring='f1_weighted', verbose=3)
grid_search_rf.fit(X_train, y_train)

# Display the best parameters and corresponding F1 score
print("Best Parameters (Random Forest):", grid_search_rf.best_params_)
print("Best Weighted F1 Score (Random Forest):", grid_search_rf.best_score_)

# Display the table of results
results_rf = pd.DataFrame(grid_search_rf.cv_results_)
columns_of_interest_rf = ['param_n_estimators', 'param_max_depth', 'param_min_samples_split', 'param_min_samples_leaf', 'mean_test_score', 'std_test_score']
results_table_rf = results_rf[columns_of_interest_rf].sort_values(by='mean_test_score', ascending=False)
print("\nResults Table (Random Forest):")
print(results_table_rf)

# Save the results table to a CSV file
results_table_rf.to_csv('grid_search_results_RF.csv', index=False)
