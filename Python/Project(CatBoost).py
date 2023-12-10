import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier as abc
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
# from class sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier as cbr
from sklearn.preprocessing import StandardScaler
# from imblearn.pipeline import Pipeline as imbpipeline
# from scipy.stats import randint 
# from imblearn.over_sampling import SMOTE
# from collections import Counter
# from sklearn.base import clone
# Load data
data = pd.read_csv('./Data/WineQT.csv')
X = data.iloc[:, 0:11]
y = data.iloc[:, 11] 
print(y.head())

# # Determine the number of unique classes
# num_classes = y.nunique()
# print("Number of unique classes:", num_classes)

#####baseline####
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Determine the most frequent class in the training set
most_frequent_class = y_train.mode()[0]
# Predict this class for all instances in the test set
baseline_predictions = np.full(shape=y_test.shape, fill_value=most_frequent_class)
# Calculate the accuracy of the baseline
baseline_accuracy = accuracy_score(y_test, baseline_predictions)
# Calculate the error rate of the baseline
baseline_error = 1 - baseline_accuracy

print(f'Baseline Accuracy: {baseline_accuracy}')
print(f'Baseline Error Rate: {baseline_error}')
##################

# Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Note: we transform the test set with the parameters learned from the training set



# Convert the data to CatBoost dataset format
# train_data = cbr.Dataset(X_train, label=y_train)

# Define the parameter grid to search
param_grid = {
    # 'grow_policy': 'Lossguide', 
    # 'num_leaves': [20, 40, 60, 80, 100],
    'max_depth': [6, 10],
    'learning_rate': [0.01, 0.1],
    # 'l2_leaf_reg': [0.001, 0.01, 0.1],
    'iterations': [50, 100], # num_trees

}
# 'l2_leaf_reg': 0.01, 'learning_rate': 0.1, 'max_depth': 8, 'num_trees': 50

# Create the CatBoost model
cbr_model = cbr(verbose=0, objective='MultiClass')

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(estimator=cbr_model, param_grid=param_grid, scoring='accuracy', cv=10, verbose=3)
grid_search.fit(X_train_scaled, y_train)

# Use RandomizedSearchCV to find the best hyperparameters
# random_search = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid, n_iter=100, scoring='accuracy', cv=5, verbose=3, random_state=42)
# random_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Get the average cross-validation score for the best parameters
best_score = grid_search.best_score_
print(f'Average Cross-Validation Score: {best_score}')

# Train a new model with the best hyperparameters
best_params = grid_search.best_params_.copy()
best_params.update({'verbose': 0, 'objective': 'MultiClass', 'early_stopping_rounds': 20}) # avoid overfitting
best_model = cbr(**best_params)
best_model = cbr(**grid_search.best_params_, verbose=0, objective='MultiClass')
best_model.fit(X_train_scaled, y_train, eval_set=(X_test, y_test), early_stopping_rounds=20)

# # Best model from random search
# best_model = random_search.best_estimator_
# # Fit the best model from random search with early stopping
# best_model.named_steps['classifier'].fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=20)
# best_model.get_params()

# Make predictions on the test set
y_pred = best_model.predict(X_test_scaled)
y_pred_train = best_model.predict(X_train)

# Evaluate the model
test_accuracy = accuracy_score(y_test, y_pred)
train_accuracy = accuracy_score(y_train, y_pred_train)
print(f'Test Accuracy with best hyperparameters: {test_accuracy}')
print(f'Training Accuracy with best hyperparameters: {train_accuracy}')