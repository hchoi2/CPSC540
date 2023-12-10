import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# Load data
data = pd.read_csv('./python/WineQT.csv')
X = data.iloc[:, 0:11]
y = data.iloc[:, 11] 
y=LabelEncoder().fit_transform(y)
# print(y.head())



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid to search
param_grid = {
    'max_depth': [4, 6, 8],
    'n_estimators': [50, 100, 200],
    'eta' : [0.01, 0.1, 0.2],
    'lambda' : [0.1, 1, 10]
}

# Create the LightGBM model
xgb_model = xgb.XGBClassifier(objective='multi:softmax')

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=5, verbose=3)
grid_search.fit(X_train, y_train)
best_score = grid_search.best_score_
print(f'Average Cross-Validation Score: {best_score}')

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Train a new model with the best hyperparameters
best_model = xgb.XGBClassifier(**grid_search.best_params_,verbose=-1, objective='multi:softmax')
best_model.fit(X_train, y_train)
best_model.get_params()

# Make predictions on the test set
y_pred = best_model.predict(X_test)
y_pred_train = best_model.predict(X_train)


# Evaluate the model
# test_accuracy = accuracy_score(y_test, y_pred)
train_accuracy = accuracy_score(y_train, y_pred_train)
# print(f'Test Accuracy with best hyperparameters: {test_accuracy}')
print(f'Training Accuracy with best hyperparameters: {train_accuracy}')