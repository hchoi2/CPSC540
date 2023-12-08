import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier as abc
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import classification_report

# Load data
data = pd.read_csv('/Users/hyejeongchoi/Desktop/hchoi_homepage/hchoi2/Python/Data/WineQT.csv')
X = data.iloc[:, 0:11]
y = data.iloc[:, 11] 
print(y.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models
abc_model = abc(base_estimator=DecisionTreeClassifier(), random_state=42)
xgb_model = xgb.XGBClassifier(random_state=42)
lgb_model = lgb.LGBMClassifier(random_state=42)

# Define hyperparameter grids (example, you can expand this)
abc_params = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]}
xgb_params = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1], 'max_depth': [3, 5, 7], 'min_child_weight': [1, 3, 5]}
lgb_params = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1], 'max_depth': [3, 5, 7], 'subsample': [0.6, 0.8, 1.0], 'feature_fraction': [0.6, 0.8, 1.0]}

# Perform GridSearchCV to find best hyperparameters (example for one model)
grid_search_abc = GridSearchCV(abc_model, abc_params, cv=3, scoring='accuracy')
grid_search_abc.fit(X_train, y_train)

grid_search_xgb = GridSearchCV(xgb_model, xgb_params, cv=3, scoring='accuracy')
grid_search_xgb.fit(X_train, y_train)

grid_search_lgb = GridSearchCV(lgb_model, lgb_params, cv=3, scoring='accuracy')
grid_search_lgb.fit(X_train, y_train)

# Train and evaluate models

# Best parameters and model evaluation
print("Best parameters for AdaBoost: ", grid_search_abc.best_params_)
best_abc_model = grid_search_abc.best_estimator_
y_pred = best_abc_model.predict(X_test)
print("Accuracy of AdaBoost: ", accuracy_score(y_test, y_pred))

print("Best parameters for XGBoost: ", grid_search_xgb.best_params_)
best_xgb_model = grid_search_xgb.best_estimator_
y_pred = best_xgb_model.predict(X_test)
print("Accuracy of XGBoost: ", accuracy_score(y_test, y_pred))

print("Best parameters for LightGBM: ", grid_search_lgb.best_params_)
best_lgb_model = grid_search_lgb.best_estimator_
y_pred = best_lgb_model.predict(X_test)
print("Accuracy of LightGBM: ", accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))