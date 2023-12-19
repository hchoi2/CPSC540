import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import catboost
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Function to plot learning curves
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure(figsize=(8, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Number of Trees")
    plt.ylabel("Score")

    # Vary the number of trees and observe how the score changes
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='accuracy')

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# Load your dataset (replace 'your_dataset.csv' with your actual dataset)
data = pd.read_csv('./Data/WineQT.csv')
X = data.iloc[:, 0:11]
y = data.iloc[:, 11]
y = LabelEncoder().fit_transform(y)

# Define the models and parameter grids
models = [
    ('Random Forest', RandomForestClassifier(), {'n_estimators': [50, 100, 150]}),
    ('GBM', GradientBoostingClassifier(), {'n_estimators': [50, 100, 150]}),
    ('LightGBM', lgb.LGBMClassifier(), {'n_estimators': [50, 100, 150]}),
    ('CatBoost', catboost.CatBoostClassifier(silent=True), {'iterations': [50, 100, 150]}),
    ('XGBoost', xgb.XGBClassifier(), {'n_estimators': [50, 100, 150]})
]

for model_name, model, param_grid in models:
    # GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)

    # Plot learning curve
    plot_learning_curve(grid_search.best_estimator_, f"Learning Curve - {model_name}", X, y)
    plt.show()

    # Display the best parameters and their corresponding score
    print(f"Best Parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best Cross-Validation Accuracy for {model_name}: {grid_search.best_score_}")
    print("\n")