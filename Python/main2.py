import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, make_scorer, precision_score, \
    recall_score, f1_score
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


# Function for cross-validation results
def print_cross_val_results(model, X, y):
    scoring_metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    for metric in scoring_metrics:
        cv_results = cross_val_score(model, X, y, cv=5, scoring=metric)
        print(f"{metric}: {cv_results.mean()}")

    print(f"--- Cross-Validation Results for {type(model).__name__} ---")
    print("\n")

# Function to plot feature importance
def plot_feature_importance(importance, names, model_type):
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    plt.figure(figsize=(10, 8))
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    plt.title(f'{model_type} - Feature Importance')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Names')
    plt.show()

# Load your dataset (replace 'your_dataset.csv' with your actual dataset)
data = pd.read_csv('./Data/WineQT.csv')
X = data.iloc[:, 0:11]
y = data.iloc[:, 11]
y=LabelEncoder().fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
gbm_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
lgb_model = lgb.LGBMClassifier(n_estimators=100, verbose=-1, learning_rate=0.1, max_depth=3, random_state=42)
catboost_model = catboost.CatBoostClassifier(iterations=100, learning_rate=0.1, depth=3, random_state=42, verbose=0)
xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Train models
random_forest_model.fit(X_train, y_train)
gbm_model.fit(X_train, y_train)
lgb_model.fit(X_train, y_train)
catboost_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# Make predictions
rf_predictions = random_forest_model.predict(X_test)
gbm_predictions = gbm_model.predict(X_test)
lgb_predictions = lgb_model.predict(X_test)
catboost_predictions = catboost_model.predict(X_test)
xgb_predictions = xgb_model.predict(X_test)

# Evaluate models
models = ['Random Forest', 'GBM', 'LightGBM', 'CatBoost', 'XGBoost']
predictions = [rf_predictions, gbm_predictions, lgb_predictions, catboost_predictions, xgb_predictions]

for model, preds in zip(models, predictions):
    accuracy = accuracy_score(y_test, preds)
    classification_report_str = classification_report(y_test, preds)
    confusion_mat = confusion_matrix(y_test, preds)

    print(f"--- {model} ---")
    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_report_str)
    print("Confusion Matrix:\n", confusion_mat)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y),
                yticklabels=np.unique(y))
    plt.title(f'Confusion Matrix - {model}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Plot learning curves for each model
for model, clf in zip(models, [random_forest_model, gbm_model, lgb_model, catboost_model, xgb_model]):
    plt = plot_learning_curve(clf, f"Learning Curve - {model}", X_train, y_train)
    plt.show()

# Feature Importance Plots for Tree-based Models
# Random Forest Feature Importance
plot_feature_importance(random_forest_model.feature_importances_, X.columns, 'Random Forest')
plt.show()

# Gradient Boosting Feature Importance
plot_feature_importance(gbm_model.feature_importances_, X.columns, 'GBM')
plt.show()

# LightGBM Feature Importance
plot_feature_importance(lgb_model.feature_importances_, X.columns, 'LightGBM')
plt.show()

# CatBoost Feature Importance
plot_feature_importance(catboost_model.feature_importances_, X.columns, 'CatBoost')
plt.show()

# XGBoost Feature Importance
plot_feature_importance(xgb_model.feature_importances_, X.columns, 'XGBoost')
plt.show()


# Cross-validation results for each model
for model, clf in zip(models, [random_forest_model, gbm_model, lgb_model, catboost_model, xgb_model]):
    print_cross_val_results(clf, X, y)