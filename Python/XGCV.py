import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
# Load data
data = pd.read_csv('./WineQT.csv')
X = data.iloc[:, 0:11]
y = data.iloc[:, 11]
print("Label Counts:")
print(y.value_counts())
y = pd.Series(LabelEncoder().fit_transform(y))

# Print the label counts
data.info()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=58)
print("Label Train Counts:")
print(y_train.value_counts())
print("Label Test Counts:")
print(y_test.value_counts())

# Define the parameter grid for XGBoost
param_grid = {
    'learning_rate': [0.1, 0.01],
    # 'max_depth': [3, 5, 7],
    # 'min_child_weight': [1, 3, 6],
    # 'subsample': [0.5, 0.7, 1],
    # 'colsample_bytree': [0.5, 0.7, 1],
}

# Custom cross-validation with early stopping
def custom_cv(X_train, y_train, param_grid, num_boost_round=100, early_stopping_rounds=50):
    best_score = 0
    best_params = None

    for params in ParameterGrid(param_grid):
        scores = []
        for train_idx, test_idx in StratifiedKFold(n_splits=5, shuffle=True, random_state=32).split(X_train, y_train):
            X_train_fold, X_test_fold = X_train.iloc[train_idx], X_train.iloc[test_idx]
            y_train_fold, y_test_fold = y_train.iloc[train_idx], y_train.iloc[test_idx]

            model = XGBClassifier(**params, objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False)
            model.fit(X_train_fold, y_train_fold, early_stopping_rounds=early_stopping_rounds, eval_set=[(X_test_fold, y_test_fold)], verbose=False)
            predictions = model.predict(X_test_fold)
            scores.append(f1_score(y_test_fold, predictions, average='weighted'))

        avg_score = np.mean(scores)
        if avg_score > best_score:
            best_score = avg_score
            best_params = params

    return best_params, best_score

# Perform custom CV with early stopping
best_params, best_score = custom_cv(X_train, y_train, param_grid)
print("Best Parameters:", best_params)
print("Best Weighted F1 Score:", best_score)

# Fit the best model
best_model = XGBClassifier(**best_params, objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False)
best_model.fit(X_train, y_train, early_stopping_rounds=50, eval_set=[(X_test, y_test)], verbose=True)
