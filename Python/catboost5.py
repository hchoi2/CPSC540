import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid
import catboost as cat

# Load data
data = pd.read_csv('./Data/WineQT.csv')
X = data.iloc[:, 0:11]
y = data.iloc[:, 11]
print("Label Counts:")
print(y.value_counts())

# Print the label counts
data.info()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=58)
print("Label Train Counts:")
print(y_train.value_counts())
print("Label Test Counts:")
print(y_test.value_counts())

# Baseline Model
most_frequent_class = y_train.mode()[0]
baseline_predictions = np.full(shape=y_test.shape, fill_value=most_frequent_class)
baseline_accuracy = np.mean(y_test == baseline_predictions)  # Use np.mean for baseline accuracy
baseline_error = 1 - baseline_accuracy

print(f'Baseline Accuracy: {baseline_accuracy}')
print(f'Baseline Error Rate: {baseline_error}')

X_train_scaled = X_train
X_test_scaled = X_test

param_grid = {
    'iterations': [200],
    'learning_rate': [0.2, 0.1],  # Add another possibility
    'max_depth': [4],
    'l2_leaf_reg': [0.01],
    'early_stopping_rounds': [20],
}

# Use StratifiedKFold for cross-validation
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Create empty lists to store training and validation losses
train_losses = []
val_losses = []

# Iterate over hyperparameter combinations
for params in ParameterGrid(param_grid):
    print("Testing hyperparameters:", params)

    # Lists to store raw losses for each fold
    fold_train_losses = []
    fold_val_losses = []

    # Perform cross-validation
    for train_index, val_index in cv.split(X_train_scaled, y_train):
        X_train_fold, X_val_fold = X_train_scaled.iloc[train_index], X_train_scaled.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        # Create a new CatBoost model with current hyperparameters
        cbr_model = cat.CatBoostClassifier(**params, verbose=0, objective='MultiClass')

        # Fit the model
        cbr_model.fit(X_train_fold, y_train_fold, eval_set=(X_val_fold, y_val_fold), early_stopping_rounds=20, verbose=0)

        # Get the evaluation history for the current fold
        eval_history = cbr_model.evals_result_

        # Collect training and validation losses for each iteration
        fold_train_losses.append(eval_history['learn']['MultiClass'])
        fold_val_losses.append(eval_history['validation']['MultiClass'])

    # Append raw losses to the overall lists
    train_losses.append(fold_train_losses)
    val_losses.append(fold_val_losses)

# Calculate the average learning curve across all folds for each set of hyperparameters
avg_train_losses = np.mean(np.array(train_losses), axis=(0, 1))
min_len = min(len(history) for history in avg_train_losses)

# Plot the average learning curve
plt.figure(figsize=(10, 6))
plt.plot(avg_train_losses[:, :min_len].T, label='Average Training Loss')
plt.title('Average Training Loss Curve for Different Hyperparameter Combinations')
plt.xlabel('Iterations')
plt.ylabel('MultiClass Loss')
plt.legend()
plt.show()
