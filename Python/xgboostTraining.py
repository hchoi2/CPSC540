import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, ParameterGrid
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import catboost as cat
import xgboost as xgb
import shap


def custom_f1_score(preds, dmatrix):
    y_true = dmatrix.get_label()
    y_pred = np.argmax(preds, axis=1)
    f1 = f1_score(y_true, y_pred, average='micro')  # 'micro' or 'macro' depending on your preference
    return 'f1_score', f1




# Load data
data = pd.read_csv('./Data/WineQT.csv')
X = data.iloc[:, 0:11]
y = data.iloc[:, 11]
print("Label Counts:")
print(y.value_counts())
#y=LabelEncoder().fit_transform(y)
y= pd.Series(LabelEncoder().fit_transform(y))


# Print the label counts
data.info()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=58)
# print("Label Train Counts:")
# print(y_train.value_counts())
# print("Label Test Counts:")
# print(y_test.value_counts())

# Baseline Model
most_frequent_class = y_train.mode()[0]
baseline_predictions = np.full(shape=y_test.shape, fill_value=most_frequent_class)
baseline_accuracy = accuracy_score(y_test, baseline_predictions)
baseline_error = 1 - baseline_accuracy

print(f'Baseline Accuracy: {baseline_accuracy}')
print(f'Baseline Error Rate: {baseline_error}')

X_train_scaled = X_train
X_test_scaled = X_test

param_grid = {
    'learning_rate': [0.025 , 0.05 , 0.1 , 0.2], #default [0.1]  [0.025 , 0.05 , 0.1 , 0.2 , 0.3]
    #'max_depth': [3,4,5], #default [3]  [2, 3, 5, 7, 10, 100]
    #'gamma': [0],  #default [0]   [0,0.1,0.2,0.3,0.4,1.0,1.5,2.0]
    #'colsample_bylevel':[1],  #default [1]  [log2,sqrt , 0.25, 1.0]
    #'subsample':[1],  #default [1]  [0.15 , 0.5 , 0.75 , 1.0]
    #'min_child_weight':[1],
}

# param_grid = [
#     {'learning_rate': [0.1], 'max_depth': [3], 'gamma': [0], 'colsample_bylevel': [1], 'subsample': [1]},
#     {'learning_rate': [0.2], 'max_depth': [4], 'gamma': [0.1], 'colsample_bylevel': [0.8], 'subsample': [0.9]},
#     {'learning_rate': [0.05], 'max_depth': [5], 'gamma': [0.2], 'colsample_bylevel': [0.6], 'subsample': [0.8]}
# ]
#cbr_model = cat.CatBoostClassifier(verbose=0, objective='MultiClass')

# Use StratifiedKFold for cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=32)

# Create empty lists to store training and validation losses
train_losses = []
val_losses = []
train_f1 = []
val_f1 = []

# Iterate over hyperparameter combinations
for params in ParameterGrid(param_grid):
    print("Testing hyperparameters:", params)

    # Create the CatBoost model with current hyperparameters
    trainModel = xgb.XGBClassifier(objective='MultiClass', n_estimators= 400)
    trainModel.set_params(**params)

    # Lists to store raw losses for each fold
    fold_train_losses = []
    fold_val_losses = []
    fold_train_f1 = []
    fold_val_f1 = []

    # Perform cross-validation
    for train_index, val_index in cv.split(X_train_scaled, y_train):
        X_train_fold, X_val_fold = X_train_scaled.iloc[train_index], X_train_scaled.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]


        # Fit the model[(X_train, y_train), (X_test, y_test)]
        trainModel.fit(X_train_fold, y_train_fold, eval_set=[(X_train_fold, y_train_fold),(X_val_fold, y_val_fold)],eval_metric=custom_f1_score,verbose=0)

        # Get the evaluation history for the current fold
        eval_history = trainModel.evals_result()

        # Collect training and validation losses for each iteration
        fold_train_losses.append(eval_history['validation_0']['mlogloss'])
        fold_val_losses.append(eval_history['validation_1']['mlogloss'])
        fold_train_f1.append(eval_history['validation_0']['f1_score'])
        fold_val_f1.append(eval_history['validation_1']['f1_score'])


    # Interpolate to have the same length
    min_len = min(len(history) for history in fold_train_losses)
    fold_train_losses = [np.interp(range(min_len), range(len(history)), history) for history in fold_train_losses]
    fold_val_losses = [np.interp(range(min_len), range(len(history)), history) for history in fold_val_losses]

    min_len = min(len(history) for history in fold_train_f1)
    fold_train_f1 = [np.interp(range(min_len), range(len(history)), history) for history in fold_train_f1]
    fold_val_f1 = [np.interp(range(min_len), range(len(history)), history) for history in fold_val_f1]

    # Append raw losses to the overall lists
    train_losses.append(fold_train_losses)
    val_losses.append(fold_val_losses)

    train_f1.append(fold_train_f1)
    val_f1.append(fold_val_f1)

# Calculate average training and validation losses across folds for each iteration
#avg_train_losses = np.mean(np.array(train_losses), axis=-1)


avg_train_losses= [np.mean(np.array(sublist), axis=0) for sublist in train_losses]
avg_val_losses= [np.mean(np.array(sublist), axis=0) for sublist in val_losses]

avg_train_f1 = [np.mean(np.array(sublist), axis=0) for sublist in train_f1]
avg_val_f1 = [np.mean(np.array(sublist), axis=0) for sublist in val_f1]


# Plot the learning curves losses for each set of hyperparameters
plt.figure(figsize=(5, 5))
for i, params in enumerate(ParameterGrid(param_grid)):
    plt.plot(avg_train_losses[i], label=f'Train_Loss - {params}')
    plt.plot(avg_val_losses[i], label=f'Val_Loss - {params}')

plt.title('Learning Curves: XGBoost')
plt.xlabel('Iterations')
plt.ylabel('MultiClass Loss')
#plt.legend(loc='Lower right')
current_timestamp = datetime.now()
timestamp_str = current_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
plt.savefig(f"XGB_learning_curves_loss_Final_{timestamp_str}.png")
plt.show()
# Plot the learning curves F1 for each set of hyperparameters
plt.figure(figsize=(5, 5))
for i, params in enumerate(ParameterGrid(param_grid)):
    plt.plot(avg_train_f1[i], label=f'Train_Loss - {params}')
    plt.plot(avg_val_f1[i], label=f'Val_Loss - {params}')

plt.title('Learning Curves: XGBoost')
plt.xlabel('Iterations')
plt.ylabel('F1')
#plt.legend(loc='Lower right')
current_timestamp = datetime.now()
timestamp_str = current_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
plt.savefig(f"XGB_learning_curves_f1_Final_{timestamp_str}.png")
plt.show()

