import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import catboost as cat
import shap

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

## Baseline Model
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

X_train_scaled = X_train
X_test_scaled = X_test

# Define the parameter grid to search
param_grid = {
    # 'grow_policy': 'Lossguide',
    # 'num_leaves': [20],
    # 'iterations': [50, 100, 200, 300], # num_trees
    # 'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
    # 'max_depth': [4, 6, 8, 10, 12],
    # 'l2_leaf_reg': [0.001, 0.01, 0.1, 1, 10]
    'iterations': [200], # num_trees
    'learning_rate': [0.2],
    'max_depth': [4],
    'l2_leaf_reg': [0.01],
    'early_stopping_rounds': [20],
}


# Create the CatBoost model
cbr_model = cat.CatBoostClassifier(verbose=0, objective='MultiClass')

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(estimator=cbr_model, param_grid=param_grid, scoring='accuracy', cv=3, verbose=3)
grid_search.fit(X_train_scaled, y_train)

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Get the average cross-validation score for the best parameters
best_score = grid_search.best_score_
print(f'Average Cross-Validation Score: {best_score}')

# Train a new model with the best hyperparameters
best_model = cat.CatBoostClassifier(**grid_search.best_params_, objective='MultiClass')
print(best_model.get_params())
best_model.fit(X_train_scaled, y_train, eval_set=(X_test_scaled, y_test), early_stopping_rounds=20) # avoid overfitting

# Make predictions on the test set
y_pred = best_model.predict(X_test_scaled)
y_pred_train = best_model.predict(X_train_scaled)

# Evaluate the model
test_accuracy = accuracy_score(y_test, y_pred)
train_accuracy = accuracy_score(y_train, y_pred_train)
print(f'Test Accuracy with best hyperparameters: {test_accuracy}')
print(f'Training Accuracy with best hyperparameters: {train_accuracy}')

# Print Classification Report and Confusion Matrix
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature Importance
feature_importances = best_model.get_feature_importance()
plt.figure(figsize=(12, 6))
plt.bar(range(len(feature_importances)), feature_importances)
plt.xticks(range(len(feature_importances)), X.columns, rotation=90)
plt.title('Feature Importances')
plt.show()
plt.savefig('feature_importances.png')
plt.close()

# Model Interpretability (SHAP Values)
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_train_scaled)
shap.summary_plot(shap_values, X_train_scaled, feature_names=X.columns)

# Extract the evaluation history
eval_history = best_model.evals_result_

# Plot the learning curves
plt.figure(figsize=(10, 6))
plt.plot(eval_history['learn']['MultiClass'], label='Training loss')
plt.plot(eval_history['validation']['MultiClass'], label='Validation loss')
plt.title('Learning Curve')
plt.xlabel('Iterations')
plt.xlim(-5, 100)
plt.ylabel('MultiClass Loss')
plt.ylim(0 ,2)
plt.legend()
plt.show()
plt.savefig('learning_curve.png')
plt.close()