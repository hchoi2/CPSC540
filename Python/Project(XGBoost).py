import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from sklearn.model_selection import learning_curve
from sklearn.metrics import f1_score, recall_score, precision_score
import math
import seaborn as sns

# Load data
data = pd.read_csv('./python/WineQT.csv')
X = data.iloc[:, 0:11]
y = data.iloc[:, 11] 
y=LabelEncoder().fit_transform(y)
# print(y.head())



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid to search
param_grid = {
    # 'learning_rate': [0.025, 0.05, 0.1, 0.2, 0.3],
    'learning_rate': [0.025, 0.05],
    # 'gamma': [0, 0.1, 0.2, 0.3, 0.4, 1, 1.5, 2],
    'gamma': [1, 1.5, 2],
    # 'max_depth': [2, 3, 5, 7, 10, 100],
    'max_depth': [2, 3, 5]
    # 'colsample_bylevel': [math.log2,math.sqrt, 0.25, 1],
    # 'subsample':[0.15, 0.5, 0.75, 1]
    # 'n_estimators': [50],
    # 'eta' : [0.01],
    # 'lambda' : [0.1]
}

# Create the XGBoost model
xgb_model = xgb.XGBClassifier(objective='multi:softmax', )

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='f1_weighted', cv=5, verbose=3)
grid_search.fit(X_train, y_train, early_stopping_rounds=200)
best_score = grid_search.best_score_
print(f'Average Cross-Validation Score: {best_score}')

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Train a new model with the best hyperparameters
best_model = xgb.XGBClassifier(**grid_search.best_params_, objective='multi:softmax')
print(best_model.get_params())
# best_model.fit(X_train, y_train)
evals_result = {}  # Initialize an empty dictionary to store training history
best_model.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    #eval_metric="mlogloss",  # Set the evaluation metric for learning curves
)
best_model.get_params()

# Make predictions on the test set
y_pred = best_model.predict(X_test)
y_pred_train = best_model.predict(X_train)

# Get the ranked results
results_df = pd.DataFrame(grid_search.cv_results_)
ranked_results = results_df[['params', 'mean_test_score']].sort_values(by='mean_test_score', ascending=False)
print("\nRanked Results based on F1 Score:")
print(ranked_results)

# Extract relevant columns for the heatmap
heatmap_data = results_df.pivot(index='param_learning_rate', columns='param_max_depth', values='mean_test_score')

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt=".3f", cbar_kws={'label': 'Weighted F1 Score'})
plt.title('GridSearchCV Results Heatmap')
plt.xlabel('max_depth')
plt.ylabel('n_estimators')
plt.show()


# Get training history
# results = best_model.evals_result()
# train_metric = results['validation_0']['mlogloss']
# valid_metric = results['validation_1']['mlogloss']

# # Plot the learning curves
# plt.figure(figsize=(10, 6))
# iterations = np.arange(1, len(train_metric) + 1)
# plt.plot(iterations, train_metric, label='Training Multi Log Loss')
# plt.plot(iterations, valid_metric, label='Validation Multi Log Loss')
# plt.title('Learning Curves')
# plt.xlabel('Iterations')
# plt.ylabel('Multi Log Loss')
# plt.legend()
# plt.show()



# f1=f1_score(y_test, y_pred, average='micro') #Use 'macro' and 'weighted' for average too
# precision=precision_score(y_test, y_pred, average='micro') #Use 'macro' and 'weighted' for average too
# recall=recall_score(y_test, y_pred, average='micro') #Use 'macro' and 'weighted' for average too
# print("F1:", f1)
# print("precision:", precision)
# print("recall:", recall)