import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from sklearn.model_selection import learning_curve

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

# Create the XGBoost model
xgb_model = xgb.XGBClassifier(objective='multi:softmax', subsample=0.5)

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=5, verbose=3)
grid_search.fit(X_train, y_train)
best_score = grid_search.best_score_
print(f'Average Cross-Validation Score: {best_score}')

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Train a new model with the best hyperparameters
best_model = xgb.XGBClassifier(**grid_search.best_params_,verbose=-1, objective='multi:softmax', subsample=0.5)
# best_model.fit(X_train, y_train)
evals_result = {}  # Initialize an empty dictionary to store training history
best_model.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_metric="mlogloss",  # Set the evaluation metric for learning curves
    verbose=True,
)
best_model.get_params()

# Make predictions on the test set
y_pred = best_model.predict(X_test)
y_pred_train = best_model.predict(X_train)


# Get training history
results = best_model.evals_result()
train_metric = results['validation_0']['mlogloss']
valid_metric = results['validation_1']['mlogloss']

# Plot the learning curves
plt.figure(figsize=(10, 6))
iterations = np.arange(1, len(train_metric) + 1)
plt.plot(iterations, train_metric, label='Training Multi Log Loss')
plt.plot(iterations, valid_metric, label='Validation Multi Log Loss')
plt.title('Learning Curves')
plt.xlabel('Iterations')
plt.ylabel('Multi Log Loss')
plt.legend()
plt.show()