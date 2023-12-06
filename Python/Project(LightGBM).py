import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, classification_report

# Load data
data = pd.read_csv('./Data/WineQT.csv')
X = data.iloc[:, 0:11]
X=normalize(X, axis= 0)
y = data.iloc[:, 11] 
print(y.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the data to LightGBM dataset format
train_data = lgb.Dataset(X_train, label=y_train)

# Define the parameter grid to search
param_grid = {
    'max_depth': [-1]
}

# Create the LightGBM model
lgb_model = lgb.LGBMClassifier(verbose=-1,num_classes=6, objective='multiclass')

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(estimator=lgb_model, param_grid=param_grid, scoring='accuracy', cv=5)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Train a new model with the best hyperparameters
best_model = lgb.LGBMClassifier(**grid_search.best_params_,verbose=-1, num_classes=6, objective='multiclass')
best_model.fit(X_train, y_train)
best_model.get_params()
# Make predictions on the test set
y_pred = best_model.predict(X_test)
y_pred_train = best_model.predict(X_train)


# Evaluate the model
test_accuracy = accuracy_score(y_test, y_pred)
train_accuracy = accuracy_score(y_train, y_pred_train)
print(f'Test Accuracy with best hyperparameters: {test_accuracy}')
print(f'Training Accuracy with best hyperparameters: {train_accuracy}')