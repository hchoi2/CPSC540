import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, classification_report,  cohen_kappa_score, f1_score

from sklearn import metrics
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
# from imblearn.over_sampling import SMOTE
#sm = SMOTE(sampling_strategy='auto', random_state=42)


# Load data
#data = pd.read_csv('./Python/Data/WineQT.csv')
data = pd.read_csv('./Data/WineQT.csv')
X = data.iloc[:, 0:11]
#X=normalize(X, axis= 0)
y = data.iloc[:, 11] 
print(y.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

# Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# # Apply SMOTE for handling class imbalance
# smote = SMOTE(random_state=42)
# X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# Convert the data to LightGBM dataset format
train_data = lgb.Dataset(X_train_scaled, label=y_train)

# Define the parameter grid to search
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    # 'num_leaves': [10, 20, 30, 40, 50, 60],
    'max_depth': [4, 6, 8, 10, 12],
    'n_estimators': [50, 100, 200, 300],
    'lambda' : [0.001, 0.01, 0.1, 1, 10]
}

# Create the LightGBM model
lgb_model = lgb.LGBMClassifier(verbose=-1, objective='multiclass')

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(estimator=lgb_model, param_grid=param_grid, scoring='accuracy', cv=5, verbose=3)
grid_search.fit(X_train_scaled, y_train)

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)
# Get the average cross-validation score for the best parameters
best_score = grid_search.best_score_
print(f'Average Cross-Validation Score: {best_score}')

# Train a new model with the best hyperparameters
best_model = lgb.LGBMClassifier(**grid_search.best_params_,verbose=-1, objective='multiclass')
best_model.fit(X_train_scaled, y_train)
best_model.get_params()
# Make predictions on the test set
y_pred = best_model.predict(X_test_scaled)
y_pred_train = best_model.predict(X_train_scaled)


# Evaluate the model
test_accuracy = accuracy_score(y_test, y_pred)
train_accuracy = accuracy_score(y_train, y_pred_train)
kappa_score = cohen_kappa_score(y_pred, y_test, weights='quadratic')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Test Accuracy with best hyperparameters: {test_accuracy}')
print(f'Training Accuracy with best hyperparameters: {train_accuracy}')
print('Cohen Kappa Score: ', kappa_score)
print('F1 Score: ', f1)