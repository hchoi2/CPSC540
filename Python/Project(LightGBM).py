import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score

# Load data
# data = pd.read_csv('./Python/Data/WineQT.csv')


def Light_wrapper(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # X_train,y_train=sm.fit_resample(X_train,y_train) --- 62 % accuracy, it's worse

    # Convert the data to LightGBM dataset format
    train_data = lgb.Dataset(X_train, label=y_train)

    # Define the parameter grid to search
    param_grid = {
        'learning_rate': [0.02],  # Boosting learning rate.
        'num_leaves': [40],
        # 'bagging_fraction': [0.8]
    }

    # Create the LightGBM model
    lgb_model = lgb.LGBMClassifier(verbose=-1, objective='multiclass')

    # Use GridSearchCV to find the best hyperparameters
    grid_search = GridSearchCV(estimator=lgb_model, param_grid=param_grid, scoring='accuracy', cv=10, verbose=3)
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters
    print("Best Hyperparameters:", grid_search.best_params_)
    # Get the average cross-validation score for the best parameters
    best_score = grid_search.best_score_
    print(f'Average Cross-Validation Score: {best_score}')

    # Train a new model with the best hyperparameters
    best_model = lgb.LGBMClassifier(**grid_search.best_params_, verbose=-1, objective='multiclass', n_estimators=100)
    best_model.fit(X_train, y_train)
    print(best_model.get_params())
    # Make predictions on the test set
    y_pred = best_model.predict(X_test)
    y_pred_train = best_model.predict(X_train)

    # Evaluate the model
    test_accuracy = accuracy_score(y_test, y_pred)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    print(f'Test Accuracy with best hyperparameters: {test_accuracy}')
    print(f'Training Accuracy with best hyperparameters: {train_accuracy}')
    print('Validation Score -> ', metrics.cohen_kappa_score(y_pred, y_test, weights='quadratic'))




    best_model = lgb.LGBMClassifier(**grid_search.best_params_, verbose=-1, objective='multiclass', n_estimators=100)
    best_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)])

    # Get training history
    evals_result = best_model.evals_result_

    # Plot the learning curves
    train_metric = evals_result['training']['multi_logloss']
    valid_metric = evals_result['valid_1']['multi_logloss']

    plt.figure(figsize=(10, 6))
    iterations = np.arange(1, len(train_metric) + 1)
    plt.plot(iterations, train_metric, label='Training Multi Log Loss')
    plt.plot(iterations, valid_metric, label='Validation Multi Log Loss')
    plt.title('Learning Curves')
    plt.xlabel('Iterations')
    plt.ylabel('Multi Log Loss')
    plt.legend()
    plt.show()

    f1 = f1_score(y_test, y_pred, average='macro')  # Use 'macro' and 'weighted' for average too
    #precision = precision_score(y_test, y_pred, average='macro')  # Use 'macro' and 'weighted' for average too
    recall = recall_score(y_test, y_pred, average='macro')  # Use 'macro' and 'weighted' for average too
    print("F1:", f1)
    #print("precision:", precision)
    print("recall:", recall)

    f1 = f1_score(y_test, y_pred, average='weighted')  # Use 'macro' and 'weighted' for average too
    #precision = precision_score(y_test, y_pred, average='weighted')  # Use 'macro' and 'weighted' for average too
    recall = recall_score(y_test, y_pred, average='weighted')  # Use 'macro' and 'weighted' for average too
    print("F1:", f1)
    #print("precision:", precision)
    print("recall:", recall)


data = pd.read_csv('./Data/WineQT.csv')
X = data.iloc[:, 0:11]
y = data.iloc[:, 11]
Light_wrapper(X, y)
