import numpy as np
import pandas as pd
import math


import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="darkgrid")
sns.set_palette(sns.color_palette("viridis"))

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold, KFold

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm
tqdm.pandas()

import warnings
warnings.filterwarnings('ignore')
print("\n\nSelecting Best Model For Classification task:\n")

df = pd.read_csv('./Data/WineQT.csv')
df.columns = [c.lower().replace(' ', '_') for c in df.columns]
df['quality'] = df['quality'].apply(lambda x: x - 3)

# Drop Duplicates
# df.drop_duplicates(subset=['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
#       'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
#       'ph', 'sulphates', 'alcohol', 'quality'], keep='first', inplace=True)
# df.reset_index(inplace=True)
# df.drop(columns = ['index'], inplace=True)

# Drop Outliers
# df = df[(df['residual_sugar']<=7) & (df['chlorides']<=0.4)]
# df.reset_index(inplace=True)
# df.drop(columns = ['index'], inplace=True)

# Balance data
#df = balance_dataset(df, 'quality')

features = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
            'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
            'ph', 'sulphates', 'alcohol']
X = df[features]
y = df['quality']

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scoring = "accuracy"

models = [
    ("Logistic Regression", LogisticRegression(random_state=42)),
    ("Decision Tree", DecisionTreeClassifier(random_state=42)),
    ("Random Forest", RandomForestClassifier(random_state=42)),
    ("Gradient Boosting", GradientBoostingClassifier(random_state=42)),
    ("SVM", SVC(random_state=42)),
    ("K-Nearest Neighbors", KNeighborsClassifier()),
    ("Naive Bayes", GaussianNB())]

results = []
model_names = []

for name, model in models:
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    results.append(scores)
    model_names.append(name)
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"{name}: Mean {scoring} = {mean_score:.2f}, Standard Deviation = {std_score:.2f}")

best_model_idx = np.argmax([np.mean(scores) for scores in results])
best_model = models[best_model_idx][1]
best_model_name = model_names[best_model_idx]
print(f"\n\nBest Model: {best_model_name}\nWith {scoring}: {np.mean(results[best_model_idx])}")


def balance_dataset(dataset, target_column):
    class_frequencies = dataset[target_column].value_counts().to_dict()
    num_classes = len(class_frequencies)
    min_frequency = min(class_frequencies.values())
    max_frequency = max(class_frequencies.values())

    # Calculate the balance ratio
    balance_ratio = min_frequency / max_frequency

    # Determine if the dataset is balanced
    is_balanced = balance_ratio >= 0.9

    balance_report = {
        "class_frequencies": class_frequencies,
        "num_classes": num_classes,
        "min_frequency": min_frequency,
        "max_frequency": max_frequency,
        "balance_ratio": balance_ratio,
        "is_balanced": is_balanced}
    return balance_report