import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier as ABC
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load data
data = pd.read_csv('/Users/hyejeongchoi/Desktop/hchoi_homepage/hchoi2/Python/Data/WineQT.csv')
X = data.iloc[:, 0:11]
y = data.iloc[:, 11] 
print(y.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)