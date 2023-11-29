import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier as ABC
import pandas as pd

# Load data
data = pd.read_csv('/Users/hyejeongchoi/Desktop/hchoi_homepage/hchoi2/Python/Data/WineQT.csv')
X = data.iloc[:, 0:11]
y = data.iloc[:, 11] 
print(y.head())