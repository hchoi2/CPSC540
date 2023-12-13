""""""
# Context
# As social drinking has increased, the red wine business has recently experienced exponential expansion. 
# Industry participants now use product quality certifications to market their goods. 
# This is a time-consuming process that also costs a lot of money because it needs to be evaluated by human experts. 
# Additionally, the cost of red wine is determined by tasters' opinions, which can vary greatly and are based on an abstract concept of wine appreciation. 
# Physicochemical tests, which are laboratory-based and take into account elements like acidity, pH level, sugar, and other chemical qualities, are an additional crucial component in the certification of red wine and the evaluation of its quality.
# If the human quality of tasting can be linked to the chemical characteristics of wine, the market for red wine could be of interest. 
# The goal of this study is to develop a model that may be used to anticipate the quality of a new red wine.

# Dataset
# This datasets is related to red variants of the Portuguese "Vinho Verde" wine. 
# The dataset describes the amount of various chemicals present in wine and their effect on it's quality.

# Input variables (based on physicochemical tests):
# fixed acidity - Non-volatile acids in wine are what cause fixed acidity. For instance, malic, citric, or tartaric acid. This kind of acid mixes the wine's taste's harmony and adds freshness.
# volatile acidity - The portion of the acid in wine that may be detected by the nose is called volatile acidity. As opposed to those acids that can be tasted. One of the most typical flaws is volatile acidity, or the souring of wine.
# citric acid - It can be used to collect wine, treat wine with acid to increase acidity, and clean filters of potential fungal and mould infections.
# residual sugar - Grape sugar that has not undergone alcohol fermentation is referred to as residual sugar.
# chlorides - The amount of minerals in the wine affects its structure as well. Their content is mostly influenced by climatic region, oenological procedures, wine storage, and ageing.
# free sulfur dioxide - The antioxidant and antibacterial qualities of sulphur dioxide make it a useful preservative. It is a very powerful antibiotic that has a major impact on consumption and can result in wine deterioration.
# total sulfur dioxide - It consists of the SO2 that is present in the wine both free and bound to other substances like aldehydes, pigments, or sugars.
# density - Wine can have a density that is less or greater than that of water. Its value is mostly influenced by the alcohol and sugar content.
# pH - Wine's pH is a gauge of its acidity. The optimal pH range for all wines is between 2.9 and 4.2. More acidic wines have lower pH values; less acidic wines have higher pH values.
# sulphates - Sulfates are a byproduct of wine's sugar being fermented by yeast into alcohol.
# alcohol - Wines' alcohol percentage is influenced by a wide range of factors, including grape varietal, berry sugar content, manufacturing methods, and growth environments. **
# quality - This is the variable we are after. This is a score between 0 and 10 where a higher score indicates that it was deemed to be a superior wine.
""""""
"""Data"""
# 1.1 | Exploring the Data
# Lets import the data and have a brief look.
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
data = pd.read_csv('./Data/WineQT.csv')
print(data)
plt.savefig("./Images/OriginalData.png")
plt.close()
# Split the data into training and testing sets
X = data.iloc[:, 0:11]
y = data.iloc[:, 11]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1.2 | Looking at components of the Data
# Let's check the various columns in dataset.
data.info()
# Note: Since, we have 1143 values and each values are Non-Null. That means our Data has no missing values.
from missingno import bar
_ = bar(data, figsize=(10, 5), color='#FF281B')
# This plot shows the same thing.

# 1.3 | Fixing the Dataframe
# Removing a useless column.
data = data.drop(columns=['Id'])
print(data)
plt.savefig("./Images/RemovedId.png")
plt.close()

# 1.4 | Visualizing the Data
# Visualizing our data to get some intuition.
from plotly.express import parallel_coordinates
labels = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", 
          "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"]
colors = ['#f94449', '#de0a26', '#a8071a', '#7a0a14', '#4d070e', '#200207']
fig = parallel_coordinates(data, color="quality", labels=labels, color_continuous_scale=colors)
fig.show()
plt.savefig("./Images/ParallelCoordinates.png")
plt.close()

"""Exploratory Data Analysis"""
# 2.1 | Exploring the variables
# To get a overview of the values and there distribution.
with pd.option_context('display.precision', 2):
    explore = data.describe().T.style.background_gradient(cmap='Reds')
print(explore)

# 2.2 | Correlation
# To understand the correlation between various descriptors.
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt

corr = data.corr()
X = data.columns
colorscale = [[0, '#f94449'], [1, '#de0a26']]
heat = go.Heatmap(z=corr, x=X, y=X, xgap=1, ygap=1, colorscale=colorscale, colorbar_thickness=20, colorbar_ticklen=3,)
layout = go.Layout(title_text='Correlation Matrix', title_x=0.5,  width=600, height=600,  xaxis_showgrid=False, yaxis_showgrid=False,
                   yaxis_autorange='reversed')
fig = go.Figure(data=[heat], layout=layout)        
fig.show()
plt.savefig("./Images/Correlation.png") 
plt.close()

# 2.3 | Boxplot
# Looking at the distribution of the data.
from plotly.subplots import make_subplots

fig = make_subplots(rows=2, cols=len(data.columns)//2)
for i, var in enumerate(data.columns):
    fig.add_trace(go.Box(y=data[var], name=var),row=i%2+1, col=i//2+1)

fig.update_traces(boxpoints='all', jitter=.3)
fig.update_layout(height=1000, showlegend=False)
fig.show()
plt.savefig("./Images/Boxplot.png")
plt.close()

# 2.4 | Scatterplot
# Scatterplot to show the interactions between the variables.
from plotly.express import scatter_matrix

fig = scatter_matrix(data_frame = data, color = 'quality', height = 1200, labels=labels)
fig.show()
plt.savefig("./Images/Scatterplot.png")
plt.close()

# 2.5 | Histogram
# Histogram to show the spread of data on the specified variables.
from plotly.figure_factory import create_distplot

fig = make_subplots(rows=4, cols=3, subplot_titles=data.columns)

for j,i in enumerate(data.columns):
    fig2 = create_distplot([data[i].values], [i])
    fig2.data[0].autobinx = True
    fig.add_trace(go.Histogram(fig2['data'][0], marker_color='#f94449'), row=j//3 + 1, col=j%3 + 1)
    fig.add_trace(go.Scatter(fig2['data'][1], marker_color='#de0a26'), row=j//3 + 1, col=j%3 + 1)

fig.update_layout(height=1200, showlegend=False, margin={"l": 0, "r": 0, "t": 20, "b": 0})
fig.show()
plt.savefig("./Images/Histogram.png")
plt.close()

# 2.6 | Distribution of the Target Variable
# Shows the distribution of the target variable. This also demonstrates the class inbalance in data.
colors = ['#f94449', '#de0a26', '#a8071a', '#7a0a14', '#4d070e', '#200207']
fig = make_subplots(rows = 1, cols = 2, specs = [[{"type": "pie"}, {"type": "bar"}]])
fig.add_trace(go.Pie(values = data.quality.value_counts(), labels = data.quality.value_counts().index, domain = dict(x=[0, 0.5]), 
                     marker = dict(colors = colors), hole = .3, name=''), row = 1, col = 1)
fig.add_trace(go.Bar(x = data.quality.value_counts().index, y = data.quality.value_counts(), name='', marker = dict(color = "quality",
                     colorscale = colors)), row = 1, col = 2)
fig.update_layout(showlegend = False)
fig.show()
plt.savefig("./Images/TargetVariable.png")
plt.close()

# 2.7 | Comparison Plot
# Plot to show weightage of different quality values with respect to the variable.
from plotly.express import box

fig = make_subplots(rows=4, cols=3, subplot_titles=[c for c in data.columns[:-1]])
for i,v in enumerate(data.columns[:-1]):
    for t in box(data, y=v, x="quality", color="quality").data:
        fig.add_trace(t, row=(i//3)+1, col=(i%3)+1)
fig.update_layout(height=1400, showlegend=False, margin={"l": 0, "r": 0, "t": 20, "b": 0})
fig.show()
plt.savefig("./Images/ComparisonPlot.png")
plt.close()

# 2.8 | Skewness
# Plot to show the skewness in data. If the skewness is high it means that the data is not normally distrbuted.
from plotly.express import bar

skewness = data.skew().sort_values(ascending=True)
fig = bar(x=skewness, y=skewness.index, color=skewness.index, labels={'x': 'Skewness', 'y':'Descriptors'})
fig.update_layout(showlegend=False)
fig.add_vline(x=1, line_dash="dash", line_color="red")
fig.show()
plt.savefig("./Images/Skewness.png")
plt.close()

# 3.1 | Fixing Unbalanced Classes 
# Since, we noticed that the classes are unbalanced from the EDA we have to balanced them before applying machine learning model.

# The Synthetic Minority Oversampling Technique (SMOTE) is a statistical technique for increasing the number of cases in a balanced manner in your dataset. 
# The component creates new instances from minority situations that you specify as input that already exist. 
# The quantity of majority cases remains unchanged as a result of this SMOTE implementation.

from sklearn import preprocessing
from imblearn.over_sampling import SMOTE 

oversample = SMOTE()
features, labels =  oversample.fit_resample(data.drop(["quality"],axis=1), data["quality"])
scaler = preprocessing.MinMaxScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
print(scaled_data)
plt.savefig("./Images/FixedUnbalancedClasses.png")
plt.close()

# 3.2 | Normalizing Data
# After experimenting with pycaret we found out that normaliztion is also useful in our case.
normalized_arr = preprocessing.normalize(scaled_data)
normalized_data = pd.DataFrame(normalized_arr, columns=features.columns)
print(normalized_data)

# 3.3 | Unskewing Data
# After experimenting which values should be unskewed the following variables where selected.
from numpy import log

unskew_data = normalized_data.copy(deep=True)
unskew_data['residual sugar'] = unskew_data['residual sugar'].replace(0.0, 0.01).apply(log)
unskew_data['chlorides'] = unskew_data['chlorides'].replace(0.0, 0.01).apply(log)
unskew_data['total sulfur dioxide'] = unskew_data['total sulfur dioxide'].replace(0.0, 0.01).apply(log)
unskew_data['free sulfur dioxide'] = unskew_data['free sulfur dioxide'].replace(0.0, 0.01).apply(log)

# 3.4 | Trying Models
# We will try different machine learning classifiers to figure out which ones we should pick.
from pycaret.classification import setup, compare_models

S = setup(data=unskew_data, target=labels, verbose=True)

# Data Facts
# Normalize (normalize) - Normalizing the data increases the accuracy of the model.
# Multicollinearity (remove_multicollinearity) - Removing multicollinearity doesn't improve the accuracy.
# Tranformation (transformation) - On its own it improves accuracy but with normalization it has negative impact.
# Removing Outliers (remove_outliers) - Removing outliers doesn't improve accuracy
# Feature selection (feature_selection) - Feature selection is not useful for most of the models.
best = compare_models(include = ['lr', 'knn', 'nb', 'dt', 'svm', 'rbfsvm', 'mlp', 'ridge', 'rf', 'qda', 'ada', 'gbc', 'lda', 
                                 'et', 'xgboost', 'lightgbm', 'catboost', 'dummy'], fold = 10, n_select=20)

# Finalizing Models
# Only taking the models with accuracy greater than 80%. 
# And doing hyper-parameter tuning manually to improve accuracy. 
# Then a confusion matrix to show the errors.
from IPython.display import Markdown, display
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib.pyplot import subplots, text
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report
# %matplotlib inline

models = [ExtraTreesClassifier(n_estimators=900, random_state=1251), 
         CatBoostClassifier(silent=True, depth=7, random_state=86), 
         LGBMClassifier(random_state=999), 
         RandomForestClassifier(n_estimators=1000, bootstrap=False, class_weight="balanced", random_state=247), 
         XGBClassifier(max_depth=5, subsample=0.7, colsample_bytree=0.8, random_state=149)]
model_name = ['Extra Trees', 'Category Boost', 'Light Gradient Boost', 'Random Forest', 'Extreme Gradient Boost']


y_train2, y_test2 = y_train.copy(), y_test.copy()

for i,j in zip(models, model_name):
    if j[0] == 'E':
        y_train = y_train2
        y_test = y_test2
    i.fit(X_train, y_train)
    y_pred = i.predict(X_test)
    cm = confusion_matrix(y_pred,y_test,labels=i.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = i.classes_)
    fig, ax = subplots(figsize=(6,6))
    ax.grid(False)
    disp.plot(cmap='Reds', ax=ax)
    text(8, 5,  j + '\n' + classification_report(y_test,y_pred, zero_division=1) + '\n' + "Accuracy percent: " + 
             str(accuracy_score(y_pred, y_test)*100)[:5], fontsize=12, fontfamily='Georgia', color='k',ha='left', va='bottom')

