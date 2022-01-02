

import pandas as pd
import matplotlib.pyplot as plt
plt.clf()

hsg=pd.read_csv('D:/UIUC_courses/IE517/IE517_FY21_HW4/housing.csv')

print("Raw Dataset: first 5 rows")
print(hsg.head())
print("Raw Dataset:info")
print(hsg.describe())

import seaborn as sns

labels=hsg.columns
sns.heatmap(hsg.corr(),annot=False)
plt.show()
labels_reduced=labels[13:]
hsg_red=hsg[labels_reduced]
#sns.heatmap(hsg_red.corr(),annot=True,annot_kws = {'size':5})
plt.show()

print('Summary of most correleated features')
print(hsg_red.describe())

sns.pairplot(hsg_red)
plt.show()

from sklearn.preprocessing import StandardScaler
hsg_scaled=StandardScaler().fit_transform(hsg)

from sklearn import datasets, linear_model
from sklearn.model_selection import cross_validate,KFold,cross_val_score
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

lasso=linear_model.Lasso()
ridge=linear_model.Ridge()

X=hsg[:,:-1]
y=hsg[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0)
pipe = Pipeline([('scaler', StandardScaler()), ('estimator', lasso)])
cv = KFold(n_splits=4)
scores = cross_val_score(pipe, X, y, cv = cv)
print(scores)
