# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 14:29:03 2021

@author: chint
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import time
import numpy as np
from sklearn.metrics import mean_squared_error as mse

df=pd.read_csv('D:/UIUC_courses/IE517/IE517_FY21_HW5/hw5_treasury yield curve data.csv')
del df['Date']
ss=StandardScaler()
ss.fit(df)
corr=df.corr()
sns.heatmap(corr) 
df2=ss.fit_transform(df)
plt.show()
X=df2[:,:-1]
y=df2[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=43)
pca=PCA()
pca.fit(X_train)
plt.bar(range(pca.n_components_),pca.explained_variance_ratio_)
plt.plot(range(pca.n_components_),np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('component')
plt.ylabel('explained variance ratio')
plt.title('Variance ratio vs pca component')
plt.show()

plt.bar([1,2,3],pca.explained_variance_ratio_[0:3])
plt.plot([1,2,3],np.cumsum(pca.explained_variance_ratio_)[0:3])
plt.xlabel('component')
plt.ylabel('explained variance ratio')
plt.title('Variance ratio vs pca component (first three)')
plt.show()
print("cumulative sum for 3 pca components:")
print(np.cumsum(pca.explained_variance_ratio_)[2])

X_pca_full=pca.fit_transform(X_train)
X_small=X_pca_full[:,:3]
plt.plot(X[:,0])
plt.xlabel('days')
plt.ylabel('yield rate')
plt.title('Yield rate plot for treasury bond 1')
plt.show()
lr=LinearRegression()
svr=SVR()
t = time.time()
lr.fit(X_train,y_train)
elapsed = time.time() - t
print('time taken to fit linear regression without pca is:')
print(elapsed)
t = time.time()
svr.fit(X_train,y_train)
elapsed = time.time() - t
print('time taken to fit SVM without pca is:')
print(elapsed)
print("train R2 of linear regression is:")
print(lr.score(X_train,y_train))
print("test R2 of SVM is:")
print(svr.score(X_train,y_train))
print("test R2 of linear regression is:")
print(lr.score(X_test,y_test))
print("test R2 of SVM is:")
print(svr.score(X_test,y_test))

print("train RMSE of linear regression is:")
print(mse(lr.predict(X_train),y_train)**0.5)
print("test RMSE of SVM is:")
print(mse(svr.predict(X_train),y_train)**0.5)
print("test RMSE of linear regression is:")
print(mse(lr.predict(X_test),y_test)**0.5)
print("test RMSE of SVM is:")
print(mse(svr.predict(X_test),y_test)**0.5)

lr=LinearRegression()
svr=SVR()
t = time.time()
lr.fit(X_small,y_train)
elapsed = time.time() - t
print('time taken to fit linear regression with pca is:')
print(elapsed)
t = time.time()
svr.fit(X_small,y_train)
elapsed = time.time() - t
print('time taken to fit SVM with pca is:')
print(elapsed)
print("train accuracy of linear regression with PCA is:")
print(lr.score(X_small,y_train))
print("test accuracy of SVM with PCA is:")
print(svr.score(X_small,y_train))
print("test accuracy of linear regression with PCA is:")
print(lr.score(pca.fit_transform(X_test)[:,:3],y_test))
print("test accuracy of SVM with PCA is:")
print(svr.score(pca.fit_transform(X_test)[:,:3],y_test))

print("train RMSE of linear regression is:")
print(mse(lr.predict(X_small),y_train)**0.5)
print("test RMSE of SVM is:")
print(mse(svr.predict(X_small),y_train)**0.5)
print("test RMSE of linear regression is:")
print(mse(lr.predict(pca.fit_transform(X_test)[:,:3]),y_test)**0.5)
print("test RMSE of SVM is:")
print(mse(svr.predict(pca.fit_transform(X_test)[:,:3]),y_test)**0.5)

print("My name is Prajwal Chinthoju")
print("My NetID is: pkc3")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
