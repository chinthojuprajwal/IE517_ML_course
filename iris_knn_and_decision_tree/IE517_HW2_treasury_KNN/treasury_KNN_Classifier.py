#Our first machine learning model
#Garreta and Moncecchi pp 10-20
#uses Iris database and SGD classifier

from sklearn import datasets
import pandas as pd
import numpy as np
#print('The scikit learn version is {}.'.format(sklearn.__version__))
path='D:/UIUC_courses/IE517/IE517_FY21_HW2/IE517_HW2_treasury_decision_tree/Treasury_squeeze_raw_score_data.csv'
treasury=pd.read_csv(path)
treasury=treasury.values
treasury=treasury[:,2:12]
X_tres, y_tres = treasury[:,:9], treasury[:,9]
X_tres= X_tres.astype(np.float)
y_tres= y_tres.astype(np.float)
X, y = X_tres[:, :9], y_tres

#[ 5.1  3.5  1.4  0.2] 0

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
# Get dataset with only the first two attributes

# Split the dataset into a training and a testing set
# Test set will be the 25% taken randomly
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.40)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

import matplotlib.pyplot as plt
values = ['False','True']
colors=['green','red']
#it doesnt like "xrange" changed to "range"
for i in range(len(values)):
    xs = X_train[:, 0][y_train == i]
    ys = X_train[:, 1][y_train == i]
    plt.scatter(xs, ys, c=colors[i])
plt.legend(values)
plt.xlabel('Price Crossing')
plt.ylabel('Price Distortion')
plt.show()
#found a typo here... incorrect from book followed by corrected code
#from sklearn.linear_modelsklearn._model import SGDClassifier
from sklearn import neighbors
accuracy_test=[]
accuracy_train=[]
for n in range(1,70):
    clf = neighbors.KNeighborsClassifier(n, weights='uniform')
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred = clf.predict(X_test)
    from sklearn import metrics

    accuracy_test.append(metrics.accuracy_score(y_test, y_pred))
    accuracy_train.append(metrics.accuracy_score(y_train, y_pred_train))

plt.plot(accuracy_test,label='Test accuaracy')
plt.plot(accuracy_train, label='Train accuaracy')


plt.xlabel('No. of nearest neighbours')
plt.ylabel('Accuracy')
plt.title('Test vs Train Accuracy plot')
plt.legend()
plt.show()
print('Maximum accuracy occurs at n=:')
print(accuracy_test.index(max(accuracy_test))+1)
print('Value of maximum accuracy:')
print((max(accuracy_test)))

# print( clf.coef_)
# #[[-28.53692691 15.05517618]
# #[ -8.93789454 -8.13185613]
# #[ 14.02830747 -12.80739966]]
# print( clf.intercept_)
# #[-17.62477802 -2.35658325 -9.7570213 ]

#don't forget to import Numpy as np here
# import numpy as np
# x_min, x_max = X_train[:, 0].min() - .5, X_train[:, 0].max() + .5
# y_min, y_max = X_train[:, 1].min() - .5, X_train[:, 1].max() + .5
# #error in case Xs or xs
# Xs = np.arange(x_min, x_max, 0.5)
# fig, axes = plt.subplots(1, 3)
# fig.set_size_inches(10, 6)
# for i in [0, 1, 2]:
#     axes[i].set_aspect('equal')
#     axes[i].set_title('Class '+ str(i) + ' versus the rest')
#     axes[i].set_xlabel('Sepal length')
#     axes[i].set_ylabel('Sepal width')
#     axes[i].set_xlim(x_min, x_max)
#     axes[i].set_ylim(y_min, y_max)
#     #error here need plt.
#     plt.sca(axes[i])
#     plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.prism)
#     ys = (-clf.intercept_[i] - Xs * clf.coef_[i, 0]) / clf.coef_[i, 1]
#     plt.plot(Xs, ys)
    
# print( clf.predict(scaler.transform([[4.7, 3.1]])) )
# #[0]

# print( clf.decision_function(scaler.transform([[4.7, 3.1]])) )
# #[[ 19.73905808 8.13288449 -28.63499119]]

# from sklearn import metrics
# y_train_pred = clf.predict(X_train)
# print( metrics.accuracy_score(y_train, y_train_pred) )
# #0.821428571429

# y_pred = clf.predict(X_test)
# print( metrics.accuracy_score(y_test, y_pred) )
# #0.684210526316

# print( metrics.classification_report(y_test, y_pred, target_names=iris.target_names) )
# #precision recall f1-score support
# #setosa 1.00 1.00 1.00 8
# #versicolor 0.43 0.27 0.33 11
# #virginica 0.65 0.79 0.71 19
# #avg / total 0.66 0.68 0.66 38

# print( metrics.confusion_matrix(y_test, y_pred) )
# #[[ 8 0 0]
# #[ 0 3 8]
# #[ 0 4 15]]

print("My name is Prajwal Chinthoju")
print("My NetID is: pkc3")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
#####STOP HERE######################









# ##error in scikit learn package, which version??
# from sklearn.cross_validation import cross_val_score, KFold
# from sklearn.pipeline import Pipeline
# # create a composite estimator made by a pipeline of the standarization and the linear model
# clf = Pipeline([(
#         'scaler', StandardScaler()),
#         ('linear_model', SGDClassifier())
# ])
# # create a k-fold cross validation iterator of k=5 folds
# cv = KFold(X.shape[0], 5, shuffle=True, random_state=33)
# # by default the score used is the one returned by score method of the estimator (accuracy)
# scores = cross_val_score(clf, X, y, cv=cv)
# print( scores )
# #[ 0.66666667 0.93333333 0.66666667 0.7 0.6 ]


# from scipy.stats import sem
# def mean_score(scores): return ("Mean score: {0:.3f} (+/- {1:.3f})").format(np.mean(scores), sem(scores))
# print( mean_score(scores) )
# #Mean score: 0.713 (+/-0.057)


