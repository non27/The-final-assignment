#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 11:10:37 2018

@author: k
"""
#import data
import math
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 
from sklearn import svm
from sklearn.metrics import confusion_matrix
from imblearn.under_sampling import ClusterCentroids
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split  
from sklearn import metrics  
data = pd.read_csv("/Users/k/Desktop/work/non27.csv")
data.head(10)

#normolization
data['nincome'] = StandardScaler().fit_transform(data['income'].values.reshape(-1,1)) 
data = data.drop(['income'], axis=1)
data['napply_amount'] = StandardScaler().fit_transform(data['apply_amount'].values.reshape(-1,1)) 
data = data.drop(['apply_amount'], axis=1)
data['noverdue_amount'] = StandardScaler().fit_transform(data['overdue_amount'].values.reshape(-1,1)) 
data = data.drop(['overdue_amount'], axis=1)
data['ncredit_amount'] = StandardScaler().fit_transform(data['credit_amount'].values.reshape(-1,1)) 
data = data.drop(['credit_amount'], axis=1)
data['noverdue_months'] = StandardScaler().fit_transform(data['overdue_months'].values.reshape(-1,1)) 
data = data.drop(['overdue_months'], axis=1)
data['noverdue_number'] = StandardScaler().fit_transform(data['overdue_number'].values.reshape(-1,1)) 
data = data.drop(['overdue_number'], axis=1)
data['nextened_number'] = StandardScaler().fit_transform(data['extened_number'].values.reshape(-1,1)) 
data = data.drop(['extened_number'], axis=1)
data['nadvanced_number'] = StandardScaler().fit_transform(data['advanced_number'].values.reshape(-1,1)) 
data = data.drop(['advanced_number'], axis=1)
data = data.drop(['id'], axis=1)

#One-Hot Encoding
edu_dum=pd.get_dummies(data['education'], prefix='edu',columns=['education'],drop_first=True) 
data = data.join(edu_dum)
data = data.drop('education', axis=1)
m_dum=pd.get_dummies(data['marriage'], prefix='m',drop_first=True) 
data = data.join(m_dum)
data = data.drop('marriage', axis=1)

#imbalanced data
count_state = pd.value_counts(data['state'] == 1,sort = True).sort_index()
count_state.plot(kind = 'bar')
plt.xlabel('True = 1;False = 0')
plt.ylabel('Number')
plt.show()

#confusion_matrix
def plot_confusion_matrix(cm, classes, title, cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#define X y
X, y = data.loc[:,data.columns != 'state'].values, data.loc[:,data.columns == 'state'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) 
#define the size of test
#sklearn.model_selection.train_test_split随机划分训练集与测试集
#train_test_split(train_data,train_target,test_size=数字, random_state=0)

#ClusterCentroids
cc = ClusterCentroids(random_state=0)
os_X,os_y = cc.fit_sample(X_train,y_train)

#svm,gamma,C
#clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
#params={'learning_rate':np.linspace(0.05,0.25,5), 'max_depth':[x for x in range(1,8,1)], 'min_samples_
#clf = GradientBoostingClassifier()
#grid = GridSearchCV(clf, params, cv=10, scoring="f1")
#grid.fit(X, y)
grid = GridSearchCV(svm.SVC(kernel='rbf',decision_function_shape='ovr'), param_grid={"C":[0.1, 1, 10], "gamma": [1, 0.1, 0.01]}, cv=3)
grid.fit(os_X,os_y.ravel())
print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))
grid.best_estimator_
best_svm=grid.best_estimator_
y_true, y_pred = y_test, best_svm.predict(X_test)

#F1_score, precision, recall, specifity, G score
print "F1_score : %.4g" % metrics.f1_score(y_true, y_pred)  
print "Recall : %.4g" % metrics.recall_score(y_true, y_pred)
recall = metrics.recall_score(y_true, y_pred)  
print "Precision : %.4g" % metrics.precision_score(y_true, y_pred)
 
#Compute confusion matrix
cnf_matrix = confusion_matrix(y_test,y_pred)
np.set_printoptions(precision=2)
print "Specifity: " , float(cnf_matrix[0,0])/(cnf_matrix[0,0]+cnf_matrix[0,1])
specifity = float(cnf_matrix[0,0])/(cnf_matrix[0,0]+cnf_matrix[0,1]) 
print "G score: " , math.sqrt(recall/ specifity) 

#Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix_svm+ClusterCentroids')
plt.show()