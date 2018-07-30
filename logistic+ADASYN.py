#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 23:09:39 2018

@author: k
"""
#import data
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler 
from imblearn.over_sampling import ADASYN
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import KFold

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

#Kfold(best_c)
def printing_Kfold_scores(x_train_data,y_train_data):
    fold = KFold(len(y_train_data),5,shuffle=False) 
    c_param_range = [0.01,0.1,1,10,100]
    results_table = pd.DataFrame(index = range(len(c_param_range),2), columns = ['C_parameter','Mean F1_score'])
    results_table['C_parameter'] = c_param_range
    j = 0
    for c_param in c_param_range:
        print('-------------------------------------------')
        print('C parameter: ', c_param)
        print('-------------------------------------------')
        print('')
        f1_scs = []
        for iteration, indices in enumerate(fold,start=1):
            lr = LogisticRegression(C = c_param, penalty = 'l1')
            lr.fit(x_train_data.iloc[indices[0],:],y_train_data.iloc[indices[0],:].values.ravel())
            y_pred_undersample = lr.predict(x_train_data.iloc[indices[1],:].values)
            f1_sc = metrics.f1_score(y_train_data.iloc[indices[1],:].values,y_pred_undersample)
            f1_scs.append(f1_sc)
            print('Iteration ', iteration,': F1_score = ', f1_sc)
        results_table.loc[j,'Mean F1_score'] = np.mean(f1_scs)
        j += 1
        print('')
        print('Mean F1_score ', np.mean(f1_scs))
        print('')
    best_c = results_table
    best_c.dtypes.eq(object) # you can see the type of best_c
    new = best_c.columns[best_c.dtypes.eq(object)] #get the object column of the best_c
    best_c[new] = best_c[new].apply(pd.to_numeric, errors = 'coerce', axis=0) # change the type of object
    best_c
    best_c = results_table.loc[results_table['Mean F1_score'].idxmax()]['C_parameter'] #calculate the mean values
    print('*********************************************************************************')
    print('Best model to choose from cross validation is with C parameter = ', best_c)
    print('*********************************************************************************')
    return best_c

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

#ADASYN
ada = ADASYN()
os_X,os_y = ada.fit_sample(X_train,y_train)
os_X = pd.DataFrame(os_X)
os_y = pd.DataFrame(os_y)

#logistic
best_c = printing_Kfold_scores(os_X,os_y)
clf_l = LogisticRegression(C = best_c, penalty = 'l1')
clf_l.fit(os_X,os_y.values.ravel())
y_pred = clf_l.predict(X_test)
#调用ravel()函数将矩阵转变成一维数组
#（ravel()函数与flatten()的区别）
# 两者所要实现的功能是一致的（将多维数组降为一维），
# 两者的区别在于返回拷贝（copy）还是返回视图（view），
# numpy.flatten() 返回一份拷贝，对拷贝所做的修改不会影响（reflects）原始矩阵，
# 而numpy.ravel()返回的是视图（view），会影响（reflects）原始矩阵。
y_true, y_pred = y_test, clf_l.predict(X_test)

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

# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix_Logistic+ADASYN')
plt.show()