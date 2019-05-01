# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 10:04:15 2018
@author: 段昊
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

#step1 load data
df = pd.read_csv('D:\\wine.csv')
df.head()

#step2 preprocessing
scaler = MinMaxScaler()
scaler.fit(df.drop('class',axis=1))
scaled_features = scaler.transform(df.drop('class',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=df.columns[1:])
X = df_feat
y = df['class']
#X = preprocessing.scale(X)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.50)

#step3 training
bpn = MLPClassifier(solver='lbfgs', alpha=1e-1, random_state=10)
bpn.fit(X_train, y_train)
prediction = bpn.predict(X_test)

#Step4 accuracy
accuracy = metrics.accuracy_score(y_test,prediction)
print('Accuracy is {:.2%}'.format(accuracy))
