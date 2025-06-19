# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 16:27:52 2025

@author: advit
"""

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


X,y=fetch_openml('mnist_784', version=1,return_X_y=True,as_frame=False)
X=X/255.0

encoder=OneHotEncoder(sparse_output=False)
y_onehot=encoder.fit_transform(y.reshape(-1,1))
X_train,X_test,y_train,y_test=train_test_split(X, y_onehot,test_size=0.1,random_state=42)

pd.DataFrame(X_train).to_csv('X_train.csv',index=False,header=False)
pd.DataFrame(X_test).to_csv('X_test.csv',index=False,header=False)
pd.DataFrame(y_train).to_csv('y_train.csv',index=False,header=False)
pd.DataFrame(y_test).to_csv('y_test.csv',index=False,header=False)

print('saved files successfully')
