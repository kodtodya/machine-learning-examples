#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 17:59:07 2019

@author: kodtodya
"""
# Data processing

# Importing libraries 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing datasets

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.preprocessing import Imputer

# Imputer clss transforms the missing vales based on strategy and axis(rows/columns)
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

# we are putting x into imputer because x contains the training data
imputer = imputer.fit(x[:, 1:3])
# Transforming the missing data with column mean and updating it
x[:, 1:3] = imputer.transform(x[:, 1:3])

# Encoding categorial data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])

onehotencode = OneHotEncoder(categorical_features=[0])
x = onehotencode.fit_transform(x).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# splitting dataset into Training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# feature scaling
from sklearn.preprocessing import StandardScaler
  
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_ttest = sc_x.transform(x_test)
 