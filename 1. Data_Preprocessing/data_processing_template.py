#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 15:10:39 2019

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

# splitting dataset into Training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
