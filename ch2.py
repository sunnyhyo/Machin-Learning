# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 15:26:48 2018

@author: HS
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn


#회귀분석용 boston data
from sklearn.datasets import load_boston
boston = load_boston()
print("Data shape: {}".format(boston.data.shape))

X, y = mglearn.datasets.load_extended_boston()
print("X.shape: {}".format(X.shape))
