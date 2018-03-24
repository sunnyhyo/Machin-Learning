# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 16:51:17 2018

@author: HS
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

#Naive Bayes Classifiers

X = np.array([[0, 1, 0, 1],
              [1, 0, 1, 1],
              [0, 0, 0, 1],
              [1, 0, 1, 0]])
y = np.array([0, 1, 0, 1])

counts = {}
for label in np.unique(y):
# iterate over each class
# count (sum) entries of 1 per feature
    counts[label] = X[y == label].sum(axis=0)
print("Feature counts:\n{}".format(counts))

#Decision Trees
import graphviz
mglearn.plots.plot_animal_tree()


