
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn



#유방암종양 data
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("cancer.keys(): \n{}".format(cancer.keys()))
print("Shape of cancer data: {}".format(cancer.data.shape))
print("Sample counts per class:\n{}".format({n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))
print("Feature names:\n{}".format(cancer.feature_names))


# boston data 회귀분석용
from sklearn.datasets import load_boston
boston = load_boston()
print("Data shape: {}".format(boston.data.shape))

X, y = mglearn.datasets.load_extended_boston()
print("X.shape: {}".format(X.shape))


