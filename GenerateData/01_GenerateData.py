# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 13:49:02 2016

@author: ventum
"""

import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import make_regression

#X, Y = make_blobs(n_samples=1200,n_features=2,centers=2, cluster_std=3.0)


#print(X)
#print(Y)



X, Y  = make_regression(n_samples=1000, n_features=2, n_informative=10, n_targets=1, bias=0.0, effective_rank=None, tail_strength=0.5, noise=10.0, shuffle=True, coef=False, random_state=None)
print(X)
print(Y)