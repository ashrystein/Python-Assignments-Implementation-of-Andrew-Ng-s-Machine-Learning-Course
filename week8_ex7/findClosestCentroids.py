import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
from sklearn.svm import SVC



def findClosestCentroids(X, centroids):
    m = X.shape[0]
    k = centroids.shape[0]
    idx = np.zeros(m,dtype=int)
    dist = 0
    indx = 0
    for i in range(m):
        xi = X[i,:]
        for j in range(k):
            ck  = centroids[j,:]
            d = np.sqrt(np.sum(np.power(xi-ck,2)))
            if j == 0 :
                dist = d
                indx = j
            elif d < dist :
                dist = d
                indx = j
        idx[i] = indx+1
    return idx