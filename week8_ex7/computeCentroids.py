import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
from sklearn.svm import SVC


def computeCentroids(X, idx, K):
    m, n = X.shape
    centroids = np.zeros((K, n))
    for i in range(K):
        mask = idx == i+1
        mu = (np.sum(X[mask],axis=0))/(np.sum(mask))
        centroids[i] = mu
    return centroids