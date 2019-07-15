import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as imag
from scipy.io import loadmat
import scipy.optimize as opt
from sklearn.svm import SVC
from findClosestCentroids import findClosestCentroids
from computeCentroids import computeCentroids
from runkMeans import runkMeans
from kMeansInitCentroids import kMeansInitCentroids
from featureNormalize import featureNormalize

def pca(X):
    (m,n) = X.shape
    U = np.zeros(n)
    S = np.zeros(n)
    sigma = (1/m)*(X.T@X)
    U, S, V = np.linalg.svd(sigma)
    return U,S,V