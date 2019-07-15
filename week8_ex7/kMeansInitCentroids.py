import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
from sklearn.svm import SVC
from findClosestCentroids import findClosestCentroids
from computeCentroids import computeCentroids
from runkMeans import runkMeans




def kMeansInitCentroids(X,K):
    randidx = np.random.permutation(X.shape[0])
    centroids = X[randidx[:K], :]
    return centroids