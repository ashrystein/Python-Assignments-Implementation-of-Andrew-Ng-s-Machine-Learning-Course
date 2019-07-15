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
from pca import pca



def projectData(X, U, K):
    U_reduce = U[:,:K]
    Z = X @ U_reduce
    return Z