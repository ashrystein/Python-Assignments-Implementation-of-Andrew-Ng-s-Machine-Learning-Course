import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as imag
from scipy.io import loadmat
import scipy.optimize as opt
from sklearn.svm import SVC
from estimateGaussian import estimateGaussian
from multivariateGaussian import multivariateGaussian
from visualizeFit import visualizeFit
from selectThreshold import selectThreshold
from cofiCostFunc import cofiCostFunc
from os.path import join


def normalizeRatings(Y, R):
    m,n = Y.shape[0], Y.shape[1]
    Ymean = np.zeros((m,1))
    Ynorm = np.zeros((m,n))
    for i in range(m):
        Ymean[i] = np.sum(Y[i,:])/np.count_nonzero(R[i,:])
        Ynorm[i,R[i,:]==1] = Y[i,R[i,:]==1] - Ymean[i]
    return Ynorm, Ymean