import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as imag
from scipy.io import loadmat
import scipy.optimize as opt
from sklearn.svm import SVC


def estimateGaussian(X):
    (m) = X.shape[0]
    mu = (1/m) * np.sum(X,axis=0)
    sig = (1/m) * np.sum((X-mu)**2,axis=0)
    #print(sig)
    return mu , sig