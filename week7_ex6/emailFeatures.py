import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
from sklearn.svm import SVC
from gaussianKernel import gaussianKernel
from dataset3Params import dataset3Params
from processEmail import processEmail


def emailFeatures(word_indices):
    n = 1899
    X = np.zeros((n,1))
    for i in word_indices:
        X[i] = 1
    return X