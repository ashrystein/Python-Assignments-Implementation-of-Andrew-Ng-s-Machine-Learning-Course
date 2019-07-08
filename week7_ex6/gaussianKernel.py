import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
from sklearn.svm import SVC


def gaussianKernel(x1, x2, sigma):
    return np.exp(-1*np.sum(np.power((x1-x2),2))/(2*(sigma**2)))