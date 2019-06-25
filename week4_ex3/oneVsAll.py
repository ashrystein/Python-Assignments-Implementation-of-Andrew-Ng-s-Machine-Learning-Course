import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
from costFunctionReg import costFunctionReg
from gradientReg import gradientReg

def oneVsAll(X,y):
    (m,n) = X.shape
    lmbda = 0.1
    k = 10
    theta = np.zeros((k,n))

    for i in range(k):
        digit_class = i if i else 10
        theta[i] = opt.fmin_cg(f = costFunctionReg, x0 = theta[i],  fprime = gradientReg,
        args = (X, (y == digit_class).flatten(), lmbda), maxiter = 50)

    return theta