import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
from costFunctionReg import costFunctionReg
from gradientReg import gradientReg


def trainLinearReg(X , y , lam):
    (m,n) = X.shape
    initial_theta = np.zeros(n)
    thetaOpt = opt.fmin_cg(f = costFunctionReg, x0 = initial_theta, fprime = gradientReg,
        args = (X,y.flatten(),lam), maxiter = 200)
    return thetaOpt