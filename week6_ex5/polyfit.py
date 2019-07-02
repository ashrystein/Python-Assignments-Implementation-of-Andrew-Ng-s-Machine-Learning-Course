import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
from costFunctionReg import costFunctionReg
from gradientReg import gradientReg
from trainLinearReg import trainLinearReg
from learningCurve import learningCurve
from polyFeatures import polyFeatures
from featureNormalize import featureNormalize

def polyfit(min_x, max_x, mu, sigma, theta, p):
    x = np.arange(min_x - 50, max_x + 50, 0.05).reshape(-1, 1)
    x = np.hstack((np.ones((x.shape[0], 1)), x))
    X_poly = polyFeatures(x, p)
    X_poly -= mu
    X_poly /= sigma
    X_poly = np.hstack((np.ones((x.shape[0], 1)), X_poly))
    plt.plot(x[:,1], np.dot(X_poly, theta), '--', lw=2)
    return