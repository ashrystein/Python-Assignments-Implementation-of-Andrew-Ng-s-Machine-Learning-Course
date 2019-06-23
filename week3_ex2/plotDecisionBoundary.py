import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from costFunction import costFunction
from gradient import gradient
from fmintnc import fmintnc
from sigmoid import sigmoid
from predict import predict
from mapFeature import mapFeature
from costFunctionReg import costFunctionReg
from gradientReg import gradientReg

def plotDecisionBoundary(theta, u, v):
    z = np.zeros((len(u), len(v)))

    for i in range(len(u)):
        for j in range(len(v)):
            z[i,j] = np.dot(mapFeature(u[i], v[j]), theta)
    return z

