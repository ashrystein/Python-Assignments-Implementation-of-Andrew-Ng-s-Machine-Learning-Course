import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
from displayData import displayData
from nnCostFunction import nnCostFunction
from sigmoid import sigmoid


def sigmoidGradient(z):
    return np.multiply(sigmoid(z),(1-sigmoid(z)))