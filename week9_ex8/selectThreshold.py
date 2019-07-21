
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

def selectThreshold(pcv,ycv):    
    bestEpsilon = 0
    bestF1 = 0
    stepsize = (max(pcv) - min(pcv)) / 1000
    epi_range = np.arange(pcv.min(),pcv.max(),stepsize)

    for epi in epi_range:
        pred = (pcv < epi)[:,np.newaxis]
        tp = np.sum(pred[ycv == 1] == 1)
        fp = np.sum(pred[ycv == 0] == 1)
        fn = np.sum(pred[ycv == 1] == 0)
        prec = tp/(tp+fp)
        rec = tp/(tp+fn)
        F1 = (2*prec*rec)/(prec+rec)
        if F1 > bestF1:
            bestF1 =F1
            bestEpsilon = epi
        
    return bestEpsilon,bestF1