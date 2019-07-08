import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
from sklearn.svm import SVC
from gaussianKernel import gaussianKernel



def dataset3Params(X, y, Xval, yval):
    min_c = 0
    min_gamma = 0
    candidates = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])
    error = 0

    for i in candidates:
        for j in candidates:
            C = i
            gamma = 1/j
            classifier = SVC(C = C,kernel="rbf",gamma=gamma)
            classifier.fit(X,y.ravel())
            classifier.predict(Xval)
            score = classifier.score(Xval,yval)
            if score > error:
                error = score
                min_c = C
                min_gamma = gamma
    return min_c , min_gamma

