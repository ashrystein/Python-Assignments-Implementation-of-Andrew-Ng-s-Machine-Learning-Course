from computeCentroids import computeCentroids
from runkMeans import runkMeans
from kMeansInitCentroids import kMeansInitCentroids
from featureNormalize import featureNormalize
from pca import pca
from projectData import projectData



def recoverData(Z, U, K):
    U_reduce = U[:,:K]
    X_rec = Z @ U_reduce.T
    return X_rec