import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
from sklearn.svm import SVC
from gaussianKernel import gaussianKernel
from dataset3Params import dataset3Params


#1
#pd.read_csv('ex1data1.txt', header = None)
data = loadmat('ex6data1.mat')
X = data['X']
y = data['y']


#1.1
mask = y[:,0] == 1
plt.plot(X[mask,0], X[mask,1] ,'k+')
plt.plot( X[~mask,0],  X[~mask,1] ,'yo')
plt.show()

C = 1 # C = 1 or 100 ....
Kernal = "linear"
classifier = SVC(C , kernel=Kernal)
classifier.fit(X,np.ravel(y))
print(classifier)

# plotting the decision boundary
X_1,X_2 = np.meshgrid(np.linspace(X[:,0].min(),X[:,1].max(),num=100),np.linspace(X[:,1].min(),X[:,1].max(),num=100))
plt.contour(X_1,X_2,classifier.predict(np.array([X_1.ravel(),X_2.ravel()]).T).reshape(X_1.shape),1,colors="b")
plt.xlim(0,4.5)
plt.ylim(1.5,5)
plt.show()

#1.2.1
x1 = np.array([1,2,1]); x2 = np.array([0,4,-1]); sigma = 2
sim = gaussianKernel(x1, x2, sigma)
print(sim)


#1.2.2
data2 = loadmat("ex6data2.mat")
X2 = data2["X"]
y2 = data2["y"]

mask = y2[:,0] == 1
plt.plot(X2[mask,0], X2[mask,1] ,'k+')
plt.plot( X2[~mask,0],  X2[~mask,1] ,'yo')

#almost gamma = 1/sigma
classifier2 = SVC(C = 1,kernel="rbf",gamma=30)
classifier2.fit(X2,y2.ravel())
#print(classifier2)

X_3,X_4 = np.meshgrid(np.linspace(X2[:,0].min(),X2[:,1].max(),num=100),np.linspace(X2[:,1].min(),X2[:,1].max(),num=100))
plt.contour(X_3,X_4,classifier2.predict(np.array([X_3.ravel(),X_4.ravel()]).T).reshape(X_3.shape),1,colors="b")
plt.show()


#1.2.3
data3 = loadmat('ex6data3.mat')
X3 = data3["X"]
y3 = data3["y"]
X3val = data3["Xval"]
y3val = data3["yval"]
#print(X3,y3,X3val,y3val)

mask = y3[:,0] == 1
plt.plot(X3[mask,0], X3[mask,1] ,'k+')
plt.plot( X3[~mask,0],  X3[~mask,1] ,'yo')
#plt.show()

(C3 , Gamma3) = dataset3Params(X3,y3,X3val,y3val)

print(C3,Gamma3)

classifier3 = SVC(C = C3,kernel="rbf",gamma=Gamma3)
classifier3.fit(X3,y3.ravel())
#print(classifier3)



X_4,X_5 = np.meshgrid(np.linspace(X3[:,0].min(),X3[:,1].max(),num=100),np.linspace(X3[:,1].min(),X3[:,1].max(),num=100))
plt.contour(X_4,X_5,classifier3.predict(np.array([X_4.ravel(),X_5.ravel()]).T).reshape(X_4.shape),1,colors="b")
plt.xlim(-0.6,0.3)
plt.ylim(-0.7,0.5)
plt.show()


#2



