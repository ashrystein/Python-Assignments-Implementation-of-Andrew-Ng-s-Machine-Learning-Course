import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from costFunction import costFunction
from gradient import gradient
from fmintnc import fmintnc
from sigmoid import sigmoid
from predict import predict

#1
data = pd.read_csv('ex2data1.txt', header=None)
X = data.iloc[:,0:2]
y = data.iloc[:,2]
m = len(y)
data_top = data.head()
print(data_top)


#1.1
mask = y == 1
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
adm = plt.plot(X[mask][0], X[mask][1] ,'k+',label = 'Admitted')
not_adm = plt.plot(X[~mask][0], X[~mask][1] ,'yo',label = 'Not admitted')


#1.2
ones = np.ones((m,1))
y = y[:,np.newaxis]
X = np.hstack((ones, X))
theta = np.zeros([3,1])
J = costFunction(theta,X,y)
print(J)

grad = gradient(theta,X,y)
print(grad)

theta_optimized = fmintnc(theta,X,y)
print(theta_optimized)

J_theta_optimized = costFunction(theta_optimized[:,np.newaxis],X,y)
print(J_theta_optimized)

#TestCase
Xtest = np.array([[1,45,85]])
test = sigmoid(np.dot(Xtest,theta_optimized))
print(test)

pred = predict(theta_optimized[:,np.newaxis] , X ,m)
acc = np.mean(pred == y)
print(acc * 100 , '%')

plot_x = [np.min(X[:,1]-2), np.max(X[:,2]+2)]
plot_y = -1/theta_optimized[2]*(theta_optimized[0] + np.dot(theta_optimized[1],plot_x))
print(plot_x)
print(plot_y)
decision_boun = plt.plot(plot_x, plot_y ,label = 'decision boundary')
plt.legend()
plt.show()
