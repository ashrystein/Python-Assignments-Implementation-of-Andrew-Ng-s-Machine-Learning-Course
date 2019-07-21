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
from selectThreshold import selectThreshold
from cofiCostFunc import cofiCostFunc
from os.path import join
from normalizeRatings import normalizeRatings
from optimizeParams import optimizeParams



#2.1
data = loadmat('ex8_movies.mat')
Y = data['Y']
R = data['R']

#plt.imshow(Y)
#plt.xlabel("Users")
#plt.ylabel("Movies")
#plt.show()

#2.2

data2 = loadmat("ex8_movieParams.mat")
X = data2['X']
Theta = data2['Theta']
num_users = data2['num_users']
num_movies = data2['num_movies']
num_features = data2['num_features']

num_users, num_movies, num_features = 4,5,3
X_test = X[:num_movies,:num_features]
Theta_test = Theta[:num_users,:num_features]
Y_test = Y[:num_movies,:num_users]
R_test = R[:num_movies,:num_users]
params = np.append(X_test.flatten(),Theta_test.flatten())


#2.2.1,2.2.2
lam = 0
J, grad, reg_J, reg_grad = cofiCostFunc(params, Y_test, R_test, num_users, num_movies, num_features,10)
#print(J, grad, reg_J, reg_grad)


#2.3
with open('movie_ids.txt',  encoding='ISO-8859-1') as fid:
    movies = fid.readlines()

movieList = []
for movie in movies:
    movieList.append(movie.strip())


my_ratings = np.zeros((1682,1))
my_ratings[0] = 4 
my_ratings[97] = 2
my_ratings[6] = 3
my_ratings[11]= 5
my_ratings[53] = 4
my_ratings[63]= 5
my_ratings[65]= 3
my_ratings[68] = 5
my_ratings[82]= 4
my_ratings[225] = 5
my_ratings[354]= 5

print("New user ratings:\n")
for i in range(len(my_ratings)):
    if my_ratings[i]>0:
        print("Rated",int(my_ratings[i]),"for ",movieList[i])


Y = np.append(my_ratings,Y,axis=1)
R = np.append((my_ratings != 0),R,axis=1)

Ynorm, Ymean = normalizeRatings(Y,R)
num_movies , num_users = Y.shape
num_features = 10

X = np.random.randn(num_movies, num_features)
Theta = np.random.randn(num_users, num_features)

initial_parameters = np.append(X.flatten(),Theta.flatten())
Lambda = 10

paramsFinal , J_history = optimizeParams(initial_parameters,Y,R,num_users,num_movies,num_features,0.001,400,Lambda)
X_Final = paramsFinal[:num_movies*num_features].reshape(num_movies,num_features)
Theta_Final = paramsFinal[num_movies*num_features:].reshape(num_users,num_features)


plt.plot(J_history[:,1] , J_history[:,0])
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")
plt.show()

pred = X_Final @ Theta_Final.T
my_predictions = pred[:,0][:,np.newaxis] + Ymean

df = pd.DataFrame(np.hstack((my_predictions,np.array(movieList)[:,np.newaxis])))
df.sort_values(by=[0],ascending=False,inplace=True)
print('Top 10 recommended movies:','\n',df[:10])