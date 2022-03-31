#https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1512123
#sum_of_sinusoid_Method.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

fc=2*10e9 #carrier frequency
c=3*10e8 
v=20 # user velocity
fd=fc*v/c #doppler frequency
Ts=0.001
a=0
b=2*np.pi
M=124
alpha=a+(b-a)*np.random.uniform(0,1,M) #uniformly distributed from 0 to 2 pi
beta=a+(b-a)*np.random.uniform(0,1,M) #uniformly distributed from 0 to 2 pi
theta=a+(b-a)*np.random.uniform(0,1,M) #uniformly distributed from 0 to 2 pi
m=np.arange(M)+1;
x=np.cos(((2*m-1)*np.pi+theta)/(4*M))
N=100000
h_re=np.zeros(N)
h_im=np.zeros(N)
for n in range(N):

  h_re[n]=1/np.sqrt(M)*np.sum(np.cos(2*np.pi*fd*x*n*Ts+alpha))
  h_im[n]=1/np.sqrt(M)*np.sum(np.sin(2*np.pi*fd*x*n*Ts+beta))


#Feature
X=np.vstack((h_re, h_im))
X=np.transpose(X)
#Target
y=X
#including Past k+1 data in feature vectors for k+1 Order Markov Approximation
l=len(y)
y=y[0:l-100]

u=1 #number of pastsequence 
Xt=X[0:l-100,:]
print(Xt.shape)
a=np.zeros(u)
for j in range(u):
    Xt=np.hstack((Xt,X[j+1:l-99+j,:]))
X=Xt

i=20 #number of Age values in the simulation
l=len(y)
age=np.zeros(i) 
score1=np.zeros(i)
score2=np.zeros(i)

for n in range(i):
  print(n)
  age[n]=n
  y_t=y[0:l-n]
  X_t=X[n:l, :]
    
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X_t, y_t, test_size=0.20)
  from sklearn.linear_model import LinearRegression
  regressor = LinearRegression()
  regressor.fit(X_train, y_train)
    
  y_pred = regressor.predict(X_train)
  y_val =  regressor.predict(X_test)
  from sklearn.metrics import mean_squared_error
  score1[n]=mean_squared_error(y_train, y_pred)
  score2[n]=mean_squared_error(y_test, y_val)

print(score1)
print(score2)
np.save('trainingerrorCSIu1.npy', score1)
np.save('inferenceerrorCSIu1.npy', score2)


