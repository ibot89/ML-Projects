# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 19:56:52 2017

@author: Botev
"""
"""
This file provides helper functions used in Image Recognition Projects with both Multiclass Logistic Regression and Neural Networks.
It provides a way to simply call the functions anywhere in the code without defining them each time.

"""
import os
import numpy as np 
#print os.getcwd()




#Initialises the weights from a uniform distribution. It initialises the bias parameter to 0.
def init_weight_and_bias(M1, M2):
    W = np.random.randn(M1, M2) / np.sqrt(M1)
    b = np.zeros(M2)
    return W.astype(np.float32), b.astype(np.float32)

# A sigmoid function
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

# A softmax function
def softmax(Z):
    expA = np.exp(Z)
    return expA / expA.sum(axis=1, keepdims=True)

# A relu Function
def reLu(Z):
    return Z*(Z>0)

# An entropy cost function used for the logistic regression
def sigmoid_cost(T,Y):
    return -(T*np.log(Y) + (1-T)*np.log(1-Y)).sum()

# A function to compute the error rate. Used to determine how well the algorithm is performing.
def error_rate_standart(T,Y):
    N = len(Y)
    error_count = 0.0
    for i in range(N):
        if T[i] != Y[i]:
            error_count +=1
    
    return float(error_count/N)

# Another Error rate function with exactly the same functionality as the above one.
def error_rate_mean(T,Y):
    return np.mean(T !=Y)

#Performs one-hot-encoding an array. It returns an indicator matrix that is then used for making preditions. This is especially useful 
# for multiclass classification problems.
def indicator_mat_conv(Ytrue):
    data_samples = len(Ytrue)
    dist_classes = len(set(Ytrue))
    indicator_mat = np.zeros((data_samples,dist_classes))
    
    for data_sample in range(data_samples):
        indicator_mat[data_sample,Ytrue[data_sample]] = 1
    return indicator_mat
# A cost function used for the Neural Network Model
def cost(T, Y):
    return -(T*np.log(Y)).sum()

def cost2(T, Y):
    # same as cost(), just uses the targets to index Y
    # instead of multiplying by a large indicator matrix with mostly 0s
    N = len(T)
    return -np.log(Y[np.arange(N), T]).mean()

# A function that is used to import the data into the model. 
def getData(balance_ones=True):
    # images are 48x48 = 2304 size vectors
    Y = []
    X = []
    first = True
    for line in open('fer2013.csv'):
        if first:
            first = False
        else:
            row = line.split(',')
            if row[0].isdigit():    
                Y.append(int(row[0]))
                X.append([int(p) for p in row[1].split()])

    X, Y = np.array(X) / 255.0, np.array(Y)
    return X,Y

# Outputs the different classes and the number of samples in each class
X, Y  = getData()
#print Y
count0 = 0
count1 = 0
count2 = 0
count3 = 0
count4 = 0
count5 = 0
count6 = 0
count7 = 0
count8 = 0
count9 = 0

for i in range(len(Y)):
    if Y[i]==0:
        count0 +=1
    if Y[i]==1:
        count1 +=1
    if Y[i]==2:
        count2 +=1
    if Y[i]==3:
        count3 +=1
    if Y[i]==4:
        count4 +=1
    if Y[i]==5:
        count5 +=1
    if Y[i]==6:
        count6 +=1
    if Y[i]==7:
        count7 +=1
    if Y[i]==8:
        count8 +=1
    else:
        count9 +=1

print("0:",count0,"1:",count1,"2:",count2,"3:",count3,"4:",count4,"5:",count5,"6:",count6,"7:",count7,"8:",count8,"9:",count9)
    
            