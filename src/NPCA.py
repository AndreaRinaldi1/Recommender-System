import numpy as np
import pandas as pd
import math
import time
import CV
import IOUtil

#This is an implementation of the paper:
#Fast Nonparametric Matrix Factorization for Large-scale Collaborative Filtering
#Kai Yu, Shenghuo Zhu, John Lafferty,Yihong Gong
def npca(Y, Yval, nrTrain, nrTest):
	N = np.shape(Y)[1] #The width (items)
	M = np.shape(Y)[0] #The height (users)
	K = np.identity(N)
	ratingsCount = nrTrain*1.0
	average = np.sum(Y)/ratingsCount
	ratingSquaredSum = squaredDiffAvg(Y,average)
	K = K*(ratingSquaredSum/ratingsCount*1.0)
	mu = np.full(N,average)
	previousLoss = 10	

	iterMax = 50; 
	for iter in range(iterMax):
		print(iter)
		B = np.zeros((N,N))
		b = np.zeros(N)
		for i in range(M):
			Oi = np.nonzero(Y[i,:])[0]
			G = np.linalg.inv(K[np.ix_(Oi,Oi)])
			t = np.matmul(G,(Y[i,Oi]-mu[Oi]).T)
			b[Oi] = b[Oi]+np.hstack(t)
			B[np.ix_(Oi,Oi)] = B[np.ix_(Oi,Oi)] - G + np.outer(t,t)
			if(i%1000==0):
				print(i)

		mu = mu + (1.0/M)*b
		K = K + (1.0/M)*np.matmul(np.matmul(K,B),K)
		soFarMat = fillValidationMatrix(Y, Yval, K, mu)
		loss = math.sqrt((((soFarMat-Yval)**2).sum())/nrTest)
		print(loss)
		if loss>previousLoss:
			return K,mu
		previousLoss = loss
	return K,mu

def predict(i,j,Y,K,mu):
	Oi = np.nonzero(Y[i,:])[0]
	first = K[j,Oi]
	second = np.linalg.inv(K[np.ix_(Oi,Oi)]) 
	third = np.vstack((Y[i,Oi]-mu[Oi]))
	return np.matmul(np.matmul(first,second),third) + mu[j]


def squaredDiffAvg(Y,average):
	M,N = np.shape(Y)
	squaredSum = 0
	for i in range(M):
		observed = np.nonzero(Y[i,:])[0]
		squaredSum += np.sum(np.power(Y[i,observed]-average,2))
	return squaredSum

def fillValidationMatrix(Xtrain, Xval, K, mu):
	nonzeros = np.nonzero(Xval)
	Xpred = np.zeros((10000,1000))
	for i in range(np.shape(nonzeros[0])[0]):
		h = nonzeros[0][i]
		w = nonzeros[1][i]
		Xpred[h,w] = predict(h,w,Xtrain, K, mu)
	return Xpred
		


if __name__ == "__main__":	
	height = 10000
	width = 1000
	X = np.load("../data/TrainSet.npy")
	Xtrain, Xval, nrTrain, nrVal = CV.splitNpy(X, 10000,1000,0.05)
	K, mu = npca(Xtrain, Xval, nrTrain, nrVal)

	#Predict the complete matrix for model combining
	Xcomplete = np.zeros((10000,1000))
	for i in range(10000):
		for j in range(1000):
			Xcomplete[i,j] = predict(i,j,X, K, mu)
	np.save("FullMatrixNPCA.npy", Xcomplete) 
	IOUtil.writeFile(Xcomplete)




