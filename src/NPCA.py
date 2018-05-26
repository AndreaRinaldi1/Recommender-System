import numpy as np
import pandas as pd
import math
import time



def parseId(stringId):
	splits = stringId.split("_")
	return int(splits[0][1:])-1,int(splits[1][1:])-1

def fillMatrix(height,width,val):
	X = np.ones(shape=(height,width))*val
	df = pd.read_csv("data_train.csv")
	ids=np.array(df['Id'])
	pred=np.array(df['Prediction'])
	for i in range(np.shape(ids)[0]):
		row,col=parseId(ids[i])
		X[row,col]=pred[i]
	return X

def npca(Y):
	N = np.shape(Y)[1] #The width (items)
	M = np.shape(Y)[0] #The height (users)
	K = np.identity(N)
	ratingsCount = 1176952.0;
	average = np.sum(Y)/ratingsCount
	ratingSquaredSum = squaredDiffAvg(Y,average)
	K = K*(ratingSquaredSum/ratingsCount*1.0)
	mu = np.full(N,average)
	
	iterMax = 50; #used for the global convergence. Could be replaced by a while loop and an error measure on a validation set
	for iter in range(iterMax):
		print(iter)
		B = np.zeros((N,N))
		b = np.zeros(N)
		start = time.time()
		for i in range(M):
			Oi = np.nonzero(Y[i,:])[0]
			G = np.linalg.inv(K[np.ix_(Oi,Oi)])
			t = np.matmul(G,(Y[i,Oi]-mu[Oi]).T)
			b[Oi] = b[Oi]+np.hstack(t)
			B[np.ix_(Oi,Oi)] = B[np.ix_(Oi,Oi)] - G + np.outer(t,t)
			if(i%1000==0):
				print(time.time()-start)
				start = time.time()

		mu = mu + (1.0/M)*b
		K = K + (1.0/M)*np.matmul(np.matmul(K,B),K)
		
	return K,mu

def squaredDiffAvg(Y,average):
	M,N = np.shape(Y)
	squaredSum = 0
	for i in range(M):
		observed = np.nonzero(Y[i,:])[0]
		squaredSum += np.sum(np.power(Y[i,observed]-average,2))
	return squaredSum
		
def predict(i,j,Y,K,mu):
	Oi = np.nonzero(Y[i,:])[0]
	first = K[j,Oi]
	second = np.linalg.inv(K[np.ix_(Oi,Oi)]) 
	third = np.vstack((Y[i,Oi]-mu[Oi]))
	return np.matmul(np.matmul(first,second),third) + mu[j]
	

def writeToCSV(X,K,mu):
	df = pd.read_csv("sampleSubmission.csv")
	ids=np.array(df['Id'])
	predictions=np.zeros(np.shape(ids)[0])
	print("Number of predictions to make: " +str(np.shape(ids)[0]))
	start = 0
	j=0
	for i in range(np.shape(ids)[0]):
		row,col=parseId(ids[i])
		predictions[i] = predict(row,col,X,K,mu)
		if(i%10000 == 0):
			if(i!=0):
				print(str(j) +" , time "+str(time.time()-start))
				j+=1
			start = time.time();
	df = pd.DataFrame({'Id':np.ndarray.flatten(ids),'Prediction':np.ndarray.flatten(predictions)})
	df.to_csv("SubmissionNPCA.csv",index=False)
	np.save("matrix", X)

height = 10000
width = 1000

X = fillMatrix(height,width,0)
K, mu = npca(X)
writeToCSV(X,K,mu)




