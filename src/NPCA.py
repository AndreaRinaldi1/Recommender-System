import numpy as np
import pandas as pd
import math
import time



def parseId(stringId):
	splits = stringId.split("_")
	return int(splits[0][1:])-1,int(splits[1][1:])-1

def splitSet(validationPercentage, height,width, inp="data_train.csv"):

	df = pd.read_csv(inp)
	ids=np.array(df['Id'])
	pred=np.array(df['Prediction'])
	nrRatings = np.shape(ids)[0]
	nrTrain = (int)(nrRatings*(1-validationPercentage))
	nrTest = nrRatings-nrTrain
	indicesTrain = np.random.choice(nrRatings,nrTrain,replace=False)
	Xtrain = np.zeros((height,width))
	Xtest = np.zeros((height,width))

	for i in indicesTrain:
		row,col=parseId(ids[i])
		Xtrain[row,col] = pred[i]
		
	indicesTest = np.setdiff1d(np.arange(nrRatings), indicesTrain)
	for i in indicesTest:
		row,col=parseId(ids[i])
		Xtest[row,col] = pred[i]

	return Xtrain,Xtest,nrTrain, nrTest

def splitNpy(X, height, width, splitPercentage):
	nonzeros = np.nonzero(X)
	ids = np.empty(np.shape(nonzeros[0])[0], dtype="U12") 
	predictions = np.empty(np.shape(nonzeros[0])[0], dtype="i")
	for i in range(np.shape(nonzeros[0])[0]):
		if(i%100000 == 0):
			print(i)
		h = nonzeros[0][i]
		w = nonzeros[1][i]
		idString = "r"+str(h+1)+"_c"+str(w+1)
		pred = X[h,w]
		ids[i] = idString
		predictions[i] = pred
	
	df = pd.DataFrame({'Id':ids,'Prediction':predictions})
	df.to_csv("temporary.csv",index=False)	
	return splitSet(splitPercentage,height,width, inp="temporary.csv")

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
	#Predict the complete matrix for model combining
	Xcomplete = np.zeros((10000,1000))
	for i in range(10000):
		for j in range(1000):
			Xcomplete[i,j] = predict(i,j,X, K, mu)
	np.save("FullMatrixNPCA.npy", Xcomplete) 
	
	

height = 10000
width = 1000

X = np.load("TrainSet.npy")
Xtrain, Xval, nrTrain, nrVal = splitNpy(X, 10000,1000,0.05)
K, mu = npca(Xtrain, Xval, nrTrain, nrVal)
writeToCSV(Xtrain,K,mu)




