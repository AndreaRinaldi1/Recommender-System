import numpy as np
import pandas as pd
from scipy.stats import wishart
import math
import time
import IOUtil
import CV

def bpmf(Y, Ytest, featureCount, itemCount, userCount, nrRatings):
	#################SOURCE##########################
	#This is an implementation of the paper :
	#Bayesian Probabilistic Matrix factorization using Markov Chain Monte Carlo
	#Ruslan Salakhutdinov, Andriy Mnih, University of Toronto
	#####################NOTATION####################
	#In this implementation, we try to follow the notation of the paper symbolic wise
	#whenever possible. The used notations are described here:
	#-Use of var_u when it is related to user, use of var_i when related to item
	#-U bar: Ubar
	#-S bar: Sbar
	#-beta 0: B0
	#-capital alpha: A_u or A_i
	#-small alpha (e.g Eq.12): alpha
	#-nu (degree of freedom for the Wishart distribution) : v
	#The other variable names should be self explanatory with regard to the bpmf paper
	#################################################

	#Initialize the hyperparameters with values proposed in paper
	M = userCount
	N = itemCount
	W_u = np.identity(featureCount)
	v_u = featureCount
	mu0_u = np.zeros(featureCount)
	W_i = np.identity(featureCount)
	v_i = featureCount
	mu0_i = np.zeros(featureCount) 

	alpha = 2 #precision (suggested in the paper)
	B0_u = 2 #(using CV)
	B0_i = 2

	
	ratingsCount = nrRatings*1.0;
	mean_rating = np.sum(Y)/ratingsCount
	offset = mean_rating
	
	#Initialize the parameters using MAP solution by a probabilistic matrix factorization
	U = np.random.standard_normal((userCount,featureCount))
	V = np.random.standard_normal((featureCount,itemCount))
	mu_u = np.average(U,axis=0)
	mu_i = np.average(V,axis=1)
	A_u = np.linalg.inv(np.cov(U.T))
	A_i = np.linalg.inv(np.cov(V))

	#Iteration
	maxIter = 20
	previousLoss = 10
	for i in range(maxIter):
		start = time.time()
		print("Iteration "+str(i))

		#Sample user hyperparameters (Eq.14 of the paper)
		Ubar = np.average(U,axis=0)
		Sbar = np.cov(U.T)
		W_u_rightTerm = (1.0*M*B0_u/(B0_u*1.0+M))*np.outer(mu0_u-Ubar,mu0_u-Ubar)
		W_u = np.linalg.inv(np.linalg.inv(W_u)+(M*Sbar)+W_u_rightTerm)
		W_u = (W_u+W_u.T)*0.5
		v_u = v_u+M
		A_u = wishart.rvs(v_u,W_u) 
		mu_u = multivariate_gaussian_randomSamples(np.linalg.inv(A_u*(B0_u+M)), (mu0_u*B0_u+M*Ubar)*(1.0/(B0_u+M)), featureCount) 
		mu0_u = (B0_u*mu0_u+M*Ubar*1.0)/(B0_u+M)
		B0_u = B0_u+M		
		


		#Sample from item hyperparameters (Eq. 14 of the paper)
		Ubar = np.average(V,axis=1)
		Sbar = np.cov(V)
		W_u_rightTerm = np.outer(mu0_i-Ubar,mu0_i-Ubar)*(1.0*N*B0_i/(B0_i*1.0+N))
		W_i = np.linalg.inv(np.linalg.inv(W_i)+(N*Sbar)+W_u_rightTerm)
		W_i = (W_i+W_i.T)*0.5
		v_i = v_i+N
		A_i = wishart.rvs(v_i,W_i)
		mu_i = multivariate_gaussian_randomSamples(np.linalg.inv(A_i*(B0_i+N)), (mu0_i*B0_i+N*Ubar)*(1.0/(B0_i+N)), featureCount)
		mu0_i = (B0_i*mu0_i+N*Ubar*1.0)/(B0_i+N)
		B0_i = B0_i+N

		#Gibbs updates over U and V
		for gibbs in range(2):
			#Infer posterior distribution for U 
			for user in range(userCount):
				indexNonZeroRatings = np.nonzero(Y[user,:])[0]
				ratedItems = Y[user,indexNonZeroRatings]
				correspFeat = V[:,indexNonZeroRatings].T
				normRatedItems = ratedItems-mean_rating

				covar = np.linalg.inv(A_u + np.matmul(correspFeat.T,correspFeat)*alpha) #Eq 12
				a = np.matmul(correspFeat.T,np.vstack(normRatedItems))*alpha
				b = np.matmul(A_u,np.vstack(mu_u))
				mean_u = np.matmul(covar,np.vstack(a+b)) #Eq 13

				Ui = multivariate_gaussian_randomSamples(covar,mean_u, featureCount)	#Eq 11
				U[user,:]=np.hstack(Ui)


			#Infer posterior distribution for V
			for item in range(itemCount):
				indexNonZeroRatings = np.nonzero(Y[:,item])[0]
				ratedUsers = Y[indexNonZeroRatings,item]
				correspFeat = U[indexNonZeroRatings,:]
				normRatedUsers = ratedUsers-mean_rating

				covar = np.linalg.inv(A_i + np.matmul(correspFeat.T,correspFeat)*alpha) #Eq 12
				a = np.matmul(correspFeat.T,np.vstack(normRatedUsers))*alpha
				b = np.matmul(A_i,np.vstack(mu_i))
				mean_i = np.matmul(covar,np.vstack(a+b)) #Eq 13

				Vi = multivariate_gaussian_randomSamples(covar,mean_i, featureCount) #Eq 11
				V[:,item]=Vi.T
		
		#Do early stopping
		print("time for the iteration : "+str(time.time()-start))
		Xfull = np.matmul(U,V)+offset
		loss = evaluate(Xfull, Ytest)
		print(loss)
		if loss>previousLoss:
			return U, V, offset

		previousLoss = loss

	return U, V, offset

def predict(U,V,offset):
	return np.matmul(U,V)+offset

def multivariate_gaussian_randomSamples(cov, mean, featureCount):
	#We use the cholesky decomposition for generating multivariate gaussian samples
	L = np.linalg.cholesky(cov)
	uncorrelated = np.random.standard_normal(featureCount)
	return np.hstack(np.dot(L,uncorrelated)) + np.ndarray.flatten(mean)


def evaluate(Xcompleted, Xtest):
	nonzeros = np.nonzero(Xtest)
	squareSum = 0
	nrRatings = 0
	for i in range(np.shape(nonzeros[0])[0]):
		h = nonzeros[0][i]
		w = nonzeros[1][i]
		squareSum+=(Xtest[h,w]-Xcompleted[h,w])**2
		nrRatings+=1
	return math.sqrt(squareSum*1.0/nrRatings)


if __name__ == "__main__":
	height = 10000
	width = 1000
	X = np.load("../data/TrainSet.npy")
	Xtrain, Xval, nrTrain, nrVal = CV.splitNpy(X, height,width,0.05)
	features = 3
	U, V, offset = bpmf(Xtrain,Xval, features,width,height,nrTrain)
	Xcomplete = predict(U,V,offset)	
	IOUtil.writeFile(Xcomplete)
	np.save("BPMF.npy",Xcomplete)
	


