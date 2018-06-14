import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
import IOUtils


def ensemble(known_ids, idsToWrite):
	"""
	Creates the features matrix for training the ensemble model on (validation set
	that has not been seen by the individual models to avoid overfitting) and
	creates the feature matrix for the prediction partin the same way.
	Computes also the weights of the samples according to their distance form the middle 0.5
	:param known_ids: the indices of the ratings to train the Ensemble model on
	:param idsToWrite: the indices the ensemble model has to predict the ratings of.
	:return: the train matrix, the prediction matrix and the list of weights of the samples
	"""

	path = "../../CIL_results/results/"
	files = [f for f in listdir(path) if isfile(join(path, f))]

	nrFiles = 0
	model_bias = getModelBias()
	for file in files:
		matrix = np.load(path + file)
		train = []
		pred = []
		print(str(file))
		for row, col in known_ids:
			train.append(matrix[row, col])
		for row2, col2 in idsToWrite:
			pred.append(matrix[row2, col2])
		diff = np.abs(np.round(train) - train)
		diffTo05 = (np.maximum(0.01, 0.5 - diff))**0.1
		weight = diffTo05 * model_bias[file]
		if nrFiles == 0:
			train_matrix = train
			pred_matrix = pred
			models_weights = weight
		else:
			train_matrix = np.vstack((train_matrix, train))
			pred_matrix = np.vstack((pred_matrix, pred))
			models_weights = np.vstack((models_weights, weight))
		nrFiles+=1

	weights_sum = np.sum(models_weights, axis=0)

	return train_matrix.T, pred_matrix.T, weights_sum


def regression(train_matrix, pred_matrix, groundTruth, weights):
	"""
	Fits an ensemble model (regressor) on the training matrix (features for the validation set)
	trying out different parameters with cross validation, and lastly predicts the unknown raitngs.
	:param train_matrix: features matrix to train the ensemble model on
	:param pred_matrix:  features matrix to make the ensemble model predict the unknown ratings.
	:param groundTruth: the true ratings for the training part
	:return: the wanted predictions
	"""
	'''mlp = MLPRegressor()
	params = {"hidden_layer_sizes": [(25,), (50,), (75,), (100,), (125,)], "activation":["relu"], "solver":["adam"],
			  "alpha": [0.0001, 0.0005, 0.001, 0.005, 0.01], "learning_rate": ["invscaling"]}
	grid = GridSearchCV(estimator=mlp, param_grid=params, iid=False, verbose=3, n_jobs=-1, scoring="neg_mean_squared_error")
	grid.fit(train_matrix, groundTruth)
	final_prediction = grid.predict(pred_matrix)
	print(grid.best_score_)
	print(grid.best_params_)
	return final_prediction'''
	ridge = Ridge()
	params = {"alpha" : [0.5, 1, 1.5, 2, 2.5, 3.5, 4], "fit_intercept" : [True, False]}
	grid = GridSearchCV(estimator=ridge, param_grid=params, iid=False, verbose=3, n_jobs=1, scoring='neg_mean_squared_error')
	grid.fit(train_matrix, groundTruth, **{'sample_weight': weights})
	final_prediction = grid.predict(pred_matrix)
	print(grid.best_score_)
	print(grid.best_params_)
	return final_prediction




def getModelBias():
	"""
	According to the performance of the individual models, we assign weights to them.
	:return: a dictionary mapping the individual models to the corresponding weights
	"""
	exp = 50
	return {'NPCA.npy': (1/1.0264)**exp, 'BPMF.npy': (1/0.99695)**exp, 'RSVD.npy': (1/0.98787)**exp,
			'Ridge.npy': (1/1.0264)**exp, 'item-itemPearson.npy': (1/1.05)**exp,
			'KMeans.npy': (1/1.06)**exp, 'Autoencoder.npy': (1/0.996)**exp}


def getIdsToWrite():
	"""
	Returns the indices of the predictions to write to csv for evaluation.
	"""
	df = pd.read_csv("../data/sampleSubmission.csv")
	ids=np.array(df['Id'])
	idsToWrite = np.zeros((len(ids), 2))
	for i in range(np.shape(ids)[0]):
		row,col=IOUtils.parseId(ids[i])
		idsToWrite[i] = row, col
	return idsToWrite.astype(int)


def groundTruth():
	"""
	As training for the Ensemble model we take a validation set that is 5% of the
	total number of known ratings in the original training matrix. These values have
	been hidden to the individual models, to avoid overfitting.
	:return: The array of indices of the ratings in the validation ser, and the actual true ratings.
	"""
	X = np.load("../data/ValidationSet.npy")
	rows, cols = np.where(X != 0)
	Ind = np.zeros((rows.size, 2), dtype=np.int)
	pred = np.zeros(rows.size)
	for i in range(rows.size):
		Ind[i] = np.array([rows[i], cols[i]])
		pred[i] = X[rows[i], cols[i]]
	return Ind.astype(int), pred


def main():
	known_ids, groundT = groundTruth()
	idsToWrite = getIdsToWrite()
	train_matrix, pred_matrix, weights = ensemble(known_ids, idsToWrite)
	final_prediction = regression(train_matrix, pred_matrix, groundT, weights)
	final_prediction = np.clip(final_prediction, 1, 5)
	IOUtils.writeFileEnsemble(final_prediction)



if __name__ == '__main__':
	main()

