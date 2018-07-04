import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import datetime
import time
import IOUtils

# input:
# 1) userItemMatrix: Is the matrix which has as rows the user and as columns the userRatings
# 2) similarityMatrix: Is the matrix which has as rows and columns the users and as generic element the similarity between user i and user j
# 3) user: which is an index to obtain all the ratings given the the user
# 4) item : all the items necessary to compute the item_prediction
#return:
# 1) res: the computed prediction for the user

def predict(userItemMatrix, similarityMatrix, user, item):
    simCoeff = similarityMatrix[item, :]
    userRatings = userItemMatrix[user, :]  # all the ratings given by the user u

    mask = np.nonzero(userRatings)

    valSimCoeff = simCoeff[mask]
    valUrat = userRatings[mask]

    num = np.sum(valSimCoeff * valUrat)
    dem = np.sum(np.absolute(valSimCoeff))
    res = num / dem
    return res

#input:
# 1) userItemMatrix: Is the matrix which has as rows the user and as columns the userRatings
# 2) similarityMatrix: Is the matrix which has as rows and columns the users and as generic element the similarity between user i and user j
#return
#1) userItemMatrix: it is the total matrix without missing values and filled with all the prediction

def predictTot(userItemMatrix, similarityMatrix):
    row = userItemMatrix.shape[0]
    col = userItemMatrix.shape[1]

    print("I am starting the prediction")

    for i in range(row):

        if (i % 1000 == 0):
            print("I am the user:" + str(i))
        for j in range(col):
            prediction = predict(userItemMatrix, similarityMatrix, i, j)
            userItemMatrix[i][j] = prediction
    return userItemMatrix

#input:
# 1) userItemMatrix: Is the matrix which has as rows the user and as columns the userRatings
#return
# 1) similarityMatrix: Is the matrix which has as rows and columns the users and as generic element the similarity between user i and user j
#                      the similarity used is the pearson coefficient
def pearsonSim(userItemMatrix):
    r = np.repeat(range(userItemMatrix.shape[1]), userItemMatrix.shape[1])
    t = np.tile(range(userItemMatrix.shape[1]), userItemMatrix.shape[1])
    pearMatrix = np.zeros(len(r))
    for i in range(len(r)):

        if i % 10000 == 0:
            print(i)

        ridx = r[i]
        tidx = t[i]

        item1 = userItemMatrix[:, ridx]
        item2 = userItemMatrix[:, tidx]
        pearson_sim = pearsonr(item1, item2)[0]
        pearMatrix[i] = pearson_sim

    pearMatrix = np.reshape(pearMatrix, (n_movies, n_movies))
    #np.save("similarityMatrix", pearMatrix)
    return pearMatrix



n_users = 10000
n_movies = 1000

train_data, _ = IOUtils.initialization()

pearSim = pearsonSim(train_data)


item_pear_prediction = predictTot(train_data, pearSim )


item_pear_prediction = np.clip(item_pear_prediction, 1, 5)
now = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
np.save("item-itemPearson" + now + "", item_pear_prediction)
