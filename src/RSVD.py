import numpy as np
import pandas as pd
import datetime
import time
import random
import math


def parseId(stringId):
    splits = stringId.split("_")
    return int(splits[0][1:])-1,int(splits[1][1:])-1


def initialization():
    full_matrix = np.load("TrainSet.npy")
    rows, cols = np.where(full_matrix != 0)
    Ind = np.zeros((rows.size, 2), dtype=np.int)
    for i in range(rows.size):
        Ind[i] = np.array([rows[i], cols[i]])
    return full_matrix, Ind


def hide_values(Ind):
    x = Ind[np.argsort(Ind[:, 0])]
    index = 0
    hidden_values = []
    while index < len(x):
        count = 0.0
        while index+1 < len(x) and x[index+1,0] == x[index,0]:
            count += 1.0
            index += 1
        #print(count)
        arg_hidden_values = [random.randint(0, count) for _ in range(math.ceil(count/10.0))]
        #print([random.randint(0, count) for _ in range (math.ceil(count/10.0)) ])
        for i in arg_hidden_values:
            hidden_values.append(x[index-i])
        index += 1
    return hidden_values


def create_training_set():
    hidden_values = hide_values(Ind)

    training_matrix = full_matrix.copy()
    for row, col in hidden_values:
        training_matrix[row, col] = 0

    remaining_values = []
    for i in range(0, len(full_matrix)):
        for j in range(0, len(full_matrix[0])):
            if training_matrix[i, j] != unknown_value:
                remaining_values.append([i, j])

    return training_matrix, remaining_values, hidden_values


def compute_biases():
    # Compute the bias of the users
    for i in range(num_users):
        b_u[i] = np.mean(training_matrix[i, :][np.where(training_matrix[i, :] != unknown_value)])

    # Compute the global mean of the ratings
    b = np.mean(training_matrix[np.where(training_matrix != unknown_value)])

    # Compute the weights of every user as the ratio btw his avg rating and the global svg rating
    w_u = b_u / b

    # Compute the bias for the movies considering the just computed weights (ratings from users that usually give higher ratings are
    # considered with a smaller percentage and viceversa for ratings from users that have smaller weights)
    for j in range(0, len(training_matrix[0])):
        ratings_movie = []
        for i in range(0, len(training_matrix)):
            if training_matrix[i, j] != unknown_value:
                ratings_movie.append(training_matrix[i, j] / w_u[i])
        b_m_weighted.append(np.mean(ratings_movie))
        ratings_movie = []


def fill_matrix():
    # Each missing value in the initial matrix is filled with the weighted mean for that movie multiplied by the weight of
    # that user (users that give higher ratings on average, will see higher values)
    for i in range(0, len(training_matrix)):
        for j in range(0, len(training_matrix[0])):
            if training_matrix[i, j] == unknown_value:
                training_matrix[i, j] = w_u[i] * b_m_weighted[j]

    return training_matrix


def writeFile(X):
    df = pd.read_csv("sampleSubmission.csv")
    ids=np.array(df['Id'])
    predictions=np.zeros(np.shape(ids)[0])
    for i in range(np.shape(ids)[0]):
        row,col=parseId(ids[i])
        predictions[i] = X[row,col]
    df = pd.DataFrame({'Id':np.ndarray.flatten(ids),'Prediction':np.ndarray.flatten(predictions)})
    now = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    df.to_csv("mySubmission"+now+".csv",index=False)


def get_rating(user, movie):
    return b_u[user] + b_m_weighted[movie] + U[user, :].dot(ZT[:, movie])


def RMSE():
    error = 0
    for usr, mov in hidden_values:
        error += pow(full_matrix[usr, mov] - get_rating(usr, mov), 2)
    return np.sqrt(error / len(hidden_values))


def sgd():
    for usr, mov in remaining_values:
        #Compute error
        prediction = get_rating(usr, mov)
        e = (training_matrix[usr, mov] - prediction)

        #Update biases
        tmp_b_u = b_u[usr]
        b_u[usr] += lr * (e - reg_b * (b_u[usr] + b_m_weighted[mov] - b))
        b_m_weighted[mov] += lr * (e - reg_b * (b_m_weighted[mov] + tmp_b_u - b))

        # Update user and item latent feature matrices
        tmp_U = U[usr, :]
        U[usr, :] += lr * (e * ZT[:, mov] - reg * U[usr, :])
        ZT[:, mov] += lr * (e * tmp_U - reg * ZT[:, mov])



def getFullMatrix():
    complete_matrix = np.zeros((num_users, num_movies))
    for i in range(num_users):
        for j in range(num_movies):
            complete_matrix[i, j] = get_rating(i, j)
    return complete_matrix

'''
def training(lr, reg, reg_b, factor):
    # weights = np.asarray([random.uniform(0.7, 1.3) for _ in range(factor)])
    # print(weights)
    full_matrix, Ind = initialization()
    training_matrix, remaining_values, hidden_values = create_training_set(full_matrix, Ind)
    b, b_u, w_u, b_m_weighted = compute_biases(training_matrix)
    training_matrix = fill_matrix(training_matrix, w_u, b_m_weighted)
    print("Computing SVD")
    U, s, ZT = np.linalg.svd(training_matrix, full_matrices=True)
    D = np.diag(s[:factor])
    U = np.matmul(U[:, :factor], np.sqrt(D))
    ZT = np.matmul(np.sqrt(D), ZT[:factor, :])

    training_process = []
    mse = 10000
    prev_mse = 100000
    i = 0
    while prev_mse - mse > 0.0005:
        prev_mse = mse

        np.random.shuffle(Ind)
        U, ZT, b_u, b_m_weighted = sgd(lr, reg, reg_b, training_matrix, remaining_values, U, ZT, b, b_u, b_m_weighted)
        b = np.mean(getFullMatrix(num_users, num_movies, U, ZT, b_u, b_m_weighted))
        mse = RMSE(full_matrix, hidden_values, U, ZT, b_u, b_m_weighted)
        training_process.append(mse)
        # print(U)
        # print(ZT)
        print("Iteration: %d ; error = %.4f" % (i + 1, mse))
        i += 1
    return mse, getFullMatrix(num_users, num_movies, U, ZT, b_u, b_m_weighted)
'''

num_users = 10000
num_movies = 1000
unknown_value = 0

lr = 0.005

for reg in [0.02, 0.05, 0.1]:
    for reg_b in [0.02, 0.05, 0.1]:
        for factor in [3, 6, 9, 12, 15]:
            for iteration in range(2):
                print("RSVD_R" + str(reg) + "_RB" + str(reg_b) + "_F" + str(factor) + "_" + str(iteration))

                full_matrix, Ind = initialization()
                training_matrix, remaining_values, hidden_values = create_training_set()

                b_u = np.zeros(num_users)
                b = 0
                w_u = np.zeros(num_users)
                b_m_weighted = []

                compute_biases()
                fill_matrix()
                print("Computing SVD")
                U, s, ZT = np.linalg.svd(training_matrix, full_matrices=True)
                D = np.diag(s[:factor])
                U = np.matmul(U[:, :factor], np.sqrt(D))
                ZT = np.matmul(np.sqrt(D), ZT[:factor, :])

                training_process = []
                mse = 10000
                prev_mse = 100000
                i = 0
                while prev_mse - mse > 0.0005:
                    prev_mse = mse

                    np.random.shuffle(Ind)
                    sgd()
                    b = np.mean(getFullMatrix())
                    mse = RMSE()
                    training_process.append(mse)

                    print("Iteration: %d ; error = %.4f" % (i + 1, mse))
                    i += 1


                #mse, final = training(lr, reg, reg_b,  factors)
                final = np.clip(getFullMatrix(), 1, 5)
                #now = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
                np.save("RSVD_R" + str(reg) + "_RB" + str(reg_b) + "_F" + str(factor) + "_E" + str(mse) + "_" + str(iteration), final)
                #writeFile(final)