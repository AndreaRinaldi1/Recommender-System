import numpy as np
import pandas as pd
import datetime
import time
import IOUtils
import CV


def sgd():
    """
    computes the stochastic gradient descent of the matrix associating users to latent factors U,
    the matrix that associates items to latent factors ZT and the biases of users and items.
    """
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
    """Each missing value in the initial matrix is filled with the weighted mean
    for that movie multiplied by the weight of that user
    (users that give higher ratings on average, will see higher values)
    return: the filled matrix
    """
    for i in range(0, len(training_matrix)):
        for j in range(0, len(training_matrix[0])):
            if training_matrix[i, j] == unknown_value:
                training_matrix[i, j] = w_u[i] * b_m_weighted[j]

    return training_matrix


def get_rating(user, movie):
    return b_u[user] + b_m_weighted[movie] + U[user, :].dot(ZT[:, movie])


def RMSE():
    error = 0
    for usr, mov in hidden_values:
        error += pow(full_matrix[usr, mov] - get_rating(usr, mov), 2)
    return np.sqrt(error / len(hidden_values))


def getFullMatrix():
    complete_matrix = np.zeros((num_users, num_movies))
    for i in range(num_users):
        for j in range(num_movies):
            complete_matrix[i, j] = get_rating(i, j)
    return complete_matrix



num_users = 10000
num_movies = 1000
unknown_value = 0

lr = 0.005

for reg in [0.01, 0.02, 0.05]:
    for reg_b in [0.02, 0.05, 0.1, 0.15]:
        for factor in [3, 6, 9, 12, 15]:
            for iteration in range(2):
                print("RSVD_R" + str(reg) + "_RB" + str(reg_b) + "_F" + str(factor) + "_" + str(iteration))

                full_matrix, Ind = IOUtils.initialization()
                training_matrix, remaining_values, hidden_values = CV.create_training_set(Ind, full_matrix, unknown_value)

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

                final = np.clip(getFullMatrix(), 1, 5)
                #now = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
                np.save("RSVD_R" + str(reg) + "_RB" + str(reg_b) + "_F" + str(factor) + "_E" + str(mse) + "_" + str(iteration), final)
                #writeFile(final)
