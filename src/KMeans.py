import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import random
import datetime
import time
from os import listdir
from os.path import isfile, join


def parseId(stringId):
    splits = stringId.split("_")
    return int(splits[0][1:])-1,int(splits[1][1:])-1


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


def getMatrix():
    X = np.load("TrainSet.npy")
    rows, cols = np.where(X != 0)
    Ind = np.zeros((rows.size, 2), dtype=np.int)
    for i in range(rows.size):
        Ind[i] = np.array([rows[i], cols[i]])
    return Ind, X

def compute_centroids(users_in_centroid):

    centroid = np.zeros(width)
    total_mask = np.zeros(width)
    for user in users_in_centroid:
        mask = X[user, :] != 0
        total_mask += mask
        centroid += X[user, :]
    if np.any(total_mask == 0):
        for i in np.argwhere(total_mask == 0):
            a = X[:, i]
            a = a[np.where(a != 0)]
            mean = np.mean(a)
            centroid[i] = mean
        total_mask[total_mask == 0] = 1
    centroid = centroid / total_mask
    return centroid


def compute_initial_centroids(chosen_centroids, num_of_samples):
    distances = np.zeros((num_of_users, len(chosen_centroids)))
    for user in range(num_of_users):
        movies = np.where(X[user, :] != 0)
        for centroid in range(len(chosen_centroids)):
            distances[user, centroid] = np.sum(np.abs(X[user, movies] - chosen_centroids[centroid][movies]))

    user_distances = np.amin(distances, axis=1)
    user_distances = user_distances / np.sum(user_distances)
    samples = np.random.choice(a=num_of_users, size=random.choice(num_of_samples), replace=False, p=user_distances)
    return compute_centroids(samples)

def update_assignments(users_to_centroids, num_of_centroids, centroids):
    index_usr = 0
    total_error = np.zeros(num_of_centroids)
    while index_usr < len(Ind):
        sum_user = np.zeros(num_of_centroids)
        usr = Ind[index_usr, 0]
        beginning_index_usr = index_usr
        for index_mu in range(num_of_centroids):
            index_usr = beginning_index_usr
            while index_usr < len(Ind) and (
                    index_usr == beginning_index_usr or Ind[index_usr, 0] == Ind[index_usr - 1, 0]):
                mov = Ind[index_usr, 1]
                sum_user[index_mu] += (X[usr, mov] - centroids[index_mu, mov])**2
                index_usr += 1
        best_centroid = np.argmin(sum_user)
        users_to_centroids[usr] = best_centroid
        total_error[best_centroid] += np.min(sum_user)

    sum_total_error = np.sum(total_error)
    return sum_total_error, users_to_centroids


def kmeans_algorithm(num_of_centroids, total_number_of_iterations, num_of_samples):

    distinct_users = np.array(list(set(Ind[:, 0])))
    users_to_centroids = np.zeros((len(distinct_users), 2)).astype(int)
    users_to_centroids[:, 0] = distinct_users
    users_to_centroids = dict(users_to_centroids)

    num_of_users = 10000

    #compute first centroid
    first_centroid = compute_centroids(random.sample(range(num_of_users), random.choice(num_of_samples)))

    centroids = first_centroid

    # compute second centroid
    distances = np.zeros(num_of_users)
    for user in range(num_of_users):
        movies = np.where(X[user, :] != 0)
        distances[user] = np.sum(np.abs(X[user, movies] - centroids[movies]))

    distances = distances / np.sum(distances)
    samples = np.random.choice(a=num_of_users, size=500, replace=False, p=distances)

    centroids = np.vstack((centroids, compute_centroids(samples)))

    for i in range(num_of_centroids-2):
        centroids = np.vstack((centroids, compute_initial_centroids(centroids, num_of_samples)))

    num_of_iterations = 0
    sum_total_error = 10000000000
    prev_total_error = 10000000000000
    while prev_total_error > sum_total_error and num_of_iterations < total_number_of_iterations:
        prev_total_error = sum_total_error

        #Update samples assignments to centroids
        sum_total_error, users_to_centroids = update_assignments(users_to_centroids, num_of_centroids, centroids)

        print("Iteration: " + str(num_of_iterations) + ". Error: " + str(sum_total_error))

        centroids_to_users = [[] for i in range(0, num_of_centroids)]
        for k, v in users_to_centroids.items():
            centroids_to_users[v].append(k)
        centroids_to_users = dict(enumerate(centroids_to_users))

        #for centroid in centroids_to_users:
           # print(len(centroids_to_users[centroid]))

        #Update centroids positions
        centroids = np.zeros((num_of_centroids, width))
        for centroid in centroids_to_users:
            centroids[centroid] = compute_centroids(centroids_to_users[centroid])

        num_of_iterations += 1

    return np.sum(sum_total_error), centroids, users_to_centroids



def train(list_of_centroids):
    list_of_iter = [2, 25, 25, 25, 22, 20, 18, 16, 14, 12, 10]
    num_of_samples = [300, 400, 500]

    errors = np.ones(len(list_of_centroids))*100000000000000
    finalMatrix = np.zeros((10000, 1000))

    num_of_attempts = 2

    for index in range(len(list_of_centroids)):
        num_of_centroids = list_of_centroids[index]
        best_centroids = np.zeros((num_of_centroids, width))

        for i in range(0, num_of_attempts):
            total_error, centroids_tmp, users_to_centroids_tpm = kmeans_algorithm(num_of_centroids, list_of_iter[index], num_of_samples)
            if total_error < errors[index]:
                errors[index] = total_error
                print("new best result:" + str(total_error))
                best_centroids = centroids_tmp
                best_users_to_centroids = users_to_centroids_tpm
        np.save("centroids" + str(num_of_centroids), best_centroids)
        for user in best_users_to_centroids:
            finalMatrix[user, :] += best_centroids[best_users_to_centroids[user]]

    finalMatrix = finalMatrix / len(list_of_centroids)
    np.save("KMeans", finalMatrix)
    writeFile(finalMatrix)
    return errors


def plot_errors(list_of_centroids, errors):

    plt.figure(figsize=(20, 15))
    plt.plot(list_of_centroids, errors, linewidth=2)
    plt.xlabel("Number of centroids", size=25)
    plt.xticks(list_of_centroids, size=20)
    plt.ylabel("Error", size=25)
    plt.grid(color='lightgrey', linestyle='-.', linewidth=1)
    plt.title("KMeans", size=30)
    plt.xlim((0, list_of_centroids[-1] +1))
    plt.savefig("kmeans.png")


def main():
    list_of_centroids = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]

    errors = train(list_of_centroids)
    plot_errors(list_of_centroids, errors)


num_of_users = 10000
width = 1000
Ind, X = getMatrix()
Ind = Ind[Ind[:, 0].argsort()]

if __name__ == '__main__':
    main()