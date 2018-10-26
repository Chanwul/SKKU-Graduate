import numpy as np
import pandas as pd
from copy import deepcopy

def data_load():
    x = pd.read_csv("./training.csv")
    x.head()
    data = x.values[:, 1:8]
    label = x.values[:, 0]
    return data, label


def eval(prediction, y):
    hit = 0
    for i, p in enumerate(prediction):
        if p == y[i]:
            hit += 1
    return hit/len(y)


def measure(data, i, arr):   # measure distance
    return np.linalg.norm(data - arr[i], axis = 1)


def Init(data, k):
    mean = np.mean(data, axis = 0)
    dev = np.std(data, axis = 0)
    return np.random.randn(k, data.shape[1]) * dev + mean


def main():
    data , diabetes = data_load()
    k = 2
    num = data.shape[0]
    centroid = Init(data, k)
    cur = deepcopy(centroid)    # current centroid
    previous = np.zeros(centroid.shape) # previous centroid
    clusters = np.zeros(num)
    distances = np.zeros((num, k))

    while np.linalg.norm(cur - previous) != 0:  # until distance is not 0
        for i in range(k):
            distances[:, i] = measure(data, i, centroid) # measure distance
        clusters = np.argmin(distances, axis = 1) # assign clusters
        for i in range(k):
            if clusters[i] == i:
                cur[i] = np.mean(data[i], axis = 0)   # caclulate distance from centroid
        previous = deepcopy(cur)  # update previous

    accuracy = eval(clusters, diabetes)
    if accuracy < 0.5:
        for i in range(num):
            clusters[i] = clusters[i] ^ 1
    accuracy = eval(clusters, diabetes)
    print(accuracy)
    for i in range(num):
        print (clusters[i])


if __name__ == "__main__":
    main()