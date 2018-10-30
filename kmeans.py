import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.metrics import classification_report

def data_load():
    x = pd.read_csv("./train.csv")
    data = x.values[:, 1:9]
    label = x.values[:, 0]
    return data, label


def eval(prediction, diabetes):
    hit = 0
    for i, p in enumerate(prediction):
        if p == diabetes[i]:
            hit += 1
    print(classification_report(diabetes, prediction, target_names=['negative', 'positive']))
    print(hit / prediction.shape[0])


def measure(data, i, arr):   # measure distance
    return np.linalg.norm(data - arr[i], axis = 1)


def init(data, k):
    mean = np.mean(data, axis = 0)
    dev = np.std(data, axis = 0)
    return np.random.randn(k, data.shape[1]) * dev + mean


def negate(clusters, num):
    count0 = 0
    count1 = 0
    for i in range(num):
        if clusters[i] == 0:
            count0 += 1
        else:
            count1 += 1
    if count0 < count1:
        for i in range(num):
            clusters[i] = clusters[i] ^ 1


def main():
    data, diabetes = data_load()
    k = 2
    num = data.shape[0]
    result = np.zeros(num)
    for n in range(1000):
        centroid = init(data, k)
        cur = deepcopy(centroid)
        clusters = np.zeros(num)
        previous = np.zeros(centroid.shape)
        distances = np.zeros((num, k))

        while np.linalg.norm(cur - previous) != 0:
            for i in range(k):
                distances[:, i] = measure(data, i, centroid)
            clusters = np.argmin(distances, axis=1)
            for i in range(k):
                if clusters[i] == i:
                    cur[i] = np.mean(data[i], axis=0)
            previous = deepcopy(cur)  # update previous
        negate(clusters, num)
        result += clusters

    for i in range(num):
        if result[i] >= 500:
            result[i] = 1
        else:
            result[i] = 0
    eval(result, diabetes)
    for i in range(num):
        print (result[i])


if __name__ == "__main__":
    main()