import csv
import numpy as np
from sklearn.metrics import classification_report

class LogisticRegression():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.num_data = self.x.shape[0]
        self.num_feature = self.x.shape[1]
        self.w = np.zeros(self.num_feature)

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def predict(self, x):
        temp2 = np.dot(x, self.w)
        predic = self._sigmoid(temp2)
        index = np.where(predic > 0.5)[0]
        result = np.zeros(predic.shape)
        result[index] = 1
        return result

    def train(self, lr, epoch):
        for i in range(0, epoch):
            temp = np.dot(self.x, self.w)
            grad = np.dot((self.y - self._sigmoid(temp)), self.x)
            self.w = self.w + lr * grad
        return self.w


def data_load(data):
    y = []
    x = []
    file = None
    if data == 'train':
        file = './train.csv'

    elif data == 'test':
        file = './test.csv'

    with open(file, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)
        for row in csv_reader:
            y.append(float(row[0]))
            feature = [1] # bias
            for i in range(1, 9):
                feature.append(float(row[i]))
            x.append(feature)
        return np.array(x, dtype="f8"), np.array(y, dtype="f8")


def eval(prediction, y):
    hit = 0
    for i, p in enumerate(prediction):
        if p == y[i]:
            hit += 1
    print(classification_report(y, prediction, target_names=['negative', 'positive']))
    print(hit/len(y))


def main():
   X_train, Y_train = data_load('train')
   model = LogisticRegression(X_train, Y_train)
   model.train(0.000001, 50000)  # learning rate, epoch
   X_test, Y_test = data_load('test')
   prediction = model.predict(X_test)
   for i in range(prediction.size):
       print(prediction[i])
   eval(prediction, Y_test)


if __name__ == "__main__":
    main()


