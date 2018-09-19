import numpy as np

class LogisticRegression():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.num_data = self.x.shape[0]
        self.num_feature = self.x.shape[1]
        self.w = np.zeros(self.num_feature)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, x):
        temp2 = np.dot(x, self.w)
        predic = self._sigmoid(temp2)
        index = np.where(predic > 0.5)[0]
        result = np.zeros(predic.shape)
        result[index] = 1
        return result

    def train(self, lr, epoch):
        for i in range(0, epoch):
            temp = np.dot(self.x,self.w)
            grad = np.dot((self.y - self._sigmoid(temp)),self.x)
            self.w = self.w + lr * grad
        return self.w
