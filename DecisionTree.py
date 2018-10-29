import pandas as pd
import math
from sklearn.metrics import classification_report, confusion_matrix


def data_load(data):
    file = None
    if data == 'train':
        file = './train.csv'

    elif data == 'test':
        file = './test.csv'

    dt = pd.read_csv(file)
    attr = ['age', 'ho_incm', 'edu', 'cfam', 'D_1_1', 'DI1_dg', 'HE_chol', 'HE_BMI']
    return dt, attr


def entropy(data, attr):    # entropy 계산
    pos, neg = count(data, attr)
    if pos == 0 or neg == 0:
        return 0
    else:
        return ((-1*pos)/(pos + neg))*math.log(pos/(pos+neg), 2) + ((-1*neg)/(pos + neg))*math.log(neg/(pos+neg), 2)    # entropy 리턴


def information_gain(data, attribute, attr, threshold):
    n = data.shape[0]
    a = 0.0
    first = data[data[attribute] < threshold]
    second = data[data[attribute] > threshold]
    for x in [first, second]:
        if x.shape[0] > 1:
            a += float(x.shape[0]/n) * entropy(x, attr)
    info_gain = entropy(data, attr) - a
    return info_gain


def calc_thres(data, attribute, attr):  # threshold 계산
    values = data[attribute].values[:]
    values = [float(x) for x in values]
    values = set(values)
    values = list(values)
    values.sort()
    gain_max = float("-inf")
    threshold = 0
    for i in range(0, len(values)-1):
        th = (values[i] + values[i+1])/2
        gain = information_gain(data, attribute, attr, th)
        if gain > gain_max:
            gain_max = gain
            threshold = th
    return threshold    # gain이 max인 threshold 리턴


class node(object):
    def __init__(self, attribute, threshold):
        self.attr = attribute
        self.th = threshold
        self.left = 0
        self.right = 0
        self.leaf = False
        self.predict = 0


def count(data, attr):
    temp = data.values[data[attr] == 1]
    temp2 = data.values[data[attr] == 0]
    a1 = temp.shape[0]
    a2 = temp2.shape[0]
    return a1, a2


def choose(data, attribute, attr):
    mx = float("-inf")
    nx = None
    threshold = 0
    for i in attribute:
        temp = calc_thres(data, i, attr)
        gain = information_gain(data, i, attr, temp)
        if gain > mx:
            mx = gain
            nx = i
            threshold = temp
    return nx, threshold


def generate(data, attributes, attr):
    pos, neg = count(data, attr)
    if pos <= 10 or neg <= 10:
        leaf = node(None, None)
        leaf.leaf = True
        if pos > neg:
            leaf.predict = 1
        else:
            leaf. predict = 0
        return leaf
    else:
        attr, th = choose(data, attributes, attr)
        tree = node(attr, th)
        tree.left = generate(data[data[attr] < th], attributes, attr)
        tree. right = generate(data[data[attr] > th], attributes, attr)
        return tree


def predict(cur, n):
    if cur.leaf:
        return cur.predict
    if n[cur.attr] <= cur.th:
        return predict(cur.left, n)
    elif n[cur.attr] > cur.th:
        return predict(cur.right, n)


def eval(root, data, result):
    num = data.shape[0]
    ans = 0
    prediction = []
    for index, row in data.iterrows():
        a = predict(root, row)
        prediction.append(a)
        if prediction[index] == row['DE1_dg']:    # 결과가 같으면
            ans += 1
    print(prediction)
    print(confusion_matrix(result, prediction))
    print(classification_report(result, prediction))
    print("Accuracy = ", ans/num)


def main():
    train_data, train_attributes = data_load("train")
    test_data, test_attributes = data_load("test")
    result = test_data.values[:, 0]
    root = generate(train_data, train_attributes, 'DE1_dg')
    eval(root, test_data, result)


if __name__ == '__main__':
    main()