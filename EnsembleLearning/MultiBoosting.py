# multiboosting for classification

import numpy as np
import pandas as pd
from sklearn import tree
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data from CSV and split training/test set
def LoadData(path, header):
    dataSet = pd.read_csv(path, header=header)
    label = dataSet[dataSet.columns[-1]]
    data = dataSet[dataSet.columns[0:-1]]
    trainSet, testSet, trainLab, testLab = train_test_split(data, label, test_size=0.5, random_state=0, stratify=label)
    x_train = trainSet.values
    y_train = trainLab.values
    x_test = testSet.values
    y_test = testLab.values
    return x_train, x_test, y_train, y_test

# set the number of members in each adaboost committee
def SetIteration(T):
    n = math.floor(math.sqrt(T))
    n = int(n)
    iteration = [T] * (5*T)
    for i in range(n):
        it = math.ceil(i * T / n)
        iteration[i] = int(it)
    return iteration

# set the possion weights
def SetPossionWeights(num_sample, T):
    n = math.floor(math.sqrt(T))
    weights = [1] * num_sample
    s = 0
    for i in range(num_sample):
        weights[i] = -math.log(np.random.randint(1, 1000) / 1000, math.e)
        s += weights[i]
    for i in range(num_sample):
        weights[i] /= s
    for i in range(num_sample):
        weights[i] *= n
    return weights

# train the member
def Train(x_train, y_train, weights):
    clf = tree.DecisionTreeClassifier()
    clf.fit(x_train, y_train, weights)
    # clf.fit(x_train, y_train)
    return clf

# train the MultiBoosting model, T is the total number of members, num_sample is the number of training samples
def MultiBoosting(T, x_train, y_train, num_sample):
    weights = [1] * num_sample
    iteration = SetIteration(T)
    beta = [0.0] * T
    k = 1
    classifiers = []
    for t in range(T):
        if iteration[k] == t:
            weights = SetPossionWeights(num_sample, T)
            k += 1
        Ct = Train(x_train, y_train, weights)
        classifiers.append(Ct)
        # a = Ct.predict(x_train)
        errors = y_train != np.reshape(Ct.predict(x_train), (len(y_train), 1))
        error = np.sum(weights*errors) / num_sample
        if error > 0.5:
            weights = SetPossionWeights(num_sample, T)
            k += 1
        elif error < 10 ** (-10):
            beta[t] = 10 ** (-10)
            weights = SetPossionWeights(num_sample, T)
            k += 1
        else:
            beta[t] = error / (1 - error)
            for i in range(num_sample):
                if errors[i]:
                    weights[i] /= (2 * error)
                else:
                    weights[i] /= (2*(1-error))
                if weights[i] < 10 ** (-8):
                    weights[i] = 10 ** (-8)
    return classifiers, beta # return the members (classifiers) and the weights of each members (beta)


# MultiBoosting Classifier Model
class MultiBoostingClassifier(object):
    def __init__(self, T): # Model initialization, decide the total number of members (T)
        self.classifiers = []
        self.beta = []
        self.classNum = 0
        self.T = T

    def fit(self, x_train, y_train): # Train the model
        n = x_train.shape[0]
        self.classNum = len(np.unique(y_train))
        self.classifiers, self.beta = MultiBoosting(self.T, x_train, y_train, n)

    def predictSample(self, sample): # Use the model to predict one instances, used in predict
        vote = [0] * self.classNum
        sample = np.reshape(sample, (1, len(sample)))
        for i in range(len(self.classifiers)):
            lab = self.classifiers[i].predict(sample)
            lab = lab[0]
            vote[lab] += math.log(1 / self.beta[i])
        cla = vote.index(max(vote))
        return cla

    def predict(self, x_test): # Use the model to predict new instances
        n = x_test.shape[0]
        pred = []
        for i in range(n):
            y = self.predictSample(x_test[i, :])
            pred.append(y)
        return pred

