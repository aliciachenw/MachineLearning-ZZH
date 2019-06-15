# iterated bagging methods for regresssion
# can also be used in the 2-class classification

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import random

# Load CSV data and split training/test set
def LoadData(path, header):
    dataSet = pd.read_csv(path, header=header)
    label = dataSet[dataSet.columns[-1]]
    data = dataSet[dataSet.columns[0:-1]]
    trainSet, testSet, trainLab, testLab = train_test_split(data, label, test_size=0.5, random_state=0)
    x_train = trainSet.values
    y_train = trainLab.values
    x_test = testSet.values
    y_test = testLab.values
    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    return x_train, x_test, y_train, y_test


# Random sampling with replacement for bagging
def RandomSampling(x_train, y_train):
    data = np.hstack([x_train, y_train])
    serial_numbers = []
    for i in range(data.shape[0]):
        serial = random.randint(0, data.shape[0] - 1)
        if i == 0:
            samples = data[serial, :]
            samples = np.reshape(samples, (1, len(samples)))
        else:
            samples = np.vstack((samples, data[serial, :]))
        serial_numbers.append(serial)
    new_x_train = samples[:, 0:(samples.shape[1]-1)]
    new_y_train = samples[:, samples.shape[1]-1]
    new_y_train = np.reshape(new_y_train, (len(new_y_train), 1))
    return new_x_train, new_y_train, serial_numbers


# Train Bagging model, numModel is the number of the members in bagging committee
def Bagging(numModel, x_train, y_train):
    samples_serial = []
    bagging_regressors = []
    for i in range(numModel):
        new_x_train, new_y_train, serial_numbers = RandomSampling(x_train, y_train)
        model = tree.DecisionTreeRegressor()
        model.fit(new_x_train, new_y_train)
        bagging_regressors.append(model)
        samples_serial.append(serial_numbers)
    return bagging_regressors, samples_serial # return the committee and the indexs of training samples used in the members

# Calculate y for the next training stage
def TrainStages(bagging_regressors, samples_serial, x_train):
    y_train = []
    for i in range(x_train.shape[0]):
        yk = 0
        count = 0
        for j in range(len(bagging_regressors)):
            if i not in samples_serial[j]:
            # if True:
                sample = x_train[i, :]
                sample = np.reshape(sample, (1, len(sample)))
                yk += bagging_regressors[j].predict(sample)
                count += 1
        if count != 0:
            yk = yk / count
        y_train.append(yk)
    y_train = np.reshape(y_train, (len(y_train), 1))
    return y_train # new y for bagging

# T: number of the members in committee, max_train_time: max training stage
def IteratedBagging(T, max_train_time, x_train, y_train):
    bagRegressors = []
    serials = []
    min_mss = float('inf')
    train_count = 0
    while True:
        clf, serial = Bagging(T, x_train, y_train)
        y_train = y_train - TrainStages(clf, serial, x_train)
        mss = np.mean(np.power(y_train, 2))
        if mss > (1.1 * min_mss) or mss < (10 ** -5) or train_count > max_train_time:
            # print('training stop!')
            break
        else:
            # print('next stage!')
            bagRegressors.append(clf)
            serials.append(serial)
            train_count += 1
            if mss < min_mss:
                min_mss = mss
    # print("Number of stages: %d\n" % len(bagRegressors))
    return bagRegressors, serials # return all committees and the indexs of training samples used in each each member

# used all committees (bagRegressors) to predict x
def Predict(bagRegressors, x):
    numStage = len(bagRegressors)
    y_pred = []
    for instant in range(x.shape[0]):
        y = 0
        sample = x[instant, :]
        sample = np.reshape(sample, (1, len(sample)))
        for i in range(numStage):
            yk = 0
            bagging_regressors = bagRegressors[i]
            for j in range(len(bagging_regressors)):
                yk += bagging_regressors[j].predict(sample)
            yk = yk / len(bagging_regressors)
            y = y + yk
        y_pred.append(y)
    y_pred = np.reshape(y_pred, (1, len(y_pred)))
    return y_pred

# Iterated Bagging Regressor Model
class IteratedBaggingRegressor(object):
    def __init__(self, bagNum, max_train): # defined the number of each committee(bagNum) and the max training stage (max_train)
        self.regressors = []
        self.serials = []
        self.beta = []
        self.bagNum = bagNum
        self.max_train = max_train

    def fit(self, x_train, y_train): # train the model
        self.regressors, self.serials = IteratedBagging(self.bagNum, self.max_train, x_train, y_train)

    def predict(self, x_test): # use the model to predict new instances
        pred = Predict(self.regressors, x_test)
        return pred

    def stage(self): # return the number of stages in the model
        return len(self.regressors)

