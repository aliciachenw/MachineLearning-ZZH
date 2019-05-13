from sklearn.linear_model import LogisticRegression
import pandas as pd
import csv
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import time

dataSet = pd.read_csv('data\iris.csv', header=None)
dataSet = pd.read_csv('data\data_banknote_authentication.csv', header=None)
X = dataSet[dataSet.columns[0:-1]]
X = X.values
y = dataSet[dataSet.columns[-1]]
y = y.values

testTime = 10

skf = StratifiedKFold(n_splits=testTime)

test_num = 0
csvfile = open("results_10fold.csv", "w", newline='')
writer = csv.writer(csvfile)
writer.writerow(["test num", "train acc", "test acc", "time"])

ave_err_tr = 0.0
ave_err_test = 0.0
ave_t = 0.0

for train, test in skf.split(X, y):
    X_train = X[train, :]
    X_test = X[test, :]
    y_train = y[train]
    y_test = y[test]
    test_num += 1

    # standard SVM
    t = time.clock()
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    tr_t = time.clock() - t
    err_tr = 1 - accuracy_score(y_train, clf.predict(X_train))
    err_test = 1 - accuracy_score(y_test, clf.predict(X_test))
    print("error in train: %.4f, error in test: %.4f, training time: %.4f" % (err_tr, err_test, tr_t))
    writer.writerow([test_num, err_tr, err_test, tr_t])
    ave_err_tr += err_tr
    ave_err_test += err_test
    ave_t += tr_t
csvfile.close()
ave_err_tr = ave_err_tr / 10
ave_err_test = ave_err_test / 10
ave_t = ave_t / 10
print("Average error in train: %.4f, Average error in test: %.4f, Average training time: %.4f" % (ave_err_tr, ave_err_test, ave_t))