from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time

dataSet = pd.read_csv('data/watermelon_alpha.csv', header=0)
data = dataSet[dataSet.columns[1:-1]]
label = dataSet[dataSet.columns[-1]]

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.25, random_state=0)

t = time.clock()
clf = LogisticRegression()
clf.fit(X_train, y_train)
tr_t = time.clock() - t
acc_tr = accuracy_score(y_train, clf.predict(X_train))
acc_test = accuracy_score(y_test, clf.predict(X_test))
print("accuracy in train: %.4f, accuracy in test: %.4f" % (acc_tr, acc_test))
print("time for training: %.4f s" % tr_t)