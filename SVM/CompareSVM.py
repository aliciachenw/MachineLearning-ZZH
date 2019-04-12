# 6.2

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

dataSet = pd.read_csv('data/watermelon_alpha.csv', header=0)
data = dataSet[dataSet.columns[1:-1]]
label = dataSet[dataSet.columns[-1]]

trainSet, testSet, trainLab, testLab = train_test_split(data, label, test_size=0.2, random_state=0)
# Test different C and kernels
C = {100, 1000, 10000}
KERNEL = {'linear', 'rbf'}

fig_num = 0
for c in C:
    for kernel in KERNEL:
        model = svm.SVC(C=c, kernel=kernel)

        model.fit(trainSet, trainLab)
        sv = model.support_vectors_
        acc_tr = accuracy_score(trainLab, model.predict(trainSet))
        acc = accuracy_score(testLab, model.predict(testSet))
        num_sv = sv.shape[0]
        print("C = %d, kernel = %s, num of sv = %d, train accuracy = %.4f, test accuracy = %.4f" % (c, kernel, num_sv, acc_tr, acc))
        plt.figure(fig_num)
        plt.clf()
        plt.scatter(trainSet[trainSet.columns[0]], trainSet[trainSet.columns[1]], c=trainLab, zorder=10, edgecolor='k', s=20, cmap=plt.cm.get_cmap('Paired'))
        plt.scatter(testSet[testSet.columns[0]], testSet[testSet.columns[1]], s=80, zorder=10, edgecolor='k', c=testLab, cmap=plt.cm.get_cmap('Paired'))
        x_min = data[data.columns[0]].min()
        x_max = data[data.columns[0]].max()
        y_min = data[data.columns[1]].min()
        y_max = data[data.columns[1]].max()

        XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
        Z = model.decision_function(np.c_[XX.ravel(), YY.ravel()])
        Z = Z.reshape(XX.shape)
        plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.get_cmap('Paired'))
        plt.contour(XX, YY, Z, colors=['k','k','k'], linestyles=['--','--','--'],  levels=[-.5, 0, .5])
        title = 'C='+str(c)+', kernel='+kernel
        plt.title(title)
        fig_num += 1
plt.show()
