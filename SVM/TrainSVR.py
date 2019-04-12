# 6.8

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


dataSet = pd.read_csv('data/watermelon_alpha.csv', header=0)
density = dataSet[dataSet.columns[1]]
density = density.values
density = np.reshape(density, (len(density), 1))
sugar = dataSet[dataSet.columns[2]]

X_train, X_test, y_train, y_test = train_test_split(density, sugar, test_size=0.2, random_state=0)
C = {100, 1000, 10000}
KERNEL = {'linear', 'rbf'}

fig_num = 0
# 对不同的参数组合进行SVR训练与测试
for c in C:
    for kernel in KERNEL:
        model = svm.SVR(C=c, kernel=kernel)

        model.fit(X_train, y_train)
        sv = model.support_vectors_
        score_tr = model.score(X_train, y_train)
        r2_tr = mean_squared_error(y_train, model.predict(X_train))
        score_test = model.score(X_test, y_test)
        r2_test = mean_squared_error(y_test, model.predict(X_test))
        num_sv = sv.shape[0]
        print("C = %d, kernel = %s, num of sv = %d, train score = %.4f, R2 loss of trainset = %.4f, test score = %.4f, R2 loss of testset = %.4f" % (c, kernel, num_sv, score_tr, r2_tr, score_test, r2_test))
