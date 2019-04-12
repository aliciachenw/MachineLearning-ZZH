# 6.10 reduce SVs from downsizing the dataset using KMeans

from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import csv
import time


def CountLabels(labels):
    labelCount = {}
    for label in labels:
        if label in labelCount:
            labelCount[label] += 1
        else:
            labelCount[label] = 1
    return labelCount


dataSet = pd.read_csv('data\Skin_NonSkin.csv', header=None)
# dataSet = pd.read_csv('data\pendigits.csv', header=None)
# dataSet = pd.read_csv('data\data_banknote_authentication.csv', header=None)
X = dataSet[dataSet.columns[0:-1]]
X = X.values
y = dataSet[dataSet.columns[-1]]
y = y.values
testTime = 10  # 交叉验证次数

# 对Skin_NonSkin取十分之一，其他数据集的时候注释掉
skf = StratifiedKFold(n_splits=10)
for _, choose in skf.split(X, y):
    X = X[choose, :]
    y = y[choose]
    break

skf = StratifiedKFold(n_splits=testTime)

# svm parameter
c = 100
kernel = 'rbf'


all_rate = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
# all_rate = [0.01, 0.05, 0.1, 0.5]
test_num = 0
csvfile = open("results.csv", "w")
writer = csv.writer(csvfile)
writer.writerow(["test num", "C", "kernel", "K rate", "sv num", "train acc", "test acc", "time"])

# 进行交叉验证
for train, test in skf.split(X, y):
    X_train = X[train, :]
    X_test = X[test, :]
    y_train = y[train]
    y_test = y[test]
    test_num += 1
    # standard SVM
    t = time.clock()
    model = svm.SVC(C=c, kernel=kernel)
    model.fit(X_train, y_train)
    tr_t = time.clock() - t
    sv = model.support_vectors_
    acc_tr = accuracy_score(y_train, model.predict(X_train))
    acc = accuracy_score(y_test, model.predict(X_test))
    num_sv = sv.shape[0]
    print("Standard SVM:C = %d, kernel = %s" % (c, kernel))
    print("num of sv = %d, train accuracy = %.4f, test accuracy = %.4f, train time = %.4f" % (
    num_sv, acc_tr, acc, tr_t))
    writer.writerow([test_num, c, kernel, 1, num_sv, acc_tr, acc, tr_t])
	
	# KMeans+SVM
    for rate in all_rate:
        t = time.clock()
		# 对每一类都进行KMeans聚类，取聚类中心作为新的训练样本
        classes = np.unique(y_train)
        classes = classes.tolist()
        n_cluster = round(X_train.shape[0] * rate) # 计算按比例rate要生成的总聚类数量
        flag = 0
        for lab in classes:
            index = [i for i,v in enumerate(y_train) if v==lab] 
            X_tr_sub = X_train[index, :] # 取训练样本中同属一类的样本
            n_sub = n_cluster * X_tr_sub.shape[0] / X_train.shape[0] # 计算按比例rate生成的某一类的样本的数量
            n_sub = round(n_sub)
            print("Class: %d, # of samples: %d, # of chosen samples: %d" % (lab, X_tr_sub.shape[0], n_sub))
            kmeans = KMeans(n_clusters=n_sub).fit(X_tr_sub)
            print("Finish KMeans!")
            center = kmeans.cluster_centers_ # 得到新的训练样本
            if flag == 0:
                X_train_re = center
                y_train_re = [lab] * n_sub
                y_train_re = np.array(y_train_re)
                y_train_re = np.reshape(y_train_re, (n_sub, 1))
                flag = 1
            else:
                X_train_re = np.vstack((X_train_re, center))
                y_temp = [lab] * n_sub
                y_temp = np.array(y_temp)
                y_temp = np.reshape(y_temp, (n_sub, 1))
                y_train_re = np.vstack((y_train_re, y_temp))
		# 对新的训练集进行训练
        model_re = svm.SVC(C=c, kernel=kernel)
        model_re.fit(X_train_re, y_train_re)
        tr_t = time.clock() - t
        sv = model_re.support_vectors_
        acc_tr = accuracy_score(y_train_re, model_re.predict(X_train_re))
        acc = accuracy_score(y_test, model_re.predict(X_test))
        num_sv = sv.shape[0]
        print("Reduced SVM:C = %d, kernel = %s, sample rate = %.4f" % (c, kernel, rate))
        print("num of sv = %d, train accuracy = %.4f, test accuracy = %.4f, train time = %.4f" % (
        num_sv, acc_tr, acc, tr_t))
        writer.writerow([test_num, c, kernel, rate, num_sv, acc_tr, acc, tr_t])

csvfile.close()