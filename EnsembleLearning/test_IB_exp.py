# Use Iterated Bagging in UCI data

from IteratedBagging import IteratedBaggingRegressor
from IteratedBagging import LoadData
from sklearn import tree
import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor

# Load data from CSV files
# x_train, x_test, y_train, y_test = LoadData('data/housing.csv', 0)
x_train, x_test, y_train, y_test = LoadData('data/Concrete_Data.csv', 0)

estimators = [("Tree", tree.DecisionTreeRegressor()),
              ("Bagging(Tree)", BaggingRegressor(tree.DecisionTreeRegressor())),
              ("AdaBoost(Tree)", AdaBoostRegressor(tree.DecisionTreeRegressor())),
              ("IteratedBagging(Tree)", IteratedBaggingRegressor(10, 5))]

n_repeat = 10 # Number of iterations for computing expectations
n_estimators = len(estimators)
n_test = x_test.shape[0]
X_train = []
Y_train = []

num_stages = 0 # calculate average stages of Iterated Bagging

# Generate different training set for n_repeat times
for i in range(n_repeat):
    x_train_sample, _, y_train_sample, _ = train_test_split(x_train, y_train, test_size=0.5)
    X_train.append(x_train_sample)
    Y_train.append(y_train_sample)

for n, (name, estimator) in enumerate(estimators):
    # Compute predictions
    y_predict = np.zeros((n_test, n_repeat))

    for i in range(n_repeat):
        # estimator.fit(X_train[i], y_train[i])
        x = X_train[i]
        y = Y_train[i].reshape((len(Y_train[i]), 1))
        if name == "Bagging(Tree)" or name == "AdaBoost(Tree)":
            y = y.ravel()
        estimator.fit(x, y)
        y_predict[:, i] = estimator.predict(x_test)
        if name == "IteratedBagging(Tree)":
            num_stages += estimator.stage()

    # Bias^2 + Variance decomposition of the mean squared error
    y_error = np.zeros((n_test, 1))

    for i in range(n_repeat):
        y_error += (y_test - np.reshape(y_predict[:, i], (n_test, 1))) ** 2

    y_error /= n_repeat

    y_bias = (y_test - np.reshape(np.mean(y_predict, axis=1), (n_test, 1))) ** 2
    y_var = np.var(y_predict, axis=1)

    print("{0}: {1:.4f} (error) = {2:.4f} (bias^2)"
          " + {3:.4f} (var)".format(name, np.mean(y_error), np.mean(y_bias), np.mean(y_var)))

ave = num_stages / n_repeat
print("average num of stages for IteratedBagging = %f" % ave)



