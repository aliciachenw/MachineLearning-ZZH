# Use Iterated Bagging in simulated data

from IteratedBagging import IteratedBaggingRegressor
from sklearn import tree
import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor

# Settings
n_repeat = 100 # Number of iterations for computing expectations
n_train = 50 # Size of the training set
n_test = 1000 # Size of the test set
noise = 0.1 # Standard deviation of the noise
np.random.seed(0)
func = 'f2' # Choose the simulated data, can use 'f1' or 'f2'

estimators = [("Tree", tree.DecisionTreeRegressor()),
              ("Bagging(Tree)", BaggingRegressor(tree.DecisionTreeRegressor())),
              ("AdaBoost(Tree)", AdaBoostRegressor(tree.DecisionTreeRegressor())),
              ("IteratedBagging(Tree)", IteratedBaggingRegressor(10, 5))] # The estimators to be compared
n_estimators = len(estimators)

# simulated data 1
def f1(x): 
    y = np.zeros((x.shape[0], 1))
    for i in range(x.shape[0]):
        y[i, 0] = 0.1 * np.exp(x[i, 0] * 4.0) + 4.0 / (1 + np.exp(-20.0 * (x[i, 1] - 0.5))) + 3.0 * x[i, 2] + 2.0 * x[i, 3] + x[i, 4]
    return y

# simulated data 2
def f2(x): 
    y = np.zeros((x.shape[0], 1))
    for i in range(x.shape[0]):
        y[i, 0] = 10 * np.sin(np.pi * x[i, 0] * x[i, 1]) + 20.0 * (x[i, 2] - 0.5) ** 2 + 10.0 * x[i, 3] + 5.0 * x[i, 4]
    return y

# Generated simulated data, choosing # of samples, noise, # of repeat and simulated method (f1 or f2)
def generate(n_samples, noise, n_repeat, function):
    y = np.zeros((n_samples, n_repeat))
    y_true = np.zeros((n_samples, 1))
    if function == 'f1':
        X = np.zeros((n_samples, 10))
        for i in range(n_samples):
            for j in range(10):
                X[i, j] = np.random.uniform()
        for i in range(n_repeat):
            ff = f1(X)
            y[:, i] = np.reshape(f1(X) + np.reshape(np.random.normal(0.0, noise, n_samples), (n_samples, 1)), n_samples)
            y_true = f1(X)
    else:
        X = np.zeros((n_samples, 6))
        for i in range(n_samples):
            for j in range(6):
                X[i, j] = np.random.uniform()
        for i in range(n_repeat):
            y[:, i] = np.reshape(f2(X) + np.reshape(np.random.normal(0.0, noise, n_samples), (n_samples, 1)), n_samples)
            y_true = f2(X)

    return X, y, y_true


X_train = []
y_train = []
num_stages = 0

# generate training set for n_repeat times
for i in range(n_repeat):
    X, y, y_true = generate(n_samples=n_train, noise=noise, n_repeat=1, function=func)
    X_train.append(X)
    y_train.append(y)

# generate testing set for n_repeat times
X_test, y_test, y_test_true = generate(n_samples=n_test, noise=noise, n_repeat=n_repeat, function=func)

for n, (name, estimator) in enumerate(estimators):
    # Compute predictions
    y_predict = np.zeros((n_test, n_repeat))

    for i in range(n_repeat):
        # estimator.fit(X_train[i], y_train[i])
        x = X_train[i]
        y = y_train[i].reshape((len(y_train[i]), 1))
        if name == "Bagging(Tree)" or name == "AdaBoost(Tree)":
            y = y.ravel()
        estimator.fit(x, y)
        y_predict[:, i] = estimator.predict(X_test)
        if name == "IteratedBagging(Tree)":
            num_stages += estimator.stage()

    # Bias^2 + Variance + Noise decomposition of the mean squared error
    y_error = np.zeros((n_test, 1))

    for i in range(n_repeat):
        y_error += (np.reshape(y_test[:, i], (n_test, 1)) - np.reshape(y_predict[:, i], (n_test, 1))) ** 2

    y_error /= n_repeat

    y_bias = (y_test_true - np.reshape(np.mean(y_predict, axis=1), (n_test, 1))) ** 2
    y_var = np.var(y_predict, axis=1)
    y_noise = np.var(y_test, axis=1)

    print("{0}: {1:.4f} (error) = {2:.4f} (bias^2)" 
          " + {3:.4f} (var) + {4:.4f} (noise)".format(name,
                                                      np.mean(y_error),
                                                      np.mean(y_bias),
                                                      np.mean(y_var),
                                                      np.mean(y_noise)))

ave = num_stages / n_repeat
print("average num of stages for IteratedBagging = %f" % ave)



