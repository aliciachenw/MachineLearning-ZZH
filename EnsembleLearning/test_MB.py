# Use UCI data to test MultiBoosting

from MultiBoosting import MultiBoostingClassifier
from MultiBoosting import LoadData
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data from CSV file
# x_train, x_test, y_train, y_test = LoadData('data/data_banknote_authentication.csv', None)
# x_train, x_test, y_train, y_test = LoadData('data/Immunotherapy.csv', None)
x_train, x_test, y_train, y_test = LoadData('data/Immunotherapy.csv', None)

estimators = [("Tree", DecisionTreeClassifier()),
              ("Bagging(Tree)", BaggingClassifier(DecisionTreeClassifier())),
              ("AdaBoost(Tree)", AdaBoostClassifier(DecisionTreeClassifier())),
              ("MultiBoosting(Tree)", MultiBoostingClassifier(100))] # the estimators to be compared

n_estimators = len(estimators)
n_repeat = 10 # Number of iterations for computing expectations
n_test = x_test.shape[0]
X_train = []
Y_train = []

for i in range(n_repeat):
    x_train_sample, _, y_train_sample, _ = train_test_split(x_train, y_train, test_size=0.9, stratify=y_train)
    X_train.append(x_train_sample)
    Y_train.append(y_train_sample)

# Generate different training set for n_repeat times
for n, (name, estimator) in enumerate(estimators):
    # Compute predictions
    y_error = []
    for i in range(n_repeat):
        # estimator.fit(X_train[i], y_train[i])
        x = X_train[i]
        y = Y_train[i].reshape((len(Y_train[i]), 1))
        if name == "Bagging(Tree)" or name == "AdaBoost(Tree)":
            y = y.ravel()
        estimator.fit(x, y)
        y_predict = estimator.predict(x_test)

        y_error.append(1 - accuracy_score(y_test, y_predict))

    mean_error = np.mean(y_error)
    var_error = np.var(y_error)
    geo_error = 1.0
    for i in range(len(y_error)):
        geo_error *= y_error[i]
    geo_error = np.sqrt(geo_error) * n_repeat

    print("{0}: mean error = {1:.6f}, var error = {2:.6f}, geometric mean error = {3:.6f}" .format(name, mean_error, var_error, geo_error))



