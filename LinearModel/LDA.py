from scipy import linalg
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors

dataSet = pd.read_csv('data/watermelon_alpha.csv', header=0)
data = dataSet[dataSet.columns[1:-1]]
label = dataSet[dataSet.columns[-1]]
X = data.values
y = label.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
lda.fit(X_train, y_train)

acc_tr = accuracy_score(y_train, lda.predict(X_train))
acc_test = accuracy_score(y_test, lda.predict(X_test))
print("accuracy in train: %.4f, accuracy in test: %.4f" % (acc_tr, acc_test))


fig = plt.figure()
plt.clf()
ax = fig.add_subplot(111, aspect='equal')
# plot trainset
y_pred = lda.predict(X_train)
tp = (y_train == y_pred)
tp0, tp1 = tp[y_train == 0], tp[y_train == 1]
X0, X1 = X_train[y_train == 0], X_train[y_train == 1]
X0_tp, X0_fp = X0[tp0], X0[~tp0]
X1_tp, X1_fp = X1[tp1], X1[~tp1]

# class 0: dots
plt.scatter(X0_tp[:, 0], X0_tp[:, 1], marker='.', color='b', s=80, edgecolor='k')
plt.scatter(X0_fp[:, 0], X0_fp[:, 1], marker='X', s=80, color='b', edgecolor='k')
# class 1: dots4
plt.scatter(X1_tp[:, 0], X1_tp[:, 1], marker='.', color='r', s=80, edgecolor='k')
plt.scatter(X1_fp[:, 0], X1_fp[:, 1], marker='X', s=80, color='r', edgecolor='k')

# plot testset
y_pred = lda.predict(X_test)
tp = (y_test == y_pred)
tp0, tp1 = tp[y_test == 0], tp[y_test == 1]
X0, X1 = X_test[y_test == 0], X_test[y_test == 1]
X0_tp, X0_fp = X0[tp0], X0[~tp0]
X1_tp, X1_fp = X1[tp1], X1[~tp1]

# class 0: dots
plt.scatter(X0_tp[:, 0], X0_tp[:, 1], marker='.', color='b', s=180, edgecolor='k')
plt.scatter(X0_fp[:, 0], X0_fp[:, 1], marker='X', s=180, color='b', edgecolor='k')
# class 1: dots
plt.scatter(X1_tp[:, 0], X1_tp[:, 1], marker='.', color='r', s=180, edgecolor='k')
plt.scatter(X1_fp[:, 0], X1_fp[:, 1], marker='X', s=180, color='r', edgecolor='k')

# class 0 and 1 : areas
nx, ny = 200, 100
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z = Z[:, 1].reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap='bwr', zorder=0)
plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='k')
# means
plt.plot(lda.means_[0][0], lda.means_[0][1], '*', color='y', markersize=15, markeredgecolor='k')
plt.plot(lda.means_[1][0], lda.means_[1][1], '*', color='y', markersize=15, markeredgecolor='k')

# plot ellipse
v, w = linalg.eigh(lda.covariance_)
u = w[0] / linalg.norm(w[0])
angle = np.arctan(u[1] / u[0])
angle = 180 * angle / np.pi  # convert to degrees
# filled Gaussian at 2 standard deviation
ell = mpl.patches.Ellipse(lda.means_[0], 2 * v[0] ** 0.5, 2 * v[1] ** 0.5, 180 + angle, facecolor='red',edgecolor='black', linewidth=2)
# ell.set_clip_box(plt.bbox)
ell.set_alpha(0.2)
ax.add_artist(ell)
ax.set_xticks(())
ax.set_yticks(())

# filled Gaussian at 2 standard deviation
ell = mpl.patches.Ellipse(lda.means_[1], 2 * v[0] ** 0.5, 2 * v[1] ** 0.5, 180 + angle, facecolor='blue',edgecolor='black', linewidth=2)
# ell.set_clip_box(plt.bbox)
ell.set_alpha(0.2)
ax.add_artist(ell)
ax.set_xticks(())
ax.set_yticks(())

plt.show()