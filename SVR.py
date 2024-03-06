import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import pandas as pd

# Generate some non-linear data (sine curve)
# X = np.sort(5 * np.random.rand(40, 1), axis=0)
# y = np.sin(X).ravel()
# y[::5] += 3 * (0.5 - np.random.rand(8))

df = pd.read_csv("data//ESG_reg.csv")
X = df.iloc[:, :-1].values
Y = df.iloc[:, 6].values

svr = SVR(kernel='rbf')
svr.fit(X, Y)

check = [[3.98,  6.67,  0.31,  0.96,  7.79,  0.95]]

X_test = np.arange(0, 5, 0.01)[:, np.newaxis]

y_pred = svr.predict(check)
print(y_pred)

# plt.scatter(X, Y, color='blue', label='Original data')
# plt.plot(X_test, y_pred, color='red', label='SVR predictions')
# plt.xlabel('X')
# plt.ylabel('y')
# plt.title('Support Vector Regression (SVR) with RBF Kernel')
# plt.legend()
# plt.show()
