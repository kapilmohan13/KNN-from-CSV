import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pickle

# Generate some non-linear data (sine curve)
# X = np.sort(5 * np.random.rand(40, 1), axis=0)
# y = np.sin(X).ravel()
# y[::5] += 3 * (0.5 - np.random.rand(8))

df = pd.read_csv("data//ESG2.csv")
X = df.iloc[:, :-1].values
Y = df.iloc[:, 10].values

# svr = SVR(kernel='rbf')
# svr.fit(X, Y)

# regr = make_pipeline(StandardScaler(),  SVR(C=0.1, kernel='rbf', epsilon=2, max_iter=50000), verbose=True)
#[26.38105007]
# regr = make_pipeline(StandardScaler(),  SVR(C=5, kernel='rbf', epsilon=2, max_iter=50000), verbose=True)
# [28.14127404]

# regr = make_pipeline(StandardScaler(),  SVR(C=7, kernel='rbf', epsilon=2, max_iter=50000), verbose=True)
# [28.55465803] this took 40 minutes or so

# regr = make_pipeline(StandardScaler(),  SVR(C=10, kernel='rbf', epsilon=2, max_iter=50000), verbose=True)
# [29.23260586] took 48 minutes

#regr = make_pipeline(StandardScaler(),  SVR(C=10, kernel='rbf', epsilon=2, max_iter=100000), verbose=True)
#[28.59101568]

regr = make_pipeline(StandardScaler(),  SVR(C=10, kernel='rbf', epsilon=20, max_iter=100000), verbose=True)
regr.fit(X, Y)

check = [[4.04, 7.18, 0.031400003, 0.70972, 0.13052, 1.5734333, 3.67, 1.498, 1, 1]]
check2 = [[14.74,7.88,0.01503,1.03713,0.23464,0.61634886,2.25,1.964,1,1]]
X_test = np.arange(0, 5, 0.01)[:, np.newaxis]

y_pred = regr.predict(check2)
print(y_pred)
#
filename = "models//" + "SVR_hyp2.sav"
with open(filename, 'wb') as f:
    pickle.dump(regr, f)

# plt.scatter(X, Y, color='blue', label='Original data')
# plt.plot(X_test, y_pred, color='red', label='SVR predictions')
# plt.xlabel('X')
# plt.ylabel('y')
# plt.title('Support Vector Regression (SVR) with RBF Kernel')
# plt.legend()
# plt.show()
