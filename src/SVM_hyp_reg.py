# Necessary imports
import pickle

import numpy as np
import pandas as pd
import xgboost as xg
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

# Load the data
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

dataset = pd.read_csv("../data//indicators_NaN.csv")
X, y = dataset.iloc[:, 2:17], dataset.iloc[:, 17]

# Splitting
train_X, test_X, train_y, test_y = train_test_split(X, y,
                                                    test_size=0.1, random_state=123)
regr = make_pipeline(StandardScaler(),  SVR(C=8, kernel='rbf', epsilon=0.3, max_iter=100000), verbose=True)
print("Running model SVR..")
regr.fit(train_X, train_y)
print("done")
# Train and test set are converted to DMatrix objects,
# as it is required by learning API.

# train_dmatrix = xg.DMatrix(data=train_X, label=train_y)
# test_dmatrix = xg.DMatrix(data=test_X, label=test_y)

# Parameter dictionary specifying base learner
param = {"booster": "gblinear", "objective": "reg:linear"}

# xgb_r = xg.train(params=param, dtrain=train_dmatrix, num_boost_round=100000)

print("Saving model..")
filename = "..//models//" + "SVM_hup_reg.sav"
with open(filename, 'wb') as f:
    pickle.dump(regr, f)
print("done")

# pred = regr.predict(test_dmatrix)
y_pred = regr.predict(test_X)
print(y_pred)

# RMSE Computation
rmse = np.sqrt(MSE(test_y, y_pred))
print("RMSE : % f" % (rmse))



x1 = np.arange(0, y_pred.size, 1)
plt.title("XG Boost")
plt.plot(x1, test_y, label="Actual")
plt.plot(x1, y_pred, label="Prediction")
plt.xlabel('x - axis')
plt.ylabel('y - axis')
plt.legend()
plt.show()
