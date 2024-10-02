# Necessary imports
import pickle

import numpy as np
import pandas as pd
import xgboost as xg
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

# Load the data
dataset = pd.read_csv("../data//indicators.csv")
X, y = dataset.iloc[:, 2:17], dataset.iloc[:, 17]

# Splittingt
train_X, test_X, train_y, test_y = train_test_split(X, y,
                                                    test_size=0.1, random_state=123)

# Train and test set are converted to DMatrix objects,
# as it is required by learning API.
train_dmatrix = xg.DMatrix(data=train_X, label=train_y)
test_dmatrix = xg.DMatrix(data=test_X, label=test_y)

# Parameter dictionary specifying base learner
param = {"booster": "gblinear", "objective": "reg:linear"}

xgb_r = xg.train(params=param, dtrain=train_dmatrix, num_boost_round=100000)

print("Saving model..")
filename = "..//models//" + "XGboostReg.sav"
with open(filename, 'wb') as f:
    pickle.dump(xgb_r, f)
print("done")

pred = xgb_r.predict(test_dmatrix)

# RMSE Com
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# putation
rmse = np.sqrt(MSE(test_y, pred))
print("RMSE : % f" % (rmse))



x1 = np.arange(0, pred.size, 1)
plt.title("XG Boost")
plt.plot(x1, test_y, label="Actual")
plt.plot(x1, pred, label="Prediction")
plt.xlabel('x - axis')
plt.ylabel('y - axis')
plt.legend()
plt.show()
