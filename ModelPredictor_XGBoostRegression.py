import pickle

import numpy as np
import pandas as pd
import xgboost as xg

# filename = "models//" + "Lasso_model.sav"
filename = "models//" + "XGboostRegression.sav"

pickled_model = pickle.load(open(filename, 'rb'))
check = [[0.383665561,12.58,3.73,0.38,3.19,3.01,1.49,8.69,26.69]]
#12

dataset = pd.read_csv("data//ESG_large_set.csv")
X = dataset.iloc[0:1, 2:11]
test_dmatrix = xg.DMatrix(data=X)


# check = [[11.06,12.28,0.00054000004,0.73857003,0.36269,13.698857,0.97,0.89,1,5]]  #15.06

# npd = np.array([11.06,12.28,0.00054000004,0.73857003,0.36269,13.698857,0.97,0.89,1,5])
# y_pred = pickled_model.predict(npd)
#23.354195902231304

y_pred = pickled_model.predict(test_dmatrix)
#29.88432455

print(y_pred)