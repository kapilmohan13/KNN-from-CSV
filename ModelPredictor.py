
import pickle

import numpy as np

filename = "models//" + "Lasso_model.sav"
# filename = "models//" + "SVR_hyp2.sav"

pickled_model = pickle.load(open(filename, 'rb'))
# check = [[3.98,  6.67,  0.31,  0.96,  7.79,  0.95]]
# check = [[11.06,12.28,0.00054000004,0.73857003,0.36269,13.698857,0.97,0.89,1,5]]  #15.06

npd = np.array([11.06,12.28,0.00054000004,0.73857003,0.36269,13.698857,0.97,0.89,1,5])
y_pred = pickled_model.predict(npd)
#23.354195902231304

# y_pred = pickled_model.predict(check)
#29.88432455

print(y_pred)