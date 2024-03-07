
import pickle

filename = "models//" + "SVR_hyp.sav"
pickled_model = pickle.load(open(filename, 'rb'))
check = [[3.98,  6.67,  0.31,  0.96,  7.79,  0.95]]
y_pred = pickled_model.predict(check)

print(y_pred)