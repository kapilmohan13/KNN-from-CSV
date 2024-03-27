# Importing libraries
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import Lasso


def main():
    start = time.time()

    #Run prediction only
    print("Running Lasso prediction..")
    precictorLasso()

    # # Run model
    # runLasso()

    end = time.time()
    print("Total time to run model training: ", end=" ")
    print(end - start)

def precictorLasso():
    filename = "models//" + "Lasso_model.sav"

    pickled_model = pickle.load(open(filename, 'rb'))
    # check = [[0.383665561, 12.58, 3.73, 0.38, 3.19, 3.01, 1.49, 8.69, 26.69]]
    npd = np.array([0.383665561, 12.58, 3.73, 0.38, 3.19, 3.01, 1.49, 8.69, 26.69])
    y_pred = pickled_model.predict(npd)
    # 29.88432455
    print(y_pred)


def runLasso():
    # Importing dataset
    df = pd.read_csv("data//ESG_large_set.csv")
    X = df.iloc[:, 2:11].values
    Y = df.iloc[:, 11].values

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Splitting dataset into train and test set
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.01, train_size=0.9, random_state=0)

    # Model training
    print("Fitting model...", end=" ")
    model = Lasso.LassoRegression(
        iterations=100000, learning_rate=0.01, l1_penalty=50)
    model.fit(X_train, Y_train)
    print("done.")

    # Write model
    print("writing model...", end=" ")
    filename = "models//" + "Lasso_model.sav"
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print("done.")

    # Prediction on test set
    Y_pred = model.predict(X_test)

    print("Predicted values: ", np.round(Y_pred[:3], 2))
    print("Real values:	 ", Y_test[:3])
    print("Trained W:	 ", round(model.W[0], 2))
    print("Trained b:	 ", round(model.b, 2))

    # Test a prediction
    check = [[4.04, 7.18, 0.031400003, 0.70972, 0.13052, 1.5734333, 3.67, 1.498, 1, 1]]  # 29.24
    check2 = [[14.74, 7.88, 0.01503, 1.03713, 0.23464, 0.61634886, 2.25, 1.964, 1, 1]]  # 18.03
    npd = np.array([14.74, 7.88, 0.01503, 1.03713, 0.23464, 0.61634886, 2.25, 1.964, 1, 1])
    y_new_pred = model.predict(npd)
    print(y_new_pred)


    # Visualization on test set
    # plt.scatter(X_test, Y_test, color='blue', label='Actual Data')
    # plt.plot(X_test, Y_pred, color='orange', label='Lasso Regression Line')
    # plt.title('Salary vs Experience (Lasso Regression)')
    # plt.xlabel('Years of Experience (Standardized)')
    # plt.ylabel('Salary')
    # plt.legend()
    # plt.show()



if __name__ == "__main__":
    main()
