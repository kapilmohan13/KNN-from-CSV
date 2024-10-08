import pickle
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    start = time.time()

    # #Run prediction only
    # print("Running Random Forest prediction..", end=" ")
    # predictorRandomForest()
    # print("done.")

    #Run model
    print("Running Random Forest model..", end=" ")
    runRandomForest()
    print("done.")

    end = time.time()
    print("Total time to run : ", end=" ")
    print(end - start)

def predictorRandomForest():
    filename = "models//" + "RandomForest_model.sav"

    pickled_model = pickle.load(open(filename, 'rb'))
    check = [[0.383665561, 12.58, 3.73, 0.38, 3.19, 3.01, 1.49, 8.69, 26.69]]
    # npd = np.array([0.383665561, 12.58, 3.73, 0.38, 3.19, 3.01, 1.49, 8.69, 26.69])
    y_pred = pickled_model.predict(check)
    # 29.88432455
    print(y_pred)

def runRandomForest():
    df = pd.read_csv('data//ESG_large_set.csv')

    X = df.iloc[:, 2:11].values
    Y = df.iloc[:, 11].values

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.01, random_state=42)

    from sklearn.ensemble import RandomForestRegressor

    # Create the model
    regressor = RandomForestRegressor(n_estimators=1000, random_state=42)

    # Train the model
    regressor.fit(X_train, y_train)
    r2 = regressor.score(X_test, y_test)
    print("r2=", end=" ")
    print(r2)

    # Write model
    print("writing model...", end=" ")
    filename = "models//" + "RandomForest_model.sav"
    with open(filename, 'wb') as f:
        pickle.dump(regressor, f)
    print("done.")

    ypred = regressor.predict(X_test)

    x1 = np.arange(0, ypred.size, 1)
    plt.plot(x1, y_test, label="Actual")
    plt.plot(x1, ypred, label="Prediction")
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.legend()
    plt.show()

    from sklearn.metrics import accuracy_score, classification_report

    # print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
    # print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()