import pickle
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

def main():
    # ypred = [45, 56, 23, 67, 90]
    # test_y = [40, 55, 29, 77, 70]
    #
    #
    # x1 = np.arange(0, len(ypred), 1)
    # plt.title("Gradient Boost SGB")
    # plt.plot(x1, test_y, label="Actual")
    # plt.plot(x1, ypred, label="Prediction")
    # plt.xlabel('x - axis')
    # plt.ylabel('y - axis')
    # plt.legend()
    # plt.show()


    start = time.time()
    # plot()
    # #Run prediction only
    # print("Running Gradient Boost prediction..", end=" ")
    # predictorGradientBoost()
    # print("done.")
    # exit()

    #Run model
    print("Running Gradient Boost model..", end=" ")
    runGradientBoost()
    print("done.")

    end = time.time()
    print("Total time to run : ", end=" ")
    print(end - start)


def predictorGradientBoost():
    filename = "models//" + "GradientBoost.sav"

    pickled_model = pickle.load(open(filename, 'rb'))
    check = [[0.383665561, 12.58, 3.73, 0.38, 3.19, 3.01, 1.49, 8.69, 26.69]]
    # npd = np.array([0.383665561, 12.58, 3.73, 0.38, 3.19, 3.01, 1.49, 8.69, 26.69])
    y_pred = pickled_model.predict(check)
    # 29.88432455
    print(y_pred)
def runGradientBoost():
    df = pd.read_csv('data//ESG_large_set.csv')

    X = df.iloc[:, 2:11].values
    Y = df.iloc[:, 11].values

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Splitting
    train_X, test_X, train_y, test_y = train_test_split(X, Y,
                                                        test_size=0.01, random_state=123)
    model = SGDRegressor(alpha=0.001, penalty='l1')
    model.fit(train_X, train_y)

    # Write model
    print("writing model...", end=" ")
    filename = "models//" + "GradientBoost.sav"
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print("done.")

    print("save x-test...", end=" ")
    filename = "temp//" + "GradientBoost_X_test.sav"
    with open(filename, 'wb') as f:
        pickle.dump(test_X, f)
    print("done.")

    print("save y-test...", end=" ")
    filename = "temp//" + "GradientBoost_Y_test.sav"
    with open(filename, 'wb') as f:
        pickle.dump(test_y, f)
    print("done.")

    ypred = model.predict(test_X)
    r2 = model.score(test_X, test_y)
    r2_sc = r2_score(test_y, ypred)
    print("r2=", end=" ")
    print(r2)
    print("r2_sc=", end=" ")
    print(r2_sc)
    # print(test_y)
    # print(ypred)
    # print("\n\nStochastic Gradient Descent Classifier Accuracy Score:", Base.accuracy_score(test_y, ypred), "%")
    #Start plotting:
    x1 = np.arange(0, ypred.size, 1)
    plt.title("Gradient Boost SGB")
    plt.plot(x1, test_y, label="Actual")
    plt.plot(x1, ypred, label="Prediction")
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.legend()
    plt.show()


def plot():

    x1 = np.arange(0, 4, 1)
    y1 = [2, 4, 1, 5]
    y2 = [4, 1, 3, 6]

    plt.plot(x1, y1, label="line 1")
    plt.plot(x1, y2, label="line 2")
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.legend()
    plt.show()


    # line 1 points
    # x1 = [1, 2, 3]
    # y1 = [2, 4, 1]
    # # plotting the line 1 points
    # plt.plot(x1, y1, label="line 1")
    #
    # # line 2 points
    # x2 = [1, 2, 3]
    # y2 = [4, 1, 3]
    # # plotting the line 2 points
    # plt.plot(x2, y2, label="line 2")
    #
    # # naming the x axis
    # plt.xlabel('x - axis')
    # # naming the y axis
    # plt.ylabel('y - axis')
    # # giving a title to my graph
    # plt.title('Two lines on same graph!')
    #
    # # show a legend on the plot
    # plt.legend()
    #
    # # function to show the plot
    # plt.show()


if __name__ == "__main__":
    main()