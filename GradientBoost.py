import pickle
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def main():
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
    filename = "models//" + "RandomForest_model.sav"

    pickled_model = pickle.load(open(filename, 'rb'))
    check = [[0.383665561, 12.58, 3.73, 0.38, 3.19, 3.01, 1.49, 8.69, 26.69]]
    # npd = np.array([0.383665561, 12.58, 3.73, 0.38, 3.19, 3.01, 1.49, 8.69, 26.69])
    y_pred = pickled_model.predict(check)
    # 29.88432455
    print(y_pred)
def runGradientBoost():
    df = pd.read_csv('data//ESG_large_set.csv')

    X = df.iloc[:, 2:11].values
    Y = df.iloc[:, 10].values

    # Splitting
    train_X, test_X, train_y, test_y = train_test_split(X, Y,
                                                        test_size=0.01, random_state=123)
    model = SGDRegressor()
    model.fit(train_X, train_y)

    # Write model
    print("writing model...", end=" ")
    filename = "models//" + "GradientBoost.sav"
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print("done.")
    ypred = model.predict(test_X)

    # print("\n\nStochastic Gradient Descent Classifier Accuracy Score:", Base.accuracy_score(test_y, ypred), "%")
    #Start plotting:
    x1 = np.arange(0, ypred.size, 1)
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