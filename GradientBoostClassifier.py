import pickle
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.naive_bayes import GaussianNB


def main():
    start = time.time()

    # #Run prediction only
    # print("Running Gradient Boost classification prediction..", end=" ")
    # predictorGradientBoostClassifier()
    # print("done.")

    #Run model
    print("Running Gradient Boost Classification model..", end=" ")
    runGradientBoostClassifier()
    print("done.")

    end = time.time()
    print("Total time to run : ", end=" ")
    print(end - start)

def predictorGradientBoostClassifier():
    filename = "models//" + "GradientBoostClassfication_model.sav"

    pickled_model = pickle.load(open(filename, 'rb'))
    check = [[0.383665561, 12.58, 3.73, 0.38, 3.19, 3.01, 1.49, 8.69, 26.69]]
    # npd = np.array([0.383665561, 12.58, 3.73, 0.38, 3.19, 3.01, 1.49, 8.69, 26.69])
    y_pred = pickled_model.predict(check)
    # 29.88432455
    print(y_pred)

def runGradientBoostClassifier():
    df = pd.read_csv('data//ESG_large_set.csv')

    X = df.iloc[:, 2:11].values
    Y = df.iloc[:, 11].values
    output_array = []
    for element in Y:
        converted_str = str(element)
        output_array.append(converted_str)
    Y = output_array

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=152)

    from sklearn.ensemble import GradientBoostingClassifier
    # Create the model
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)

    # Train the model
    model.fit(X_train, y_train)
    # r2 = regressor.score(X_test, y_test)
    # print("r2=", end=" ")
    # print(r2)

    # Write model
    print("writing model...", end=" ")
    filename = "models//" + "GradientBoostClassification_model.sav"
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print("done.")

    y_pred = model.predict(X_test)

    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        ConfusionMatrixDisplay,
        f1_score,
    )

    y_pred = model.predict(X_test)
    accuray = accuracy_score(y_pred, y_test)
    f1 = f1_score(y_pred, y_test, average="weighted")

    print("Accuracy:", accuray)
    print("F1 Score:", f1)


    labels = list(set(output_array))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    print("Plotting..")

    disp.plot(xticks_rotation="vertical")
    plt.show()

    from sklearn.metrics import accuracy_score
    # accuracy = accuracy_score(y_test, ypred)
    # precision = precision_score(y_test, ypred, average=None)
    # recall = recall_score(y_test, ypred, average=None)
    #
    # print("Accuracy:", accuracy)
    # print("Precision:", precision)
    # print("Recall:", recall)



if __name__ == "__main__":
    main()