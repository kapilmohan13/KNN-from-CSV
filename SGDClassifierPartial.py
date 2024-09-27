import pickle
import time

import numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def main():
    start = time.time()

    # #Run prediction only
    # print("Running Random Forest prediction..", end=" ")
    # predictorRandomForest()
    # print("done.")

    # # Run model
    # print("Running SVM Classifier model..", end=" ")
    # runSGDClassifier()
    # print("done.")

    # Run update model
    print("Running SVM Classifier model update..", end=" ")
    updateSGDClassifier()
    print("done.")

    # # Run model cross validation
    # print("Running SVM classification model and cross validation..", end=" ")
    # runSGDClassifierCV()
    # print("done.")

    end = time.time()
    print("Total time to run : ", end=" ")
    print(end - start)

def updateSGDClassifier():
    filename = "models//" + "SGDClassifierPartial_model.sav"
    pickled_model = pickle.load(open(filename, 'rb'))
    df = pd.read_csv('data//ESG_large_set.csv')
    Y = df.iloc[:, 11].values
    output_array = []
    for element in Y:
        converted_str = str(element)
        output_array.append(converted_str)

    Y = output_array
    # classes = numpy.unique(Y)

    df_partial = pd.read_csv('data//ESG_small_set.csv')
    X_partial = df_partial.iloc[:, 2:11].values
    Y_partial = df_partial.iloc[:, 11].values
    output_array = []
    for element in Y_partial:
        converted_str = str(element)
        output_array.append(converted_str)

    Y_partial = output_array
    scaler = StandardScaler()
    X_partial = scaler.fit_transform(X_partial)
    pickled_model.partial_fit(X_partial, Y_partial, classes=numpy.unique(Y))

    # Write model
    print("writing model...", end=" ")
    filename = "models//" + "SGDClassifierPartial_model.sav"
    with open(filename, 'wb') as f:
        pickle.dump(pickled_model, f)
    print("done.")

    ypred = pickled_model.predict(X_partial)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(Y_partial, ypred)
    precision = precision_score(Y_partial, ypred, average=None)
    recall = recall_score(Y_partial, ypred, average=None)
    f1 = f1_score(ypred, Y_partial, average="weighted")


    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("f1_score:", f1)



    labels = list(set(output_array))
    cm = confusion_matrix(Y_partial, ypred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    print("Plotting..")

    disp.plot(xticks_rotation="vertical")
    plt.show()






def predictorSGDClassifier():
    filename = "models//" + "SGDClassifierPartial_model.sav"


    pickled_model = pickle.load(open(filename, 'rb'))
    check = [[0.383665561, 12.58, 3.73, 0.38, 3.19, 3.01, 1.49, 8.69, 26.69]]
    # npd = np.array([0.383665561, 12.58, 3.73, 0.38, 3.19, 3.01, 1.49, 8.69, 26.69])
    y_pred = pickled_model.predict(check)
    # 29.88432455
    print(y_pred)


def runSGDClassifier():
    df = pd.read_csv('data//ESG_large_set.csv')

    X = df.iloc[:, 2:11].values
    Y = df.iloc[:, 11].values
    output_array = []
    for element in Y:
        converted_str = str(element)
        output_array.append(converted_str)

    Y = output_array

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.01, random_state=421)

    from sklearn.ensemble import RandomForestRegressor

    # Create the model
    # model = make_pipeline(StandardScaler(), SGDClassifier(max_iter=5000, loss="squared_hinge", n_jobs=4, verbose=0, alpha=0.0001, tol=1e-3))
    #Accuracy: 0.770293609671848

    pipeline = make_pipeline(StandardScaler(), SGDClassifier(warm_start=True, max_iter=5000, loss="log_loss", n_jobs=4, verbose=0, alpha=0.0001, tol=1e-3))

    # Train the model
    startmodel = time.time()
    print("Starting model training[initial partial fit]...", end=" ")
    # model = SGDClassifier(warm_start=True, max_iter=5000, loss="log_loss", n_jobs=4, verbose=0, alpha=0.0001, tol=1e-3)
    # scaler = StandardScaler()
    # x_std = scaler.fit_transform(X_train)
    # y_std = scaler.fit_transform(y_train)
    pipeline.fit(X_train, y_train)
    model = pipeline[1]
    startmodel = time.time()
    print("DONE")
    endmodeltime = time.time()
    print("Model training time:", end=" ")
    print(endmodeltime - startmodel)

    # r2 = regressor.score(X_test, y_test)
    # print("r2=", end=" ")
    # print(r2)

    # Write model
    print("writing model...", end=" ")
    filename = "models//" + "SGDClassifierPartial_model.sav"
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print("done.")

    ypred = pipeline.predict(X_test)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, ypred)
    precision = precision_score(y_test, ypred, average=None)
    recall = recall_score(y_test, ypred, average=None)
    f1 = f1_score(ypred, y_test, average="weighted")


    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("f1_score:", f1)



    labels = list(set(output_array))
    cm = confusion_matrix(y_test, ypred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    print("Plotting..")

    disp.plot(xticks_rotation="vertical")
    plt.show()


    # x1 = np.arange(0, ypred.size, 1)
    # plt.plot(x1, y_test, label="Actual")
    # plt.plot(x1, ypred, label="Prediction")
    # plt.xlabel('x - axis')
    # plt.ylabel('y - axis')
    # plt.legend()
    # plt.show()

    from sklearn.metrics import accuracy_score, classification_report

    # print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
    # print(classification_report(y_test, y_pred))


def runSGDClassifierCV():
    df = pd.read_csv('data//ESG_large_set.csv')

    X = df.iloc[:, 2:11].values
    Y = df.iloc[:, 11].values
    output_array = []
    for element in Y:
        converted_str = str(element)
        output_array.append(converted_str)

    Y = output_array

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.01, random_state=421)

    from sklearn.ensemble import RandomForestRegressor

    # Create the model
    model = RandomForestClassifier(max_depth=17, random_state=123)
    n_estimators = [10, 100, 1000]
    max_features = ['sqrt', 'log2']
    # define grid search
    grid = dict(n_estimators=n_estimators, max_features=max_features)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
    grid_result = grid_search.fit(X_train, y_train)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    # Write model
    print("writing model...", end=" ")
    filename = "models//" + "RandomForestClassfier_model.sav"
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print("done.")

    ypred = model.predict(X_test)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, ypred)
    precision = precision_score(y_test, ypred, average=None)
    recall = recall_score(y_test, ypred, average=None)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)

    # x1 = np.arange(0, ypred.size, 1)
    # plt.plot(x1, y_test, label="Actual")
    # plt.plot(x1, ypred, label="Prediction")
    # plt.xlabel('x - axis')
    # plt.ylabel('y - axis')
    # plt.legend()
    # plt.show()

    from sklearn.metrics import accuracy_score, classification_report

    # print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
    # print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
