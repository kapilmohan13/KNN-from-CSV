import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)


# Generate some non-linear data (sine curve)
# X = np.sort(5 * np.random.rand(40, 1), axis=0)
# y = np.sin(X).ravel()
# y[::5] += 3 * (0.5 - np.random.rand(8))

df = pd.read_csv("data//ESG_large_set.csv")
X = df.iloc[:, 2:11].values
Y = df.iloc[:, 11].values

output_array = []
for element in Y:
    converted_str = str(element)
    output_array.append(converted_str)

Y = output_array
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=49)
# Scale the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5000)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("Saving model..")
filename = "models//" + "MyKNN_model.sav"
with open(filename, 'wb') as f:
    pickle.dump(knn, f)
print("done")

# Collect metrics


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

# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred,  average=None)
# recall = recall_score(y_test, y_pred,  average=None)
#
# print("Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)

# regr = make_pipeline(StandardScaler(),  SVR(C=8, kernel='rbf', epsilon=0.3, max_iter=100000), verbose=True)
# print("Running model SVR..")
# regr.fit(X_train, y_train)
# print("done")
#
# check = [[4.04, 7.18, 0.031400003, 0.70972, 0.13052, 1.5734333, 3.67, 1.498, 1, 1]]
# check2 = [[14.74,7.88,0.01503,1.03713,0.23464,0.61634886,2.25,1.964,1,1]]
# X_test = np.arange(0, 5, 0.01)[:, np.newaxis]
#
# y_pred = regr.predict(X_test)
# print(y_pred)
# #

# plt.scatter(X, Y, color='blue', label='Original data')
# plt.plot(X_test, y_pred, color='red', label='SVR predictions')
# plt.xlabel('X')
# plt.ylabel('y')
# plt.title('Support Vector Regression (SVR) with RBF Kernel')
# plt.legend()
# plt.show()

# x1 = np.arange(0, y_pred.size, 1)
# plt.title("SVR")
# plt.plot(x1, y_test, label="Actual")
# plt.plot(x1, y_pred, label="Prediction")
# plt.xlabel('x - axis')
# plt.ylabel('y - axis')
# plt.legend()
# plt.show()
