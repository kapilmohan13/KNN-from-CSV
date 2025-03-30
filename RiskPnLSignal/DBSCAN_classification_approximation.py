import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# # Generate sample data
# X = np.array([[1, 2], [2, 2], [2, 3],
#               [8, 7], [8, 8], [25, 80],
#               [1, 1], [7, 8], [25, 79]])
df = pd.read_csv('..//data//GOLDBEES.csv')
df['time'] = pd.to_datetime(df['time'], format='%d-%m-%Y %H:%M:%S').dt.time
data = df[['time', 'intc']]

data.sort_values(by='time', inplace=True)
data = data[['intc']]
scaler = StandardScaler()
OriginalData = data
data = scaler.fit_transform(data)
X=data


# Run DBSCAN
db = DBSCAN(algorithm='auto', eps=3, leaf_size=100, metric='euclidean',
                metric_params=None, min_samples=2, n_jobs=None, p=None)
clusters = db.fit_predict(data)
# Labels from DBSCAN
labels = db.labels_
print("DBSCAN labels:", labels)


#STEP 2
# Filter out noise points (labeled as -1 by DBSCAN)
filtered_X = X[labels != -1]
filtered_labels = labels[labels != -1]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(filtered_X, filtered_labels, test_size=0.3, random_state=42)


#STEP3
from sklearn.neighbors import KNeighborsClassifier

# Train k-NN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict on test set
y_pred_knn = knn.predict(X_test)

# Evaluate accuracy
print("k-NN Accuracy:", accuracy_score(y_test, y_pred_knn))


#STEP4
from sklearn.svm import SVC

# Train SVM classifier
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Predict on test set
y_pred_svm = svm.predict(X_test)

# Evaluate accuracy
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))


#STEP5
# New data point
# new_point = np.array([[2]])
inputdata = [[71.64]]
inputdata = np.array([[71.64]])
# scaler = load('..//models//standard_scaler.pkl')

new_point = scaler.transform(inputdata)
print(new_point[0][0])

# Predict using k-NN
print("k-NN Cluster Prediction:", knn.predict(new_point))

# Predict using SVM
print("SVM Cluster Prediction:", svm.predict(new_point))
