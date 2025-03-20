import numpy as np
import pandas as pd
from joblib import dump
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('..//data//GOLDBEES_SMALL.csv')
data = df[['time', 'intc']]
data.sort_values(by='time', inplace=True)
data = data[['intc']]

scaler = StandardScaler()
OriginalData = data
data = scaler.fit_transform(data)

#Save scaler
dump(scaler, '..//models//standard_scaler.pkl')


#fit model
dbscan = DBSCAN(algorithm='auto', eps=0.05, leaf_size=30, metric='euclidean',
                metric_params=None, min_samples=2, n_jobs=None, p=None)
clusters = dbscan.fit_predict(data)

#Identifying outliers
outliers = data[clusters == -1]
mapped_outliers = scaler.inverse_transform(outliers)
print(mapped_outliers)


labels = dbscan.labels_
print("cluster labels", labels)

core_samples = dbscan.components_
core_sample_indices = dbscan.core_sample_indices_

#save model
dump(dbscan, '..//models/dbscan_model.joblib')

def classify_new_points(new_data, core_samples, eps):
    new_labels = []

    for point in new_data:
        distance = np.linalg.norm(core_samples - point, axis=1)
        if np.any(distance <=eps):
            minDistance = int(np.argmin(distance))
            # print(minDistance)
            new_labels.append(int(labels[minDistance]))
        else:
            new_labels.append(-1)
    return  new_labels

print("labels:" )
print(dbscan.labels_)
#Fit new data
new_data = [[71.644]]
new_data = scaler.transform(new_data)
new_labels = classify_new_points(new_data, core_samples, eps=1)
print("any outliers?", new_labels)