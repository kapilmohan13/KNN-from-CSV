import numpy as np
from joblib import load

dbscan_loaded = load('..//models//dbscan_model.joblib')

labels = dbscan_loaded.labels_
# print("cluster labels: ", labels)
core_samples = dbscan_loaded.components_
core_samples_indices = dbscan_loaded.core_sample_indices_
print("core samples:")
print(core_samples)
RED = '\033[91m'
RESET = '\033[0m'

def print_in_red(text):
    print(f"{RED}{text}{RESET}")

inputdata = [[71.64]]
scaler = load('..//models//standard_scaler.pkl')

new_data = scaler.transform(inputdata)
print(new_data[0][0])

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

new_labels = classify_new_points(new_data, core_samples, eps=0.02)
print(new_labels)
if(new_labels[0] == -1):
    text = str(inputdata[0][0])+" Outlier detected!! "
    print_in_red(text)
else:
    print_in_red("Passed.")