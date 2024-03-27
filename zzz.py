import pickle
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def main():

    df = pd.read_csv('data//ESG_large_set.csv')

    X = df.iloc[:, 2:11].values
    Y = df.iloc[:, 10].values

    # Standardize features
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)

    # Splitting
    train_X, test_X, train_y, test_y = train_test_split(X, Y,
                                                        test_size=0.0001, random_state=123)

    print(train_X)
    print(train_y)
    print(test_X)
    print(test_y)





if __name__ == "__main__":
    main()