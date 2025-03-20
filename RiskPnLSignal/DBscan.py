import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import joblib


# Load data from CSV
def load_data(file_path):
    return pd.read_csv(file_path)


# Scale the data
def scale_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler


# Fit DBSCAN model
def fit_dbscan(scaled_data):
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(scaled_data)
    return dbscan, clusters


# Save model and scaler
def save_model_and_scaler(model, scaler, model_path, scaler_path):
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)


# Load model and scaler
def load_model_and_scaler(model_path, scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


# Predict new data
def predict_new_data(model, scaler, new_data):
    scaled_new_data = scaler.transform(new_data)
    return model.predict(scaled_new_data)


# Main function
def main():
    # Step 1: Load the data
    data = load_data('..//data//GOLDBEES_SMALL.csv')  # Replace 'data.csv' with your CSV file path
    data = data[['time', 'intc']]
    data.sort_values(by='time', inplace=True)
    data = data[['intc']]
    # Step 2: Scale the data
    scaled_data, scaler = scale_data(data)

    # Step 3: Fit DBSCAN model
    dbscan_model, clusters = fit_dbscan(scaled_data)

    # Step 4: Save model and scaler
    save_model_and_scaler(dbscan_model, scaler, 'dbscan_model.joblib', 'scaler.joblib')

    # Detect outliers (DBSCAN labels -1 are considered outliers)
    outliers = np.where(clusters == -1)[0]

    # Print results
    print("Cluster labels:", clusters)
    print("Outliers indices:", outliers)

    # Example of predicting new data after training
    # Load new data for prediction
    new_data = pd.DataFrame([[100]])  # Replace with your new data

    # Make sure the new data has the same number of features as the training data
    # Load model and scaler
    loaded_model, loaded_scaler = load_model_and_scaler('dbscan_model.joblib', 'scaler.joblib')

    # Get predictions for the new data
    if not new_data.empty:
        predictions = predict_new_data(loaded_model, loaded_scaler, new_data)
        print("New data cluster predictions:", predictions)
    else:
        print("No new data provided for prediction.")


if __name__ == "__main__":
    main()