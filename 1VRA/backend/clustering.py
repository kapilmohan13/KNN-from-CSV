
import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from pathlib import Path

class ClusteringModel:
    """
    Handles DBSCAN clustering logic, including parameter tuning (Elbow Method).
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.feature_columns = []
        
    def load_and_preprocess(self, file_path: Path):
        """
        Loads CSV, filters relevant features, and scales data.
        Excludes non-numeric and specific irrelevant columns.
        """
        df = pd.read_csv(file_path)
        
        # Identify columns to exclude
        exclude_patterns = ['date', 'time', 'timestamp', 'stat', 'ssboe', 'cluster', 'label', 'intc'] # 'intc' might be raw close price, keep it? user says "all indicators, volatility etc". Maybe keep 'intc' if it's numeric.
        # But 'stat' and 'ssboe' specifically excluded.
        # Also need to handle 'Date' or 'Time' if present.
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter exclusions
        selected_cols = []
        for col in numeric_cols:
            is_excluded = False
            col_lower = col.lower()
            for pat in exclude_patterns:
                if pat in col_lower:
                    is_excluded = True
                    break
            if not is_excluded:
                selected_cols.append(col)
        
        if not selected_cols:
            raise ValueError("No valid numeric features found for clustering after filtering.")
            
        self.feature_columns = selected_cols
        X = df[selected_cols].values
        
        # Handle NaNs (fill with mean or drop? DBSCAN hates NaNs)
        # Assuming indicators might have NaNs at start. Drop rows? Or fill? 
        # Standard: Drop rows with NaNs in features.
        # But we want to return the full DF with clusters. 
        # We'll fill NaNs with 0 or Mean. Mean is safer for indicators like RSI.
        # Actually, let's just drop NaNs for training, but we need to map back to original DF?
        # If we drop rows, the output DF will be shorter.
        # Let's fill forward/backward first, then fill 0.
        X_df = pd.DataFrame(X, columns=selected_cols)
        X_df = X_df.ffill().bfill().fillna(0)
        X = X_df.values
        
        X_scaled = self.scaler.fit_transform(X)
        return df, X_scaled, selected_cols

    def find_optimal_eps(self, X_scaled, k: int = 3):
        """
        Finds optimal epsilon using improved K-Distance Graph (Elbow Method).
        Uses multiple techniques: geometric method, derivative-based, and percentile fallback.
        k: nearest neighbor count (usually min_samples, or user defined k).
        """
        # Calculate distances to k-th nearest neighbor
        neigh = NearestNeighbors(n_neighbors=k)
        nbrs = neigh.fit(X_scaled)
        distances, indices = nbrs.kneighbors(X_scaled)
        
        # Sort distances (taking the k-th neighbor, which is at index k-1)
        k_distances = np.sort(distances[:, k-1])
        
        # METHOD 1: Geometric approach (max distance from line connecting start and end)
        n_points = len(k_distances)
        all_coords = np.vstack((range(n_points), k_distances)).T
        
        first_point = all_coords[0]
        line_vec = all_coords[-1] - all_coords[0]
        line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
        
        vec_from_first = all_coords - first_point
        scalar_prod = np.sum(vec_from_first * line_vec_norm, axis=1)
        vec_from_first_parallel = np.outer(scalar_prod, line_vec_norm)
        vec_to_line = vec_from_first - vec_from_first_parallel
        dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
        
        idx_elbow_geometric = np.argmax(dist_to_line)
        eps_geometric = k_distances[idx_elbow_geometric]
        
        # METHOD 2: Derivative-based approach (find where slope changes most)
        # Calculate first derivative (rate of change)
        derivatives = np.diff(k_distances)
        # Find where derivative changes most (second derivative)
        second_derivatives = np.diff(derivatives)
        # The elbow is where second derivative is maximum (sharpest increase)
        idx_elbow_derivative = np.argmax(second_derivatives) + 1  # +1 due to diff offset
        eps_derivative = k_distances[idx_elbow_derivative] if idx_elbow_derivative < len(k_distances) else eps_geometric
        
        # METHOD 3: Conservative percentile-based approach
        # Use 70th-75th percentile as a safer option to avoid too large eps
        eps_percentile_70 = np.percentile(k_distances, 70)
        eps_percentile_75 = np.percentile(k_distances, 75)
        
        # Choose the most conservative (smallest) among the methods
        # But not too small (at least 50th percentile)
        eps_min_threshold = np.percentile(k_distances, 50)
        
        # Take the minimum of geometric and derivative methods
        eps_candidate = min(eps_geometric, eps_derivative)
        
        # If candidate is too large (> 75th percentile), use 70th percentile instead
        if eps_candidate > eps_percentile_75:
            optimal_eps = eps_percentile_70
        else:
            # Use candidate but ensure it's not too small
            optimal_eps = max(eps_candidate, eps_min_threshold)
        
        return optimal_eps, k_distances

    def train_dbscan(self, X_scaled, eps, min_samples):
        """
        Trains DBSCAN model.
        """
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = self.model.fit_predict(X_scaled)
        
        # Calculate silhouette score (exclude noise -1 if possible, or include?)
        # Standard: metric usually ignores -1 or treats as separate.
        # Only calculate if > 1 cluster (excluding noise) or inclusive.
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
        
        sil_score = -1.0
        if n_clusters > 1:
            try:
                # Sample for performance if dataset is large? No, dataset likely small (<100k).
                sil_score = silhouette_score(X_scaled, labels) 
            except:
                sil_score = -1.0
                
        return labels, n_clusters, sil_score
    
    def find_optimal_k_kmeans(self, X_scaled, k_min: int = 2, k_max: int = 10):
        """
        Finds optimal number of clusters (K) for K-Means using the Elbow Method.
        Returns optimal K and inertia values for visualization.
        """
        inertias = []
        silhouette_scores = []
        k_range = range(k_min, k_max + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            inertias.append(kmeans.inertia_)
            
            # Calculate silhouette score
            if k > 1:
                try:
                    score = silhouette_score(X_scaled, labels)
                    silhouette_scores.append(score)
                except:
                    silhouette_scores.append(-1.0)
            else:
                silhouette_scores.append(-1.0)
        
        # Find elbow using geometric method (same as DBSCAN eps finding)
        inertias_arr = np.array(inertias)
        n_points = len(inertias_arr)
        all_coords = np.vstack((range(n_points), inertias_arr)).T
        
        first_point = all_coords[0]
        line_vec = all_coords[-1] - all_coords[0]
        line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
        
        vec_from_first = all_coords - first_point
        scalar_prod = np.sum(vec_from_first * line_vec_norm, axis=1)
        vec_from_first_parallel = np.outer(scalar_prod, line_vec_norm)
        vec_to_line = vec_from_first - vec_from_first_parallel
        dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
        
        idx_elbow = np.argmax(dist_to_line)
        optimal_k = k_min + idx_elbow
        
        return optimal_k, list(inertias), list(silhouette_scores), list(k_range)
    
    def train_kmeans(self, X_scaled, n_clusters, random_state=42):
        """
        Trains K-Means model.
        """
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = self.model.fit_predict(X_scaled)
        
        # Calculate metrics
        inertia = self.model.inertia_
        
        sil_score = -1.0
        if n_clusters > 1:
            try:
                sil_score = silhouette_score(X_scaled, labels)
            except:
                sil_score = -1.0
                
        return labels, n_clusters, sil_score, inertia
        
    def save_model(self, output_dir: Path, base_filename: str, algorithm: str = 'dbscan'):
        """
        Saves the scaler and model using pickle.
        algorithm: 'dbscan' or 'kmeans'
        """
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
            
        scaler_path = output_dir / f"{base_filename}_scaler.pkl"
        model_path = output_dir / f"{base_filename}_{algorithm}.pkl"
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
            
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
            
        return str(scaler_path), str(model_path)
