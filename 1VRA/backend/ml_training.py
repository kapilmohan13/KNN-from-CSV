import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from sklearn.preprocessing import MinMaxScaler
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# Columns to exclude from features
EXCLUDE_PATTERNS = ['stat', 'ssboe', 'time', 'date', 'timestamp', 'risk_label', 'label', 'parsed']


def _deduplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle duplicate column names by keeping only the first occurrence.
    This prevents df['col'] from returning a DataFrame instead of a Series.
    """
    if df.columns.duplicated().any():
        # Keep only the first occurrence of each duplicate column
        df = df.loc[:, ~df.columns.duplicated(keep='first')]
    return df


def _select_features(df: pd.DataFrame) -> list:
    """
    Select numeric feature columns, excluding non-finance and label columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected = []
    for col in numeric_cols:
        col_lower = col.lower()
        excluded = False
        for pat in EXCLUDE_PATTERNS:
            if pat in col_lower:
                excluded = True
                break
        if not excluded:
            selected.append(col)
    return selected


def _get_label_thresholds(df: pd.DataFrame) -> dict:
    """
    Extracts volatility label thresholds from the data.
    Looks for 'Volatility' and 'Risk_Label' columns.
    Returns dict with threshold boundaries and label mapping.
    """
    if 'Volatility' not in df.columns or 'Risk_Label' not in df.columns:
        return None

    # Derive thresholds from data: find the boundary between labels
    label_groups = df.groupby('Risk_Label')['Volatility'].agg(['min', 'max'])
    
    thresholds = {}
    if 'Low Risk' in label_groups.index:
        thresholds['low_max'] = float(label_groups.loc['Low Risk', 'max'])
    if 'Medium Risk' in label_groups.index:
        thresholds['med_min'] = float(label_groups.loc['Medium Risk', 'min'])
        thresholds['med_max'] = float(label_groups.loc['Medium Risk', 'max'])
    if 'High Risk' in label_groups.index:
        thresholds['high_min'] = float(label_groups.loc['High Risk', 'min'])
    
    return thresholds


def _apply_labels(volatility_values: pd.Series, thresholds: dict) -> pd.Series:
    """
    Apply risk labels to predicted volatility values using the same thresholds
    that were used for the original labelling.
    """
    if not thresholds:
        return pd.Series(['Unknown'] * len(volatility_values))
    
    low_max = thresholds.get('low_max', 0)
    high_min = thresholds.get('high_min', float('inf'))
    
    def classify(val):
        if pd.isna(val):
            return 'Unknown'
        if val <= low_max:
            return 'Low Risk'
        elif val >= high_min:
            return 'High Risk'
        else:
            return 'Medium Risk'
    
    return volatility_values.apply(classify)


class LSTMModelWrapper:
    """
    Simplistic wrapper to make a Keras LSTM model act like a scikit-learn regressor.
    Handles internal scaling and 3D reshaping.
    """
    def __init__(self, **params):
        self.params = params
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
    def fit(self, X, y):
        # Scale
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1))
        
        # Reshape to (samples, time_steps=1, features)
        X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        
        # Build Model
        self.model = Sequential([
            LSTM(self.params.get('units', 32), input_shape=(1, X.shape[1]), activation='tanh'),
            Dropout(0.1),
            Dense(1)
        ])
        
        self.model.compile(optimizer='adam', loss='mse')
        
        # Train
        self.model.fit(
            X_reshaped, y_scaled, 
            epochs=self.params.get('epochs', 5), 
            batch_size=self.params.get('batch_size', 32),
            verbose=0
        )
        return self
        
    def predict(self, X):
        X_scaled = self.scaler_X.transform(X)
        X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        y_pred_scaled = self.model.predict(X_reshaped, verbose=0)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        return y_pred.flatten()


def _get_model(algorithm: str):
    """
    Instantiate the appropriate regression model and return its parameters.
    """
    if algorithm == 'xgboost':
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost is not installed.")
        
        params = {
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.1,
            'verbosity': 0,
            'random_state': 42,
            'n_jobs': -1
        }
        model = XGBRegressor(**params)
        return model, params
    
    elif algorithm == 'lstm':
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow/Keras is not installed.")
            
        params = {
            'units': 32,
            'epochs': 10,
            'batch_size': 32,
            'optimizer': 'adam'
        }
        model = LSTMModelWrapper(**params)
        return model, params
    
    elif algorithm == 'random_forest':
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42,
            'n_jobs': -1
        }
        model = RandomForestRegressor(**params)
        return model, params
    
    else:
        # Default to Linear Regression
        params = {
            'fit_intercept': True,
            'copy_X': True
        }
        model = LinearRegression(**params)
        return model, params


def _build_segments(df: pd.DataFrame, mode: str, size_param: int, status_callback=None) -> list:
    """
    Divides data into segments for validation.
    Returns a list of (start_idx, end_idx) tuples.
    """
    total_rows = len(df)
    
    if mode == 'month':
        time_col = next((c for c in ['time', 'datetime', 'Timestamp'] if c in df.columns), None)
        if not time_col:
            if status_callback:
                status_callback("Warning: No time/datetime column found for monthly mode. Falling back to count mode.")
            mode = 'count'
        else:
            try:
                # Format: 15-04-2025 09:15:00 or standard SQL format
                df_temp = df.copy()
                df_temp['dt'] = pd.to_datetime(df[time_col], dayfirst=True, errors='coerce')
                
                # Check if conversion worked (if not, try without dayfirst)
                if df_temp['dt'].isna().all():
                     df_temp['dt'] = pd.to_datetime(df[time_col], errors='coerce')
                
                # Check if conversion worked
                if df_temp['dt'].isna().all():
                     # Try auto format
                     df_temp['dt'] = pd.to_datetime(df['time'], errors='coerce')
                
                if df_temp['dt'].isna().any() and status_callback:
                     status_callback("Warning: Some timestamps could not be parsed.")
                
                # Group by Month
                df_temp['period'] = df_temp['dt'].dt.to_period('M')
                groups = df_temp.groupby('period').groups
                periods = sorted(groups.keys())
                
                if len(periods) <= size_param:
                    if status_callback:
                        status_callback(f"Not enough months ({len(periods)}) for window {size_param}. Falling back to count mode.")
                    mode = 'count'
                else:
                    segments = []
                    for p in periods:
                        indices = groups[p]
                        segments.append((indices[0], indices[-1] + 1))
                    
                    if status_callback:
                        status_callback(f"Divided data into {len(segments)} calendar months.")
                    return segments
            except Exception as e:
                if status_callback:
                    status_callback(f"Error in monthly grouping: {str(e)}. Falling back to count mode.")
                mode = 'count'

    # Fallback / Count Mode
    # Divide evenly into segments
    n_segments = max(size_param + 2, 10) 
    segment_size = total_rows // n_segments
    
    if segment_size < 5:
        segment_size = 5
        n_segments = total_rows // 5
        
    segments = []
    for i in range(n_segments):
        start = i * segment_size
        end = (i + 1) * segment_size if i < n_segments - 1 else total_rows
        if start < total_rows:
            segments.append((start, end))
            
    if status_callback:
        status_callback(f"Divided data into {len(segments)} segments of ~{segment_size} rows each.")
    return segments


def rolling_window_validation(
    df: pd.DataFrame,
    target_col: str = 'Volatility',
    algorithm: str = 'linear_regression',
    window_size: int = 6,
    segment_mode: str = 'count',
    status_callback=None,
    thresholds: dict = None
) -> dict:
    """
    Rolling Window Validation for a selected Algorithm.
    """
    model_obj, model_params = _get_model(algorithm)
    
    results = {
        'model': algorithm.replace('_', ' ').title(),
        'parameters': model_params,
        'validation': 'Rolling Window',
        'window_size': window_size,
        'segment_mode': segment_mode,
        'folds': [],
        'regression_metrics': {},
        'classification_metrics': {},
        'confusion_matrix': None,
        'all_actuals': [],
        'all_predictions': [],
        'all_actual_labels': [],
        'all_predicted_labels': [],
        'thresholds': {}
    }
    
    # Get label thresholds from data if not provided
    if thresholds is None:
        thresholds = _get_label_thresholds(df)
    
    if thresholds:
        results['thresholds'] = thresholds
        if status_callback:
            status_callback(f"Label thresholds used: {thresholds}")
    
    # Select features
    feature_cols = _select_features(df)
    if status_callback:
        status_callback(f"Features selected ({len(feature_cols)}): {feature_cols}")
    
    # Deduplicate columns to prevent 2D array issues
    df = _deduplicate_columns(df)
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    
    # Re-select features after dedup (column list may have changed)
    feature_cols = _select_features(df)
    # Ensure target is NOT in features (it's numeric so _select_features picks it up)
    feature_cols = [c for c in feature_cols if c != target_col]
    
    # Remove rows with NaN in features or target
    time_cols = [c for c in ['time', 'datetime', 'Timestamp'] if c in df.columns]
    keep_cols = feature_cols + [target_col] + time_cols
    df_clean = df[keep_cols].dropna().reset_index(drop=True)
    
    if status_callback:
        status_callback(f"Clean data: {len(df_clean)} rows after dropping NaN.")
    
    # Build Segments
    segments = _build_segments(df_clean, segment_mode, window_size, status_callback)
    n_segments = len(segments)
    
    if n_segments <= window_size:
        raise ValueError(f"Not enough segments ({n_segments}) for window size {window_size}.")
    
    X = df_clean[feature_cols].values
    y = df_clean[target_col].values
    
    all_actual = []
    all_pred = []
    
    # Rolling window: train on [i..i+window_size], test on [i+window_size]
    n_folds = n_segments - window_size
    
    for fold in range(n_folds):
        # Training window: segments [fold ... fold+window_size-1]
        train_start = segments[fold][0]
        train_end = segments[fold + window_size - 1][1]
        
        # Test window: segment [fold+window_size]
        test_start = segments[fold + window_size][0]
        test_end = segments[fold + window_size][1]
        
        if test_end <= test_start:
            break
        
        X_train = X[train_start:train_end]
        y_train = y[train_start:train_end]
        X_test = X[test_start:test_end]
        y_test = y[test_start:test_end]
        
        # Train
        model_obj.fit(X_train, y_train)
        y_pred = model_obj.predict(X_test)
        
        # Fold metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        fold_info = {
            'fold': fold + 1,
            'train_rows': len(X_train),
            'test_rows': len(X_test),
            'mse': float(mse),
            'mae': float(mae),
            'r2': float(r2)
        }
        results['folds'].append(fold_info)
        
        all_actual.extend(y_test.tolist())
        all_pred.extend(y_pred.tolist())
        
        if status_callback:
            status_callback(f"Fold {fold+1}/{n_folds}: MSE={mse:.6f}, MAE={mae:.6f}, R²={r2:.4f}")
    
    # Aggregate regression metrics
    all_actual = np.array(all_actual)
    all_pred = np.array(all_pred)
    
    results['regression_metrics'] = {
        'overall_mse': float(mean_squared_error(all_actual, all_pred)),
        'overall_mae': float(mean_absolute_error(all_actual, all_pred)),
        'overall_r2': float(r2_score(all_actual, all_pred)),
        'overall_rmse': float(np.sqrt(mean_squared_error(all_actual, all_pred)))
    }
    
    results['all_actuals'] = all_actual.tolist()
    results['all_predictions'] = all_pred.tolist()
    
    # Classification metrics (label predicted volatility)
    if thresholds:
        actual_labels = _apply_labels(pd.Series(all_actual), thresholds)
        predicted_labels = _apply_labels(pd.Series(all_pred), thresholds)
        
        results['all_actual_labels'] = actual_labels.tolist()
        results['all_predicted_labels'] = predicted_labels.tolist()
        
        # Get unique labels in consistent order
        label_order = ['Low Risk', 'Medium Risk', 'High Risk']
        present_labels = sorted(set(actual_labels) | set(predicted_labels), 
                               key=lambda x: label_order.index(x) if x in label_order else 99)
        
        results['classification_metrics'] = {
            'accuracy': float(accuracy_score(actual_labels, predicted_labels)),
            'precision_macro': float(precision_score(actual_labels, predicted_labels, average='macro', zero_division=0)),
            'recall_macro': float(recall_score(actual_labels, predicted_labels, average='macro', zero_division=0)),
            'f1_macro': float(f1_score(actual_labels, predicted_labels, average='macro', zero_division=0)),
            'classification_report': classification_report(actual_labels, predicted_labels, zero_division=0)
        }
        
        cm = confusion_matrix(actual_labels, predicted_labels, labels=present_labels)
        results['confusion_matrix'] = {
            'matrix': cm.tolist(),
            'labels': present_labels
        }
        
        if status_callback:
            status_callback(f"Classification Accuracy: {results['classification_metrics']['accuracy']:.4f}")
    
    return results


def walk_forward_validation(
    df: pd.DataFrame,
    target_col: str = 'Volatility',
    algorithm: str = 'linear_regression',
    initial_window: int = 3,
    segment_mode: str = 'count',
    status_callback=None,
    thresholds: dict = None
) -> dict:
    """
    Walk-Forward (Expanding Window) Validation for a selected Algorithm.
    """
    model_obj, model_params = _get_model(algorithm)
    
    results = {
        'model': algorithm.replace('_', ' ').title(),
        'parameters': model_params,
        'validation': 'Walk-Forward (Expanding)',
        'initial_window': initial_window,
        'segment_mode': segment_mode,
        'folds': [],
        'regression_metrics': {},
        'classification_metrics': {},
        'confusion_matrix': None,
        'all_actuals': [],
        'all_predictions': [],
        'all_actual_labels': [],
        'all_predicted_labels': [],
        'thresholds': {}
    }
    
    # Get label thresholds from data if not provided
    if thresholds is None:
        thresholds = _get_label_thresholds(df)
        
    if thresholds:
        results['thresholds'] = thresholds
        if status_callback:
            status_callback(f"Label thresholds used: {thresholds}")
    
    # Select features
    feature_cols = _select_features(df)
    if status_callback:
        status_callback(f"Features selected ({len(feature_cols)}): {feature_cols}")
    
    # Deduplicate columns to prevent 2D array issues
    df = _deduplicate_columns(df)
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    
    # Re-select features after dedup
    feature_cols = _select_features(df)
    # Ensure target is NOT in features
    feature_cols = [c for c in feature_cols if c != target_col]
    
    # Clean data
    time_cols = [c for c in ['time', 'datetime', 'Timestamp'] if c in df.columns]
    keep_cols = feature_cols + [target_col] + time_cols
    df_clean = df[keep_cols].dropna().reset_index(drop=True)
    
    if status_callback:
        status_callback(f"Clean data: {len(df_clean)} rows after dropping NaN.")
    
    # Build Segments
    segments = _build_segments(df_clean, segment_mode, initial_window, status_callback)
    n_segments = len(segments)
    
    if n_segments <= initial_window:
        raise ValueError(f"Not enough segments ({n_segments}) for initial window {initial_window}.")
    
    X = df_clean[feature_cols].values
    y = df_clean[target_col].values
    
    all_actual = []
    all_pred = []
    
    # Walk-forward: start with initial_window segments, expand each fold
    n_folds = n_segments - initial_window
    
    for fold in range(n_folds):
        train_start = segments[0][0]  # Always start from beginning (expanding)
        train_end = segments[initial_window + fold - 1][1]
        test_start = segments[initial_window + fold][0]
        test_end = segments[initial_window + fold][1]
        
        if test_end <= test_start:
            break
        
        X_train = X[train_start:train_end]
        y_train = y[train_start:train_end]
        X_test = X[test_start:test_end]
        y_test = y[test_start:test_end]
        
        # Train
        model_obj.fit(X_train, y_train)
        y_pred = model_obj.predict(X_test)
        
        # Fold metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        fold_info = {
            'fold': fold + 1,
            'train_rows': len(X_train),
            'test_rows': len(X_test),
            'mse': float(mse),
            'mae': float(mae),
            'r2': float(r2)
        }
        results['folds'].append(fold_info)
        
        all_actual.extend(y_test.tolist())
        all_pred.extend(y_pred.tolist())
        
        if status_callback:
            status_callback(f"Fold {fold+1}/{n_folds}: Train={len(X_train)}, Test={len(X_test)}, MSE={mse:.6f}, R²={r2:.4f}")
    
    # Aggregate regression metrics
    all_actual = np.array(all_actual)
    all_pred = np.array(all_pred)
    
    results['regression_metrics'] = {
        'overall_mse': float(mean_squared_error(all_actual, all_pred)),
        'overall_mae': float(mean_absolute_error(all_actual, all_pred)),
        'overall_r2': float(r2_score(all_actual, all_pred)),
        'overall_rmse': float(np.sqrt(mean_squared_error(all_actual, all_pred)))
    }
    
    results['all_actuals'] = all_actual.tolist()
    results['all_predictions'] = all_pred.tolist()
    
    # Classification metrics
    if thresholds:
        actual_labels = _apply_labels(pd.Series(all_actual), thresholds)
        predicted_labels = _apply_labels(pd.Series(all_pred), thresholds)
        
        results['all_actual_labels'] = actual_labels.tolist()
        results['all_predicted_labels'] = predicted_labels.tolist()
        
        label_order = ['Low Risk', 'Medium Risk', 'High Risk']
        present_labels = sorted(set(actual_labels) | set(predicted_labels),
                               key=lambda x: label_order.index(x) if x in label_order else 99)
        
        results['classification_metrics'] = {
            'accuracy': float(accuracy_score(actual_labels, predicted_labels)),
            'precision_macro': float(precision_score(actual_labels, predicted_labels, average='macro', zero_division=0)),
            'recall_macro': float(recall_score(actual_labels, predicted_labels, average='macro', zero_division=0)),
            'f1_macro': float(f1_score(actual_labels, predicted_labels, average='macro', zero_division=0)),
            'classification_report': classification_report(actual_labels, predicted_labels, zero_division=0)
        }
        
        cm = confusion_matrix(actual_labels, predicted_labels, labels=present_labels)
        results['confusion_matrix'] = {
            'matrix': cm.tolist(),
            'labels': present_labels
        }
        
        if status_callback:
            status_callback(f"Classification Accuracy: {results['classification_metrics']['accuracy']:.4f}")
    
    return results


def save_results_to_file(results: dict, output_dir: Path, prefix: str = 'ml') -> dict:
    """
    Saves regression metrics, classification report, and confusion matrix to files.
    Returns dict of saved file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = {}
    
    # 1. Regression metrics
    reg_file = output_dir / f"{prefix}_regression_metrics.json"
    with open(reg_file, 'w') as f:
        json.dump({
            'model': results.get('model'),
            'validation': results.get('validation'),
            'regression_metrics': results.get('regression_metrics'),
            'folds': results.get('folds')
        }, f, indent=2)
    saved_files['regression_metrics'] = str(reg_file)
    
    # 2. Classification report
    if results.get('classification_metrics'):
        cls_file = output_dir / f"{prefix}_classification_report.txt"
        with open(cls_file, 'w') as f:
            f.write(f"Model: {results.get('model')}\n")
            f.write(f"Validation: {results.get('validation')}\n")
            f.write(f"Thresholds: {results.get('thresholds')}\n\n")
            f.write(f"Accuracy: {results['classification_metrics']['accuracy']:.4f}\n")
            f.write(f"Precision (macro): {results['classification_metrics']['precision_macro']:.4f}\n")
            f.write(f"Recall (macro): {results['classification_metrics']['recall_macro']:.4f}\n")
            f.write(f"F1 (macro): {results['classification_metrics']['f1_macro']:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(results['classification_metrics']['classification_report'])
        saved_files['classification_report'] = str(cls_file)
    
    # 3. Confusion matrix
    if results.get('confusion_matrix'):
        cm_file = output_dir / f"{prefix}_confusion_matrix.json"
        with open(cm_file, 'w') as f:
            json.dump(results['confusion_matrix'], f, indent=2)
        saved_files['confusion_matrix'] = str(cm_file)
    
    # 4. Predictions CSV
    if results.get('all_actuals') and results.get('all_predictions'):
        pred_df = pd.DataFrame({
            'Actual_Volatility': results['all_actuals'],
            'Predicted_Volatility': results['all_predictions']
        })
        if results.get('all_actual_labels'):
            pred_df['Actual_Label'] = results['all_actual_labels']
        if results.get('all_predicted_labels'):
            pred_df['Predicted_Label'] = results['all_predicted_labels']
        
        pred_file = output_dir / f"{prefix}_predictions.csv"
        pred_df.to_csv(pred_file, index=False)
        saved_files['predictions'] = str(pred_file)
    
    return saved_files
