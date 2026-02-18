import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


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


def rolling_window_validation(
    df: pd.DataFrame,
    target_col: str = 'Volatility',
    window_size: int = 6,
    status_callback=None
) -> dict:
    """
    Rolling Window Validation for Linear Regression.
    
    Train on a fixed window, test on the next period.
    Roll forward each iteration.
    
    window_size: number of "months" (approximated as groups of ~21*12 bars,
    but since data is 5-min bars, we'll use fractional splits based on total rows).
    
    For simplicity: divide data into `window_size + remaining` chunks.
    Each chunk = total_rows / (total number of months-equivalent).
    
    Actually, let's be smarter: group by calendar month from the 'time' column.
    """
    results = {
        'model': 'Linear Regression',
        'validation': 'Rolling Window',
        'window_size': window_size,
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
    
    # Get label thresholds from data
    thresholds = _get_label_thresholds(df)
    if thresholds:
        results['thresholds'] = thresholds
        if status_callback:
            status_callback(f"Label thresholds extracted: {thresholds}")
    
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
    df_clean = df[feature_cols + [target_col]].dropna().reset_index(drop=True)
    if 'Risk_Label' in df.columns:
        # Keep Risk_Label aligned
        risk_labels = df.loc[df_clean.index, 'Risk_Label'] if len(df_clean) == len(df) else None
    
    if status_callback:
        status_callback(f"Clean data: {len(df_clean)} rows after dropping NaN.")
    
    # Split into monthly-equivalent chunks
    # Estimate: if 5-min bars, ~75 bars/day, ~1500 bars/month (20 trading days)
    # But let's just divide evenly into segments
    total_rows = len(df_clean)
    # Minimum segments needed: window_size + 1 (at least 1 test)
    n_segments = max(window_size + 2, 8)  # At least 8 segments for meaningful rolling
    segment_size = total_rows // n_segments
    
    if segment_size < 10:
        raise ValueError(f"Not enough data for {n_segments} segments. Total rows: {total_rows}")
    
    if status_callback:
        status_callback(f"Data divided into {n_segments} segments of ~{segment_size} rows each.")
    
    X = df_clean[feature_cols].values
    y = df_clean[target_col].values
    
    all_actual = []
    all_pred = []
    
    # Rolling window: train on [i..i+window_size], test on [i+window_size]
    n_folds = n_segments - window_size
    
    for fold in range(n_folds):
        train_start = fold * segment_size
        train_end = (fold + window_size) * segment_size
        test_start = train_end
        test_end = min((fold + window_size + 1) * segment_size, total_rows)
        
        if test_end <= test_start:
            break
        
        X_train = X[train_start:train_end]
        y_train = y[train_start:train_end]
        X_test = X[test_start:test_end]
        y_test = y[test_start:test_end]
        
        # Train
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
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
    initial_window: int = 3,
    status_callback=None
) -> dict:
    """
    Walk-Forward (Expanding Window) Validation for Linear Regression.
    
    Start with initial_window segments, test on next segment.
    Then expand training window by adding that segment, retrain, test on next, etc.
    """
    results = {
        'model': 'Linear Regression',
        'validation': 'Walk-Forward (Expanding)',
        'initial_window': initial_window,
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
    
    # Get label thresholds from data
    thresholds = _get_label_thresholds(df)
    if thresholds:
        results['thresholds'] = thresholds
        if status_callback:
            status_callback(f"Label thresholds extracted: {thresholds}")
    
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
    df_clean = df[feature_cols + [target_col]].dropna().reset_index(drop=True)
    
    if status_callback:
        status_callback(f"Clean data: {len(df_clean)} rows after dropping NaN.")
    
    total_rows = len(df_clean)
    n_segments = max(initial_window + 2, 8)
    segment_size = total_rows // n_segments
    
    if segment_size < 10:
        raise ValueError(f"Not enough data for {n_segments} segments. Total rows: {total_rows}")
    
    if status_callback:
        status_callback(f"Data divided into {n_segments} segments of ~{segment_size} rows each.")
    
    X = df_clean[feature_cols].values
    y = df_clean[target_col].values
    
    all_actual = []
    all_pred = []
    
    # Walk-forward: start with initial_window segments, expand each fold
    n_folds = n_segments - initial_window
    
    for fold in range(n_folds):
        train_start = 0  # Always start from beginning (expanding)
        train_end = (initial_window + fold) * segment_size
        test_start = train_end
        test_end = min((initial_window + fold + 1) * segment_size, total_rows)
        
        if test_end <= test_start:
            break
        
        X_train = X[train_start:train_end]
        y_train = y[train_start:train_end]
        X_test = X[test_start:test_end]
        y_test = y[test_start:test_end]
        
        # Train
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
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
