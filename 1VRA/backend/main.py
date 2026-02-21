import os
import shutil
import pandas as pd
import numpy as np
import logging
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
from pathlib import Path
from indicators import TechnicalIndicators
from labelling import RiskLabeller
from clustering import ClusteringModel
from ml_training import rolling_window_validation, walk_forward_validation, save_results_to_file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="VRA Backend")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Global status tracker (simple in-memory for this single-user app)
processing_status = {
    "status": "idle",
    "message": "",
    "details": [],
    "last_file": "",
    "output_file": ""
}

# Track cluster output files per algorithm (for Hybrid-ML tab)
cluster_output_files = {
    "dbscan": "",
    "kmeans": ""
}

def update_status(status, message, detail=None):
    processing_status["status"] = status
    processing_status["message"] = message
    if detail:
        processing_status["details"].append(f"{datetime.now().strftime('%H:%M:%S')} - {detail}")
        logger.info(detail)

def process_file_task(file_path: Path, output_path: Path):
    try:
        update_status("processing", "Starting processing...", "Loading file...")
        
        # Load Data
        try:
            df = pd.read_csv(file_path)
            update_status("processing", "File loaded", f"Loaded {len(df)} rows. Columns: {list(df.columns)}")
        except Exception as e:
            update_status("error", "Failed to load file", f"Error: {str(e)}")
            return

        # Validate Columns
        required_cols = ['time', 'intc'] # Minimum required
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
             update_status("error", "Validation failed", f"Missing columns: {missing}")
             return

        # Sort by Time
        update_status("processing", "Sorting data", "Parsing timestamps and sorting...")
        try:
            # Assuming format "10-02-2026 15:25:00" from user example
            df['parsed_time'] = pd.to_datetime(df['time'], dayfirst=True) 
            df = df.sort_values('parsed_time', ascending=True).reset_index(drop=True)
            df.drop(columns=['parsed_time'], inplace=True)
            update_status("processing", "Sorted", f"Data sorted from {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
        except Exception as e:
            update_status("error", "Sorting failed", f"Error parsing dates: {str(e)}")
            return

        # Calculate Indicators
        update_status("processing", "Calculating Indicators", "Starting technical analysis...")
        
        # Mapping columns
        close = df['intc']
        # Use intv (interval volume) if available, else v, else 0
        if 'intv' in df.columns:
            volume = df['intv']
        elif 'v' in df.columns:
            volume = df['v']
        else:
            volume = pd.Series(np.ones(len(df))) # Fallback
            
        high = df['inth'] if 'inth' in df.columns else close
        low = df['intl'] if 'intl' in df.columns else close
        
        # RSI (6? User mentioned RSI, GenerateIndicators had default 14 or 6. User said "RSI". I'll use 14 as standard default if not specified)
        # Update: User "RSI" usually implies 14. 
        update_status("processing", "Calculating RSI", "Calculating Relative Strength Index (14)...")
        df['RSI'] = TechnicalIndicators.rsi(close, period=14)
        
        # Stochastic RSI
        update_status("processing", "Calculating Stoch RSI", "Calculating Stochastic RSI...")
        df['StochRSI'] = TechnicalIndicators.stochastic_rsi(close)
        
        # MACD
        update_status("processing", "Calculating MACD", "Calculating MACD (12, 26, 9)...")
        macd_df = TechnicalIndicators.macd(close)
        df = pd.concat([df, macd_df], axis=1)
        
        # MA
        update_status("processing", "Calculating MA", "Calculating Moving Average (16)...")
        df['MA_16'] = TechnicalIndicators.moving_average(close, period=16)
        
        # ROC
        update_status("processing", "Calculating ROC", "Calculating Rate of Change (10)...")
        df['ROC'] = TechnicalIndicators.rate_of_change(close, period=10)
        
        # Williams %R
        update_status("processing", "Calculating Williams %R", "Calculating Williams %R (14)...")
        df['Williams_R'] = TechnicalIndicators.williams_r(high, low, close)
        
        # VWMA
        update_status("processing", "Calculating VWMA", "Calculating Volume Weighted MA...")
        df['VWMA'] = TechnicalIndicators.vwma(close, volume)
        
        # Linear Regression MA
        update_status("processing", "Calculating LRMA", "Calculating Linear Regression MA (14)... this might take a moment.")
        df['LRMA'] = TechnicalIndicators.linear_regression_ma(close, period=14)

        # Save Output
        update_status("processing", "Saving", f"Saving processed file to {output_path}...")
        df.to_csv(output_path, index=False)
        
        processing_status["output_file"] = str(output_path)
        update_status("completed", "Processing Complete", f"Successfully saved to {output_path}")


    except Exception as e:
        update_status("error", "Unexpected Error", f"Critical processing error: {str(e)}")
        logger.error(e, exc_info=True)


def process_volatility_task(file_path: Path, output_path: Path, window: int):
    try:
        update_status("processing", "Starting Volatility Calculation", f"Loading {file_path}...")
        
        # Load Data
        try:
            df = pd.read_csv(file_path)
            update_status("processing", "File loaded", f"Loaded {len(df)} rows.")
        except Exception as e:
            update_status("error", "Failed to load file", f"Error: {str(e)}")
            return

        # Check for 'intc' (Close Price)
        col_name = 'intc'
        if col_name not in df.columns:
            # Fallback for standard CSVs if 'intc' isn't there, look for 'Close' or 'close'
            found = False
            for c in ['Close', 'close']:
                if c in df.columns:
                    col_name = c
                    found = True
                    break
            if not found:
                update_status("error", "Column Missing", "Could not find 'intc', 'Close', or 'close' column.")
                return

        # Calculate Volatility
        update_status("processing", "Computing Volatility", f"Calculating Std Dev of Log Returns (Window={window})...")
        try:
            close = df[col_name]
            # Use the new volatility method
            df['Volatility'] = TechnicalIndicators.volatility(close, period=window)
        except Exception as e:
             update_status("error", "Calculation Error", f"Error calculating volatility: {str(e)}")
             return

        # Save Output
        update_status("processing", "Saving", f"Saving file to {output_path}...")
        df.to_csv(output_path, index=False)
        
        processing_status["output_file"] = str(output_path)
        # Update last_file so subsequent steps can check it if needed, or user can re-use it
        # However, last_file usually tracks the Uploaded file. 
        # But for 'pre-selecting', the frontend might want the OUTPUT of the last step.
        # I'll keep output_file updated.
        
        update_status("completed", "Processing Complete", f"Volatilty added. Saved to {output_path}")

    except Exception as e:
        update_status("error", "Unexpected Error", f"Critical error: {str(e)}")
        logger.error(e, exc_info=True)



@app.post("/upload")
async def upload_file(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    # Reset status
    processing_status["status"] = "starting"
    processing_status["details"] = []
    processing_status["message"] = "Upload started"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{file.filename}"
    file_path = DATA_DIR / filename
    output_filename = f"PROCESSED_{filename}"
    output_path = DATA_DIR / output_filename
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    processing_status["last_file"] = filename
    update_status("queued", "File queued", f"File uploaded to {file_path}")
    
    # Start background processing
    background_tasks.add_task(process_file_task, file_path, output_path)
    
    return {"message": "File uploaded successfully, processing started", "filename": filename}


@app.post("/calculate-volatility")
async def calculate_volatility(
    file_path: str = None, 
    window: int = 21, 
    background_tasks: BackgroundTasks = None
):
    # If no file path provided, use the last output file from processing
    if not file_path:
        if processing_status["output_file"]:
            file_path = processing_status["output_file"]
        else:
            raise HTTPException(status_code=400, detail="No file selected and no previous output available.")
    
    # Clean up path string if it comes from frontend with extra quotes or something (basic safety)
    file_path_obj = Path(file_path)
    
    if not file_path_obj.exists():
         raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    # Determine output path (overwrite or modified name? Prompt says "written back", implying overwrite OR update.
    # To be safe and enable chainability without destroying history, I'll prefer a new file, 
    # BUT prompt checks "IF the file already have vlatility... it should be calculated and written back".
    # I'll overwrite the INPUT file if it's already a "PROCESSED" file, or separate.
    # Actually, simpler to just save to a new name to avoid open-file conflicts on Windows?
    # Let's try to overwrite if it's in DATA_DIR.
    # PROMPT: "User should have the option for other file as well... The file generated by indicators calculor should be preselected"
    
    # Strategy: output to the same filename if possible, or append _VOL if it's a raw upload.
    # If the file is already "PROCESSED_...", let's just update it in place? 
    # Or creates "VOL_PROCESSED..."?
    # I'll go with updating the file in place if it's a generated file, effectively "adding" the column.
    
    # However, to avoid "The process cannot access the file because it is being used by another process" (common on Windows),
    # I will write to a temp name then replace, or just write to a new name "VOL_..." just to be safe and visible.
    
    timestamp = datetime.now().strftime("%H%M%S")
    output_filename = f"VOL_{file_path_obj.name}" if not file_path_obj.name.startswith("VOL_") else file_path_obj.name
    output_path = DATA_DIR / output_filename
    
    # Reset status
    processing_status["status"] = "starting"
    processing_status["details"] = []
    processing_status["message"] = "Volatility Calculation Queued"
    
    background_tasks.add_task(process_volatility_task, file_path_obj, output_path, window)
    
    return {"message": "Volatility calculation started", "output_path": str(output_path)}


def process_labeling_task(file_path: Path, output_path: Path, strategy: str):
    try:
        update_status("processing", "Starting Risk Labeling", f"Loading {file_path}...")
        
        # Load Data
        df = pd.read_csv(file_path)
        
        col_name = 'Volatility'
        if col_name not in df.columns:
             update_status("error", "Column Missing", f"'{col_name}' column required for labeling.")
             return
            
        update_status("processing", "Applying Labels", f"Strategy: {strategy}...")
        
        if strategy == "relative_thresholds":
            try:
                RiskLabeller.relative_thresholds(df, col_name)
            except Exception as e:
                update_status("error", "Labeling Failed", str(e))
                return
        
        # Save Output
        update_status("processing", "Saving", f"Saving file to {output_path}...")
        df.to_csv(output_path, index=False)
        
        processing_status["output_file"] = str(output_path)
        update_status("completed", "Processing Complete", f"Labels added. Saved to {output_path}")

    except Exception as e:
        update_status("error", "Unexpected Error", f"Critical error: {str(e)}")
        logger.error(e, exc_info=True)


@app.post("/apply-labels")
async def apply_labels(
    file_path: str = None, 
    strategy: str = "relative_thresholds", 
    background_tasks: BackgroundTasks = None
):
    if not file_path:
        if processing_status["output_file"]:
            file_path = processing_status["output_file"]
        else:
            raise HTTPException(status_code=400, detail="No file selected.")
            
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
         raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    timestamp = datetime.now().strftime("%H%M%S")
    # Clean up name: prevent LBL_LBL_LBL_...
    base_name = file_path_obj.name
    if base_name.startswith("LBL_"):
        output_filename = base_name # overwrite or keep same prefix
    elif base_name.startswith("VOL_"):
        # e.g. VOL_data.csv -> LBL_data.csv (removing VOL usually cleaner but let's just make it LBL_VOL_...)
        # prompt: "The file generated by indicators calculor should be preselected... add this column after... preview screen will show these labels"
        # I'll create `LBL_{base_name}`
        output_filename = f"LBL_{base_name}"
    else:
        output_filename = f"LBL_{base_name}"
        
    output_path = DATA_DIR / output_filename
    
    # Reset status
    processing_status["status"] = "starting"
    processing_status["details"] = []
    processing_status["message"] = "Risk Labeling Queued"
    
    background_tasks.add_task(process_labeling_task, file_path_obj, output_path, strategy)
    
    return {"message": "Labeling started", "output_path": str(output_path)}


def process_clustering_task(file_path: Path, output_path: Path, n: int, k: int):
    try:
        update_status("processing", "Starting Clustering", f"Loading {file_path}...")
        
        cluster_model = ClusteringModel()
        
        # Load & Preprocess
        try:
            df, X_scaled, features = cluster_model.load_and_preprocess(file_path)
            update_status("processing", "Data Loaded", f"Features: {len(features)} selected.")
        except Exception as e:
            update_status("error", "Preprocessing Failed", str(e))
            return

        # Find Optimal Eps (Elbow Method)
        update_status("processing", "Tuning Parameters", f"Calculating K-Distance (k={k})...")
        try:
            optimal_eps, k_distances = cluster_model.find_optimal_eps(X_scaled, k=k)
            # Store for visualization
            processing_status["k_distances"] = k_distances.tolist()[:500]  # Limit to 500 points for performance
            processing_status["optimal_eps"] = float(optimal_eps)
            # Log finding
            update_status("processing", "Optimal Eps Found", f"Elbow at eps={optimal_eps:.4f}")
        except Exception as e:
            update_status("error", "Tuning Failed", str(e))
            return 
            
        # Run DBSCAN
        update_status("processing", "Running DBSCAN", f"Eps={optimal_eps:.4f}, Min_Samples={n}...")
        try:
            labels, n_clusters, sil_score = cluster_model.train_dbscan(X_scaled, optimal_eps, n)
            
            # Analyze Cluster Sizes
            unique, counts = np.unique(labels, return_counts=True)
            dist_str = ", ".join([f"{u}: {c}" for u, c in zip(unique, counts)])
            
            update_status("processing", "Clustering Complete", 
                          f"Clusters: {n_clusters}, Noise: {list(labels).count(-1)}. Score: {sil_score:.4f}")
        except Exception as e:
             update_status("error", "DBSCAN Failed", str(e))
             return

        # Save Results
        df['Cluster'] = labels
        
        # One-hot encode cluster assignments for hybrid model use
        update_status("processing", "One-Hot Encoding", "Generating one-hot encoded cluster columns...")
        cluster_dummies = pd.get_dummies(df['Cluster'], prefix='Cluster').astype(int)
        df = pd.concat([df, cluster_dummies], axis=1)
        update_status("processing", "Encoding Complete", f"Added {len(cluster_dummies.columns)} one-hot columns: {list(cluster_dummies.columns)}")
        
        df.to_csv(output_path, index=False)
        
        # Save Model
        models_dir = MODELS_DIR
            
        base_name = file_path.stem
        # Ensure unique model name with timestamp?
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_base = f"{base_name}_{timestamp}"
        
        scaler_path, model_path = cluster_model.save_model(models_dir, model_base, algorithm='dbscan')
        
        # Log Final Success
        processing_status["output_file"] = str(output_path)
        cluster_output_files["dbscan"] = str(output_path)
        img_url = "" # Could generate plot image path here if we had plotting
        
        msg = f"Clustering Done. Found {n_clusters} clusters (eps={optimal_eps:.2f}). Saved to {output_path} and models/{model_base}..."
        update_status("completed", "Processing Complete", msg)

    except Exception as e:
        update_status("error", "Unexpected Error", f"Critical error: {str(e)}")
        logger.error(e, exc_info=True)


@app.post("/train-dbscan")
async def train_dbscan(
    file_path: str = None, 
    n: int = 100, 
    k: int = 3, 
    background_tasks: BackgroundTasks = None
):
    if not file_path:
        if processing_status["output_file"]:
            file_path = processing_status["output_file"]
        else:
            raise HTTPException(status_code=400, detail="No file selected.")
            
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
         raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    # Output Filename: CLUSTER_LBL_...
    base_name = file_path_obj.name
    if not base_name.startswith("CLUSTER_"):
        output_filename = f"CLUSTER_{base_name}"
    else:
        output_filename = base_name
        
    output_path = DATA_DIR / output_filename
    
    # Reset status
    processing_status["status"] = "starting"
    processing_status["details"] = []
    processing_status["message"] = "Clustering Queued"
    
    background_tasks.add_task(process_clustering_task, file_path_obj, output_path, n, k)
    
    return {"message": "Clustering started", "output_path": str(output_path)}


@app.post("/train-kmeans")
async def train_kmeans(
    file_path: str = None, 
    k: str = "auto",  # Can be 'auto' or a number
    k_min: int = 2,
    k_max: int = 10,
    background_tasks: BackgroundTasks = None
):
    if not file_path:
        if processing_status["output_file"]:
            file_path = processing_status["output_file"]
        else:
            raise HTTPException(status_code=400, detail="No file selected.")
            
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
         raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    # Output Filename: CLUSTER_LBL_...
    base_name = file_path_obj.name
    if not base_name.startswith("CLUSTER_"):
        output_filename = f"CLUSTER_{base_name}"
    else:
        output_filename = base_name
        
    output_path = DATA_DIR / output_filename
    
    # Parse K value
    if k == "auto" or k == "":
        k_value = None  # Will use elbow method
    else:
        try:
            k_value = int(k)
        except:
            k_value = None
    
    # Reset status
    processing_status["status"] = "starting"
    processing_status["details"] = []
    processing_status["message"] = "K-Means Clustering Queued"
    
    background_tasks.add_task(process_kmeans_task, file_path_obj, output_path, k_value, k_min, k_max)
    
    return {"message": "K-Means clustering started", "output_path": str(output_path)}


def process_kmeans_task(file_path: Path, output_path: Path, k_clusters: int, k_min: int, k_max: int):
    try:
        update_status("processing", "Starting K-Means Clustering", f"Loading {file_path}...")
        
        cluster_model = ClusteringModel()
        
        # Load & Preprocess
        try:
            df, X_scaled, features = cluster_model.load_and_preprocess(file_path)
            update_status("processing", "Data Loaded", f"Features: {len(features)} selected.")
        except Exception as e:
            update_status("error", "Preprocessing Failed", str(e))
            return

        # Determine optimal K if not provided
        if k_clusters is None:
            update_status("processing", "Finding Optimal K", f"Testing K range {k_min}-{k_max}...")
            try:
                optimal_k, inertias, sil_scores, k_range = cluster_model.find_optimal_k_kmeans(X_scaled, k_min, k_max)
                # Store for visualization
                processing_status["inertias"] = inertias
                processing_status["silhouette_scores"] = sil_scores
                processing_status["k_range"] = k_range
                processing_status["optimal_k"] = int(optimal_k)
                k_clusters = optimal_k
                update_status("processing", "Optimal K Found", f"Elbow at K={k_clusters}")
            except Exception as e:
                update_status("error", "K Finding Failed", str(e))
                return
        else:
            update_status("processing", "Using Specified K", f"K={k_clusters}")
            
        # Run K-Means
        update_status("processing", "Running K-Means", f"K={k_clusters}...")
        try:
            labels, n_clusters, sil_score, inertia = cluster_model.train_kmeans(X_scaled, k_clusters)
            
            # Analyze Cluster Sizes
            unique, counts = np.unique(labels, return_counts=True)
            dist_str = ", ".join([f"{u}: {c}" for u, c in zip(unique, counts)])
            
            update_status("processing", "Clustering Complete", 
                          f"Clusters: {n_clusters}, Inertia: {inertia:.2f}, Score: {sil_score:.4f}")
        except Exception as e:
             update_status("error", "K-Means Failed", str(e))
             return

        # Save Results
        df['Cluster'] = labels
        
        # One-hot encode cluster assignments for hybrid model use
        update_status("processing", "One-Hot Encoding", "Generating one-hot encoded cluster columns...")
        cluster_dummies = pd.get_dummies(df['Cluster'], prefix='Cluster').astype(int)
        df = pd.concat([df, cluster_dummies], axis=1)
        update_status("processing", "Encoding Complete", f"Added {len(cluster_dummies.columns)} one-hot columns: {list(cluster_dummies.columns)}")
        
        df.to_csv(output_path, index=False)
        
        update_status("processing", "Saving Models", "Persisting trained model...")
        try:
            model_base = output_path.stem
            scaler_path, model_path = cluster_model.save_model(MODELS_DIR, model_base, algorithm='kmeans')
            update_status("success", "K-Means Clustering Complete", 
                          f"Results saved. Models: {model_base}")
        except Exception as e:
            update_status("error", "Model Save Failed", str(e))
            return
        
        processing_status["output_file"] = str(output_path)
        cluster_output_files["kmeans"] = str(output_path)
        processing_status["status"] = "completed"
        processing_status["message"] = f"K-Means Done. Found {n_clusters} clusters. Saved to {output_path}"
        
    except Exception as e:
        update_status("error", "Unexpected Error", f"Critical error: {str(e)}")
        logger.error(e, exc_info=True)


# ===================== ML TRAINING =====================

# Global store for ML results (single-user app)
ml_results_store = {}

def process_ml_training_task(
    file_path: Path, 
    algorithm: str, 
    validation_type: str, 
    window_size: int,
    segment_mode: str
):
    try:
        update_status("processing", "Starting ML Training", f"Loading {file_path}...")
        
        df = pd.read_csv(file_path)
        update_status("processing", "Data Loaded", f"Loaded {len(df)} rows.")
        
        # Check required columns
        if 'Volatility' not in df.columns:
            update_status("error", "Column Missing", "'Volatility' column not found. Run Volatility calculation first.")
            return
        
        def status_cb(msg):
            update_status("processing", "Training", msg)
        
        update_status("processing", "Training Started", 
                      f"Algorithm: {algorithm}, Validation: {validation_type}, Window: {window_size}, Mode: {segment_mode}")
        
        if validation_type == 'rolling':
            results = rolling_window_validation(
                df, target_col='Volatility', algorithm=algorithm, window_size=window_size, segment_mode=segment_mode, status_callback=status_cb
            )
        elif validation_type == 'walk_forward':
            results = walk_forward_validation(
                df, target_col='Volatility', algorithm=algorithm, initial_window=window_size, segment_mode=segment_mode, status_callback=status_cb
            )
        else:
            update_status("error", "Invalid Validation", f"Unknown validation type: {validation_type}")
            return
        
        # Save results to files
        update_status("processing", "Saving Results", "Writing metrics and predictions to files...")
        prefix = f"{algorithm}_{validation_type}"
        output_dir = DATA_DIR / "ml_results"
        saved_files = save_results_to_file(results, output_dir, prefix)
        
        # Store results in memory for API access
        ml_results_store['latest'] = results
        ml_results_store['saved_files'] = saved_files
        
        # Build summary
        reg = results.get('regression_metrics', {})
        cls = results.get('classification_metrics', {})
        summary_parts = [
            f"Overall MSE: {reg.get('overall_mse', 0):.6f}",
            f"Overall MAE: {reg.get('overall_mae', 0):.6f}",
            f"Overall R²: {reg.get('overall_r2', 0):.4f}",
            f"Overall RMSE: {reg.get('overall_rmse', 0):.6f}"
        ]
        if cls:
            summary_parts.append(f"Classification Accuracy: {cls.get('accuracy', 0):.4f}")
            summary_parts.append(f"F1 Macro: {cls.get('f1_macro', 0):.4f}")
        
        summary = " | ".join(summary_parts)
        
        processing_status["output_file"] = saved_files.get('predictions', '')
        # Store file paths in status for the frontend
        processing_status["ml_saved_files"] = saved_files
        
        update_status("completed", "ML Training Complete", 
                      f"Done. {len(results.get('folds', []))} folds completed. {summary}")
        update_status("completed", "ML Training Complete",
                      f"Files saved to: {output_dir}")
    
    except Exception as e:
        update_status("error", "ML Training Error", f"Critical error: {str(e)}")
        logger.error(e, exc_info=True)


@app.post("/train-ml")
async def train_ml(
    file_path: str = None,
    algorithm: str = "linear_regression",
    validation_type: str = "rolling",
    window_size: int = 6,
    segment_mode: str = "count",
    background_tasks: BackgroundTasks = None
):
    if not file_path:
        if processing_status["output_file"]:
            file_path = processing_status["output_file"]
        else:
            raise HTTPException(status_code=400, detail="No file selected.")
    
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    
    # Reset status
    processing_status["status"] = "starting"
    processing_status["details"] = []
    processing_status["message"] = "ML Training Queued"
    
    background_tasks.add_task(
        process_ml_training_task, file_path_obj, algorithm, validation_type, window_size, segment_mode
    )
    
    return {"message": "ML Training started", "validation": validation_type, "window": window_size}


@app.get("/ml-results")
async def get_ml_results():
    if 'latest' not in ml_results_store:
        return {"error": "No ML results available yet. Run training first."}
    
    results = ml_results_store['latest']
    saved_files = ml_results_store.get('saved_files', {})
    
    return {
        "model": results.get('model'),
        "parameters": results.get('parameters'),
        "validation": results.get('validation'),
        "segment_mode": results.get('segment_mode'),
        "regression_metrics": results.get('regression_metrics'),
        "classification_metrics": {
            k: v for k, v in results.get('classification_metrics', {}).items() 
            if k != 'classification_report'
        },
        "classification_report": results.get('classification_metrics', {}).get('classification_report', ''),
        "folds": results.get('folds'),
        "thresholds": results.get('thresholds'),
        "saved_files": saved_files
    }


@app.get("/ml-confusion-matrix")
async def get_confusion_matrix():
    if 'latest' not in ml_results_store:
        return {"error": "No ML results available yet."}
    
    results = ml_results_store['latest']
    cm = results.get('confusion_matrix')
    if not cm:
        return {"error": "No confusion matrix available. Check if labels were present in the data."}
    
    return cm


# ===================== HYBRID-ML TRAINING =====================

# Global store for hybrid ML results
hybrid_results_store = {}

def process_hybrid_training_task(
    file_path: Path,
    algorithm: str,
    validation_type: str,
    window_size: int,
    segment_mode: str,
    cluster_source: str
):
    try:
        update_status("processing", "Starting Hybrid-ML Training", f"Loading {file_path}...")
        
        df = pd.read_csv(file_path)
        update_status("processing", "Data Loaded", f"Loaded {len(df)} rows, {len(df.columns)} columns.")
        
        # Check required columns
        if 'Volatility' not in df.columns:
            update_status("error", "Column Missing", "'Volatility' column not found.")
            return
        
        # Verify one-hot cluster columns exist
        cluster_ohe_cols = [c for c in df.columns if c.startswith('Cluster_')]
        if not cluster_ohe_cols:
            update_status("error", "No One-Hot Columns", 
                          "No Cluster_* columns found. Run clustering with one-hot encoding first.")
            return
        
        update_status("processing", "Cluster Features Found", 
                      f"Using {len(cluster_ohe_cols)} one-hot columns from {cluster_source}: {cluster_ohe_cols}")
        
        # Drop the raw 'Cluster' column (keep one-hot only)
        if 'Cluster' in df.columns:
            df = df.drop(columns=['Cluster'])
            update_status("processing", "Preprocessing", "Dropped raw 'Cluster' column (keeping one-hot encoding).")
        
        def status_cb(msg):
            update_status("processing", "Training", msg)
        
        update_status("processing", "Training Started", 
                      f"Algorithm: {algorithm}, Validation: {validation_type}, Window: {window_size}, Mode: {segment_mode}, Cluster Source: {cluster_source}")
        
        if validation_type == 'rolling':
            results = rolling_window_validation(
                df, target_col='Volatility', algorithm=algorithm, window_size=window_size, segment_mode=segment_mode, status_callback=status_cb
            )
        elif validation_type == 'walk_forward':
            results = walk_forward_validation(
                df, target_col='Volatility', algorithm=algorithm, initial_window=window_size, segment_mode=segment_mode, status_callback=status_cb
            )
        else:
            update_status("error", "Invalid Validation", f"Unknown validation type: {validation_type}")
            return
        
        # Tag results with hybrid info
        results['model'] = f"{results.get('model', algorithm)} (Hybrid + {cluster_source.upper()})"
        results['cluster_source'] = cluster_source
        results['cluster_features'] = cluster_ohe_cols
        
        # Save results to files
        update_status("processing", "Saving Results", "Writing metrics and predictions to files...")
        prefix = f"hybrid_{cluster_source}_{algorithm}_{validation_type}"
        output_dir = DATA_DIR / "ml_results"
        saved_files = save_results_to_file(results, output_dir, prefix)
        
        # Store results in memory
        hybrid_results_store['latest'] = results
        hybrid_results_store['saved_files'] = saved_files
        
        # Build summary
        reg = results.get('regression_metrics', {})
        cls = results.get('classification_metrics', {})
        summary_parts = [
            f"Overall MSE: {reg.get('overall_mse', 0):.6f}",
            f"Overall MAE: {reg.get('overall_mae', 0):.6f}",
            f"Overall R\u00b2: {reg.get('overall_r2', 0):.4f}",
            f"Overall RMSE: {reg.get('overall_rmse', 0):.6f}"
        ]
        if cls:
            summary_parts.append(f"Classification Accuracy: {cls.get('accuracy', 0):.4f}")
            summary_parts.append(f"F1 Macro: {cls.get('f1_macro', 0):.4f}")
        
        summary = " | ".join(summary_parts)
        
        processing_status["output_file"] = saved_files.get('predictions', '')
        processing_status["ml_saved_files"] = saved_files
        
        update_status("completed", "Hybrid-ML Training Complete", 
                      f"Done. {len(results.get('folds', []))} folds. {summary}")
        update_status("completed", "Hybrid-ML Training Complete",
                      f"Files saved to: {output_dir}")
    
    except Exception as e:
        update_status("error", "Hybrid-ML Error", f"Critical error: {str(e)}")
        logger.error(e, exc_info=True)


@app.get("/cluster-files")
async def get_cluster_files():
    """Returns the tracked DBSCAN and K-Means output file paths."""
    return cluster_output_files


@app.post("/train-hybrid")
async def train_hybrid(
    file_path: str = None,
    algorithm: str = "linear_regression",
    validation_type: str = "rolling",
    window_size: int = 6,
    segment_mode: str = "count",
    cluster_source: str = "dbscan",
    background_tasks: BackgroundTasks = None
):
    # If no file_path, auto-pick from cluster_output_files
    if not file_path:
        file_path = cluster_output_files.get(cluster_source, '')
        if not file_path:
            raise HTTPException(status_code=400, 
                detail=f"No {cluster_source.upper()} clustered file found. Run clustering first.")
    
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    
    # Reset status
    processing_status["status"] = "starting"
    processing_status["details"] = []
    processing_status["message"] = "Hybrid-ML Training Queued"
    
    background_tasks.add_task(
        process_hybrid_training_task, file_path_obj, algorithm, validation_type, window_size, segment_mode, cluster_source
    )
    
    return {"message": "Hybrid-ML Training started", "cluster_source": cluster_source, "file": file_path}


@app.get("/hybrid-results")
async def get_hybrid_results():
    if 'latest' not in hybrid_results_store:
        return {"error": "No Hybrid-ML results available yet. Run training first."}
    
    results = hybrid_results_store['latest']
    saved_files = hybrid_results_store.get('saved_files', {})
    
    return {
        "model": results.get('model'),
        "parameters": results.get('parameters'),
        "validation": results.get('validation'),
        "segment_mode": results.get('segment_mode'),
        "cluster_source": results.get('cluster_source'),
        "cluster_features": results.get('cluster_features'),
        "regression_metrics": results.get('regression_metrics'),
        "classification_metrics": {
            k: v for k, v in results.get('classification_metrics', {}).items() 
            if k != 'classification_report'
        },
        "classification_report": results.get('classification_metrics', {}).get('classification_report', ''),
        "folds": results.get('folds'),
        "thresholds": results.get('thresholds'),
        "saved_files": saved_files
    }


@app.get("/hybrid-confusion-matrix")
async def get_hybrid_confusion_matrix():
    if 'latest' not in hybrid_results_store:
        return {"error": "No Hybrid-ML results available yet."}
    
    results = hybrid_results_store['latest']
    cm = results.get('confusion_matrix')
    if not cm:
        return {"error": "No confusion matrix available."}
    
    return cm


@app.get("/status")
async def get_status():
    return processing_status

@app.get("/preview")
async def get_preview():
    if not processing_status["output_file"]:
        return {"error": "No output file generated yet"}
    
    try:
        df = pd.read_csv(processing_status["output_file"])
        # Replace NaN with null for JSON compatibility
        df = df.replace({np.nan: None})
        preview = df.tail(20).to_dict(orient="records") # Show last 20 rows (most recent indicators)
        return {
            "file": processing_status["output_file"],
            "columns": list(df.columns),
            "preview": preview,
            "total_rows": len(df)
        }
    except Exception as e:
         return {"error": f"Could not read output file: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
