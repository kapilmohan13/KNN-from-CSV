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
from pathlib import Path
from indicators import TechnicalIndicators
from labelling import RiskLabeller

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

# Global status tracker (simple in-memory for this single-user app)
processing_status = {
    "status": "idle",
    "message": "",
    "details": [],
    "last_file": "",
    "output_file": ""
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
