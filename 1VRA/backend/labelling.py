
import pandas as pd
import numpy as np

class RiskLabeller:
    """
    Handles risk labeling logic for Volatility Data.
    """

    @staticmethod
    def relative_thresholds(df: pd.DataFrame, col_name: str = 'Volatility') -> pd.DataFrame:
        """
        Labels data based on relative thresholds (33rd and 66th percentiles).
        
        Binning:
        - Low Risk: < 33rd percentile
        - Medium Risk: 33rd <= x <= 66th percentile
        - High Risk: > 66th percentile
        """
        if col_name not in df.columns:
            raise ValueError(f"Column '{col_name}' not found in DataFrame.")

        # Calculate Percentiles
        p33 = df[col_name].quantile(0.33)
        p66 = df[col_name].quantile(0.66)
        
        def label_row(val):
            if pd.isna(val):
                return None # Or "Unknown"
            if val < p33:
                return "Low Risk"
            elif val <= p66:
                return "Medium Risk"
            else:
                return "High Risk"

        # Apply labels
        df['Risk_Label'] = df[col_name].apply(label_row)
        
        # Add metadata or separate col for thresholds? 
        # For now just the label.
        return df
