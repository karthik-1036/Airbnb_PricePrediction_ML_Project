# src/data_loader.py

import pandas as pd
from pathlib import Path

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load Airbnb dataset from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    df = pd.read_csv(file_path)
    print(f"✅ Data loaded successfully with shape: {df.shape}")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic cleaning like dropping missing or unnecessary columns.
    
    Args:
        df (pd.DataFrame): Raw dataframe.
    
    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    # Drop irrelevant or high-cardinality columns
    drop_cols = ['id', 'name', 'host_name', 'last_review']
    df = df.drop(columns=drop_cols, errors='ignore')
    
    # Fill missing numerical values
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
    
    # Drop rows with any remaining missing values
    df = df.dropna()

    # Add this at the end of the function
    # Remove listings with unrealistic prices
    df = df[(df['price'] > 0) & (df['price'] < 1000)]  # You can adjust the upper limit


    print(f"✅ Data cleaned. Remaining shape: {df.shape}")
    return df
