# file_utils.py
import pandas as pd
import os

def load_csv(path):
    """
    Load a CSV file into a pandas DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)

def save_csv(df, path, index=False):
    """
    Save a pandas DataFrame to a CSV file.
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    df.to_csv(path, index=index)
    print(f"Saved CSV to: {path}")
