import pandas as pd

def load_data(filepath):
    """Loads the dataset from a given file path."""
    return pd.read_excel(filepath)