import pandas as pd
from src.config import RAW_DATA_PATH


# Load raw METABRIC dataset
def load_raw_metabric_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    return df
