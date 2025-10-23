import os
import re
import pandas as pd
from typing import List
from .config import PATHS

ID_LIKE = re.compile(r"(?:^|\b)(id|row|index|number|customerid|rownumber)$", re.IGNORECASE)

def find_first_csv(data_dir: str) -> str:
    for name in os.listdir(data_dir):
        if name.lower().endswith(".csv"):
            return os.path.join(data_dir, name)
    raise FileNotFoundError(f"No CSV found in {data_dir}. Place the Kaggle CSV there.")

def load_raw_dataframe() -> pd.DataFrame:
    csv_path = find_first_csv(PATHS.data_dir)
    df = pd.read_csv(csv_path)
    return df

def choose_numeric_features(df: pd.DataFrame, forced_features: List[str] | None = None) -> pd.DataFrame:
    if forced_features:
        # use only forced features that are present
        cols = [c for c in forced_features if c in df.columns]
        if not cols:
            raise ValueError("None of the forced features were found in the dataset.")
        return df[cols].copy()

    # Auto-pick numeric columns with variance and avoid id-like fields
    num_df = df.select_dtypes(include=["number"]).copy()
    keep = []
    for col in num_df.columns:
        if ID_LIKE.search(col):
            continue
        if num_df[col].nunique() <= 1:
            continue
        keep.append(col)
    if not keep:
        raise ValueError("No suitable numeric columns found for clustering.")
    return num_df[keep].copy()

def encode_simple_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    # If there are simple cats like Gender with few categories, one-hot encode.
    # We will only one-hot columns with <= 10 unique values and not too many columns.
    cat_cols = [c for c in df.columns if df[c].dtype == "object" and df[c].nunique() <= 10]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df
