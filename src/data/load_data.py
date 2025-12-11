# src/data/load_data.py
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # project root
DATA_RAW = ROOT / "data" / "raw" / "telecom_raw.csv"
DATA_PROCESSED = ROOT / "data" / "processed" / "telecom_processed.csv"

def load_raw_csv(path: str = None) -> pd.DataFrame:
    p = Path(path) if path else DATA_RAW
    if not p.exists():
        raise FileNotFoundError(f"Raw data file not found at: {p}")
    return pd.read_csv(p)

def save_processed(df, path: str = None) -> None:
    p = Path(path) if path else DATA_PROCESSED
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    print(f"Saved processed CSV to: {p}")

if __name__ == "__main__":
    df = load_raw_csv()
    print("Loaded raw shape:", df.shape)
