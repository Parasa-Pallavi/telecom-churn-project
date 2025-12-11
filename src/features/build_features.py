# src/features/build_features.py
import pandas as pd
from pathlib import Path
from typing import Any

def tenure_band(tenure: Any) -> str:
    try:
        t = float(tenure)
    except Exception:
        return "unknown"
    if t <= 2:
        return "0-2"
    if t <= 6:
        return "3-6"
    if t <= 12:
        return "7-12"
    if t <= 24:
        return "13-24"
    return "25+"

def clean_contract(x: Any) -> str:
    if pd.isna(x):
        return "unknown"
    return str(x).strip().lower().replace(" ", "_")

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # tenure numeric
    if "tenure" in df.columns:
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0).astype(float)
    else:
        df["tenure"] = 0.0
    df["tenure_band"] = df["tenure"].apply(tenure_band)

    # contract
    if "Contract" in df.columns:
        df["contract_type"] = df["Contract"].apply(clean_contract)
    else:
        df["contract_type"] = "unknown"

    # MonthlyCharges numeric & high flag
    if "MonthlyCharges" in df.columns:
        df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0.0)
        median = df["MonthlyCharges"].median() if len(df) else 0.0
        df["high_monthly"] = (df["MonthlyCharges"] > median).astype(int)
    else:
        df["MonthlyCharges"] = 0.0
        df["high_monthly"] = 0

    # TotalCharges numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    else:
        df["TotalCharges"] = 0.0

    # churn flag 0/1
    if "Churn" in df.columns:
        df["churn_flag"] = df["Churn"].astype(str).str.strip().str.lower().isin(["yes", "y", "1", "true"]).astype(int)
    else:
        df["churn_flag"] = 0

    return df

if __name__ == "__main__":
    # quick local test if run directly
    sample = pd.DataFrame({
        "tenure": [1, 5, 30],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "MonthlyCharges": ["20", "80", "120"],
        "TotalCharges": ["20", "400", "3600"],
        "Churn": ["Yes", "No", "No"]
    })
    print(add_features(sample))
