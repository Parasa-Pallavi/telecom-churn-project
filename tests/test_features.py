# tests/test_features.py
from src.features.build_features import add_features
import pandas as pd

def test_add_features_basic():
    df = pd.DataFrame({
        "tenure": [1, 10, 30],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "MonthlyCharges": [20.0, 70.0, 100.0],
        "TotalCharges": ["20","700","3000"],
        "Churn": ["Yes", "No", "No"]
    })
    out = add_features(df)
    assert "tenure_band" in out.columns
    assert "contract_type" in out.columns
    assert out["churn_flag"].tolist() == [1, 0, 0]
