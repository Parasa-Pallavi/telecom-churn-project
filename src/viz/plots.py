# src/viz/plots.py
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.data.load_data import load_raw_csv
from src.features.build_features import add_features
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def save_fig(fig, name):
    p = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    print("Saved:", p)

def churn_by_contract(df):
    fig, ax = plt.subplots(figsize=(7,4))
    if "Contract" in df.columns and "Churn" in df.columns:
        sns.countplot(data=df, x="Contract", hue="Churn", ax=ax)
        ax.set_title("Churn by Contract")
        save_fig(fig, "churn_by_contract.png")
    else:
        print("Columns Contract/Churn not present")

def monthly_box(df):
    fig, ax = plt.subplots(figsize=(6,4))
    if "MonthlyCharges" in df.columns and "Churn" in df.columns:
        sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=ax)
        ax.set_title("MonthlyCharges by Churn")
        save_fig(fig, "monthly_box.png")
    else:
        print("MonthlyCharges/Churn missing")

def make_all():
    raw = load_raw_csv()
    df = add_features(raw)
    churn_by_contract(raw)
    monthly_box(raw)
    # additional plots can be added similarly

if __name__ == "__main__":
    make_all()
