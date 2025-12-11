# src/models/train.py
import joblib
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from src.data.load_data import load_raw_csv, save_processed
from src.features.build_features import add_features

ROOT = Path(__file__).resolve().parents[2]
MODEL_OUT = ROOT / "models" / "churn_pipeline.joblib"

def prepare_data(df: pd.DataFrame):
    df2 = add_features(df)
    features = ["MonthlyCharges", "TotalCharges", "tenure", "contract_type", "tenure_band", "high_monthly"]
    X = df2[features].copy()
    y = df2["churn_flag"].astype(int)
    return X, y

def make_onehot_encoder():
    """
    Return a OneHotEncoder instance that works across scikit-learn versions:
    some versions accept sparse_output=False, others accept sparse=False.
    """
    try:
        # preferred for newer scikit-learn
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # fallback for older scikit-learn
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return encoder

def build_pipeline(num_feats, cat_feats):
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_feats),
        ("cat", make_onehot_encoder(), cat_feats)
    ])
    pipeline = ImbPipeline(steps=[
        ("preproc", pre),
        ("smote", SMOTE(random_state=42)),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    return pipeline

def train_and_save():
    print("Loading data...")
    df = load_raw_csv()
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    num_feats = ["MonthlyCharges", "TotalCharges", "tenure"]
    cat_feats = ["contract_type", "tenure_band"]
    pipe = build_pipeline(num_feats, cat_feats)
    print("Fitting pipeline (this may take a short while)...")
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:,1]
    print(classification_report(y_test, preds))
    try:
        print("ROC AUC:", roc_auc_score(y_test, proba))
    except Exception:
        pass
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_OUT)
    print("Saved model to:", MODEL_OUT)
    return pipe

if __name__ == "__main__":
    train_and_save()
