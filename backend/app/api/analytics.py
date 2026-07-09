from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.dataset import Dataset
from app.services.data_quality import run_quality_check
import pandas as pd
import json
import os

router = APIRouter()
UPLOAD_DIR = "uploads"

def load_dataframe(dataset_id: int, db: Session) -> tuple:
    """Helper: loads a dataset from DB and returns (dataset, dataframe)"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    filepath = os.path.join(UPLOAD_DIR, dataset.filename)
    df = pd.read_csv(filepath)
    return dataset, df


@router.get("/{dataset_id}/summary")
def get_summary(dataset_id: int, db: Session = Depends(get_db)):
    """
    Returns high-level statistics for every numeric column.
    Used to populate the KPI cards and stats table on the dashboard.
    """
    _, df = load_dataframe(dataset_id, db)
    
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    text_cols = df.select_dtypes(include="object").columns.tolist()
    
    summary = {}
    for col in numeric_cols:
        summary[col] = {
            "mean": round(float(df[col].mean()), 2),
            "median": round(float(df[col].median()), 2),
            "min": round(float(df[col].min()), 2),
            "max": round(float(df[col].max()), 2),
            "std": round(float(df[col].std()), 2),
            "nulls": int(df[col].isnull().sum()),
            "null_percent": round(df[col].isnull().sum() / len(df) * 100, 1)
        }
    
    return {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "numeric_columns": numeric_cols,
        "text_columns": text_cols,
        "summary": summary
    }


@router.get("/{dataset_id}/chart-data")
def get_chart_data(dataset_id: int, x_col: str, db: Session = Depends(get_db), y_col: str = None):
    _, df = load_dataframe(dataset_id, db)

    if x_col not in df.columns:
        raise HTTPException(status_code=400, detail="Column not found in dataset")

    if y_col and y_col not in df.columns:
        y_col = None

    # More reliable text detection — check if majority of values are non-numeric strings
    try:
        pd.to_numeric(df[x_col])
        col_is_text = False
    except (ValueError, TypeError):
        col_is_text = True

    # Case 1: text column — value counts (e.g. sentiment, category)
    if col_is_text and not y_col:
        unique_count = df[x_col].nunique()
        if unique_count > 50:
            raise HTTPException(status_code=400, detail=f"'{x_col}' has {unique_count} unique values — too many to chart. Pick a categorical column like 'sentiment'.")
        counts = df[x_col].value_counts().head(20)
        return {
            "x": counts.index.tolist(),
            "y": [int(v) for v in counts.values.tolist()],
            "x_label": x_col,
            "y_label": "Count",
            "is_timeseries": False
        }

    # Case 2: text column with numeric y — group by x, mean of y
    if col_is_text and y_col:
        unique_count = df[x_col].nunique()
        if unique_count > 50:
            raise HTTPException(status_code=400, detail=f"'{x_col}' has too many unique values to group by.")
        grouped = df.groupby(x_col)[y_col].mean().reset_index()
        return {
            "x": grouped[x_col].tolist(),
            "y": grouped[y_col].round(2).tolist(),
            "x_label": x_col,
            "y_label": y_col,
            "is_timeseries": False
        }

    # Case 3: numeric column — bin into ranges
    if not y_col:
        y_col = x_col

    if y_col not in df.columns:
        raise HTTPException(status_code=400, detail="No valid Y column provided")

    try:
        df["x_binned"] = pd.cut(df[x_col].astype(float), bins=15).astype(str)
        grouped = df.groupby("x_binned")[y_col].mean().reset_index()
        return {
            "x": grouped["x_binned"].tolist(),
            "y": grouped[y_col].round(2).tolist(),
            "x_label": x_col,
            "y_label": y_col,
            "is_timeseries": False
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not chart column: {str(e)}")


@router.get("/{dataset_id}/distribution")
def get_distribution(dataset_id: int, col: str, db: Session = Depends(get_db)):
    """
    Returns histogram data for a single numeric column.
    Frontend calls: /api/analytics/1/distribution?col=sales
    Used to show how values are spread (bell curve, skewed, etc.)
    """
    _, df = load_dataframe(dataset_id, db)
    
    if col not in df.columns:
        raise HTTPException(status_code=400, detail="Column not found")
    
    # Create 20 histogram buckets
    counts, bin_edges = pd.cut(df[col].dropna(), bins=20, retbins=True)
    hist_counts = counts.value_counts(sort=False).values.tolist()
    bin_labels = [f"{bin_edges[i]:.1f}–{bin_edges[i+1]:.1f}" for i in range(len(bin_edges)-1)]
    
    return {
        "labels": bin_labels,
        "counts": hist_counts,
        "column": col
    }


@router.get("/{dataset_id}/correlations")
def get_correlations(dataset_id: int, db: Session = Depends(get_db)):
    """
    Returns a correlation matrix for all numeric columns.
    Correlation tells you: when column A goes up, does column B go up too?
    Values range from -1 (opposite) to +1 (perfectly related).
    This is shown as a heatmap in the dashboard.
    """
    _, df = load_dataframe(dataset_id, db)
    
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 numeric columns for correlations")
    
    corr_matrix = numeric_df.corr().round(3)
    
    return {
        "columns": corr_matrix.columns.tolist(),
        "matrix": corr_matrix.values.tolist()
    }

@router.get("/{dataset_id}/quality")
def get_quality_report(dataset_id: int, db: Session = Depends(get_db)):
    """
    Runs a full data quality check before ML training.
    Returns a score, grade, and specific issues to fix.
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    try:
        result = run_quality_check(dataset.filename)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quality check error: {str(e)}")