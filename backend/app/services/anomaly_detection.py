import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os

UPLOAD_DIR = "uploads"


def run_anomaly_detection(filename: str, contamination: float = 0.05) -> dict:
    """
    Detects anomalies using two approaches:
    1. IQR method — flags values outside 1.5x interquartile range per column
    2. Isolation Forest — detects rows that are globally unusual across all columns
    
    contamination: expected proportion of outliers (default 5%)
    """
    filepath = os.path.join(UPLOAD_DIR, filename)
    df = pd.read_csv(filepath)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found — anomaly detection requires numeric data")

    if len(df) < 10:
        raise ValueError("Not enough rows for anomaly detection (need at least 10)")

    # Fill missing values for analysis
    df_clean = df[numeric_cols].fillna(df[numeric_cols].mean())

    # --- Method 1: IQR per column ---
    column_anomalies = []
    row_flag_counts = pd.Series(0, index=df.index)

    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = df_clean[(df_clean[col] < lower) | (df_clean[col] > upper)]
        outlier_count = len(outliers)
        outlier_pct = round(outlier_count / len(df) * 100, 1)

        if outlier_count > 0:
            column_anomalies.append({
                "column": col,
                "outlier_count": outlier_count,
                "outlier_percent": outlier_pct,
                "lower_bound": round(float(lower), 3),
                "upper_bound": round(float(upper), 3),
                "min_value": round(float(df_clean[col].min()), 3),
                "max_value": round(float(df_clean[col].max()), 3),
                "mean": round(float(df_clean[col].mean()), 3),
                "severity": "high" if outlier_pct > 10 else "medium" if outlier_pct > 5 else "low"
            })
            row_flag_counts[outliers.index] += 1

    # --- Method 2: Isolation Forest (global anomaly detection) ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)

    clf = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100
    )
    anomaly_labels = clf.fit_predict(X_scaled)
    anomaly_scores = clf.decision_function(X_scaled)

    # -1 = anomaly, 1 = normal
    anomaly_mask = anomaly_labels == -1
    anomaly_indices = df.index[anomaly_mask].tolist()
    total_anomalies = int(anomaly_mask.sum())

    # Build detailed anomaly rows — show top 20 most anomalous
    score_series = pd.Series(anomaly_scores, index=df.index)
    top_anomaly_indices = score_series[anomaly_mask].nsmallest(20).index.tolist()

    anomaly_rows = []
    for idx in top_anomaly_indices:
        row = df.iloc[idx]
        row_data = {}
        for col in numeric_cols:
            val = float(row[col]) if pd.notna(row[col]) else None
            row_data[col] = val

        # Explain why this row is anomalous
        reasons = []
        for col in numeric_cols:
            if col in [ca["column"] for ca in column_anomalies]:
                col_info = next(ca for ca in column_anomalies if ca["column"] == col)
                val = row_data.get(col)
                if val is not None:
                    if val < col_info["lower_bound"]:
                        reasons.append(f"{col}={val} is unusually low (normal range: {col_info['lower_bound']} to {col_info['upper_bound']})")
                    elif val > col_info["upper_bound"]:
                        reasons.append(f"{col}={val} is unusually high (normal range: {col_info['lower_bound']} to {col_info['upper_bound']})")

        anomaly_rows.append({
            "row_index": int(idx),
            "anomaly_score": round(float(anomaly_scores[idx]), 4),
            "values": row_data,
            "reasons": reasons[:3] if reasons else ["Globally unusual combination of values"]
        })

    # Distribution of anomaly scores for chart
    score_bins = pd.cut(anomaly_scores, bins=20)
    score_dist = score_bins.value_counts()
    score_dist = score_dist.sort_index()
    score_distribution = [
        {
            "range": str(interval),
            "count": int(count),
            "is_anomaly": float(interval.mid) < clf.offset_
        }
        for interval, count in score_dist.items()
        if pd.notna(interval)
    ]

    return {
        "total_rows": len(df),
        "total_anomalies": total_anomalies,
        "anomaly_percent": round(total_anomalies / len(df) * 100, 1),
        "contamination_used": contamination,
        "numeric_columns_analyzed": numeric_cols,
        "column_anomalies": sorted(column_anomalies, key=lambda x: x["outlier_count"], reverse=True),
        "anomaly_rows": anomaly_rows,
        "score_distribution": score_distribution,
        "summary": f"Found {total_anomalies} anomalous rows ({round(total_anomalies/len(df)*100,1)}%) out of {len(df)} total rows across {len(numeric_cols)} numeric columns."
    }