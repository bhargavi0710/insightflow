import pandas as pd
import numpy as np
import os

UPLOAD_DIR = "uploads"


def run_quality_check(filename: str) -> dict:
    filepath = os.path.join(UPLOAD_DIR, filename)
    df = pd.read_csv(filepath)

    issues = []
    warnings = []
    info = []
    score = 100  # start perfect, deduct for issues

    total_rows = len(df)
    total_cols = len(df.columns)

    # 1 — Missing values
    null_report = []
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        null_pct = round(null_count / total_rows * 100, 1)
        if null_pct > 0:
            null_report.append({
                "column": col,
                "null_count": null_count,
                "null_percent": null_pct,
                "severity": "high" if null_pct > 30 else "medium" if null_pct > 10 else "low"
            })
            if null_pct > 30:
                issues.append(f"'{col}' is {null_pct}% empty — consider dropping this column")
                score -= 10
            elif null_pct > 10:
                warnings.append(f"'{col}' has {null_pct}% missing values — will be filled with column mean")
                score -= 5
            else:
                info.append(f"'{col}' has {null_count} missing values ({null_pct}%) — minor, auto-filled")
                score -= 1

    # 2 — Duplicate rows
    duplicate_count = int(df.duplicated().sum())
    duplicate_pct = round(duplicate_count / total_rows * 100, 1)
    if duplicate_count > 0:
        if duplicate_pct > 10:
            issues.append(f"{duplicate_count} duplicate rows ({duplicate_pct}%) — significant data quality issue")
            score -= 15
        else:
            warnings.append(f"{duplicate_count} duplicate rows ({duplicate_pct}%) — minor, consider removing")
            score -= 5

    # 3 — Constant columns (zero variance)
    constant_cols = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            constant_cols.append(col)
            issues.append(f"'{col}' has only one unique value — useless as a feature, should be dropped")
            score -= 10

    # 4 — High cardinality text columns
    high_cardinality = []
    for col in df.select_dtypes(include="object").columns:
        unique_count = df[col].nunique()
        unique_pct = round(unique_count / total_rows * 100, 1)
        if unique_pct > 80:
            high_cardinality.append({"column": col, "unique_count": unique_count, "unique_percent": unique_pct})
            warnings.append(f"'{col}' has {unique_count} unique values ({unique_pct}%) — likely free text, not usable as a category feature")

    # 5 — Class imbalance (for low-cardinality columns)
    imbalance_report = []
    for col in df.columns:
        if df[col].nunique() <= 10 and df[col].nunique() >= 2:
            value_counts = df[col].value_counts(normalize=True)
            majority_pct = round(float(value_counts.iloc[0]) * 100, 1)
            minority_pct = round(float(value_counts.iloc[-1]) * 100, 1)
            if majority_pct > 90:
                imbalance_report.append({
                    "column": col,
                    "majority_class": str(value_counts.index[0]),
                    "majority_percent": majority_pct,
                    "minority_percent": minority_pct
                })
                warnings.append(f"'{col}' is heavily imbalanced ({majority_pct}% one class) — classifier may be biased toward majority")
                score -= 8

    # 6 — Highly correlated numeric columns
    correlation_issues = []
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] >= 2:
        corr_matrix = numeric_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_pairs = []
        for col in upper.columns:
            for idx in upper.index:
                val = upper.loc[idx, col]
                if pd.notna(val) and val > 0.95:
                    high_corr_pairs.append({
                        "col1": idx,
                        "col2": col,
                        "correlation": round(float(val), 3)
                    })
                    correlation_issues.append({
                        "col1": idx,
                        "col2": col,
                        "correlation": round(float(val), 3)
                    })
        if high_corr_pairs:
            warnings.append(f"{len(high_corr_pairs)} pairs of columns are 95%+ correlated — one of each pair could be dropped without losing information")
            score -= 5

    # 7 — Dataset size check
    if total_rows < 50:
        issues.append(f"Only {total_rows} rows — too small for reliable ML. Results will be unreliable.")
        score -= 20
    elif total_rows < 200:
        warnings.append(f"Only {total_rows} rows — ML results may not generalize well. More data would help.")
        score -= 5

    # 8 — All text dataset
    numeric_count = len(df.select_dtypes(include="number").columns)
    if numeric_count == 0:
        info.append("No numeric columns detected — tabular ML not available. Use Text Classification instead.")

    score = max(0, score)

    if score >= 90:
        grade = "Excellent"
        grade_color = "emerald"
    elif score >= 75:
        grade = "Good"
        grade_color = "blue"
    elif score >= 50:
        grade = "Fair"
        grade_color = "amber"
    else:
        grade = "Poor"
        grade_color = "red"

    return {
        "score": score,
        "grade": grade,
        "grade_color": grade_color,
        "total_rows": total_rows,
        "total_columns": total_cols,
        "issues": issues,
        "warnings": warnings,
        "info": info,
        "null_report": null_report,
        "duplicate_count": duplicate_count,
        "duplicate_percent": duplicate_pct,
        "constant_columns": constant_cols,
        "high_cardinality_columns": high_cardinality,
        "imbalance_report": imbalance_report,
        "correlation_issues": correlation_issues,
        "ml_ready": score >= 50 and total_rows >= 20 and numeric_count > 0
    }