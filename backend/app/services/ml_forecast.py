import pandas as pd
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import shap
import joblib
import os
import json

UPLOAD_DIR = "uploads"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def get_model_path(dataset_id: int, target_col: str) -> str:
    return os.path.join(MODELS_DIR, f"model_{dataset_id}_{target_col}.joblib")


def get_meta_path(dataset_id: int, target_col: str) -> str:
    return os.path.join(MODELS_DIR, f"meta_{dataset_id}_{target_col}.json")


def run_forecast(filename: str, target_col: str, dataset_id: int = None):
    filepath = os.path.join(UPLOAD_DIR, filename)
    df = pd.read_csv(filepath)

    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in dataset")

    is_classification = False

    if df[target_col].dtype == object:
        unique_vals = df[target_col].nunique()
        if unique_vals > 20:
            raise ValueError(f"'{target_col}' has {unique_vals} unique text values — too many categories to classify")
        is_classification = True
    else:
        non_null_vals = df[target_col].dropna()
        is_whole_numbers = (non_null_vals == non_null_vals.round()).all()
        if non_null_vals.nunique() <= 10 and is_whole_numbers:
            is_classification = True

    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c != target_col]
    text_cols = [c for c in df.select_dtypes(include="object").columns if c != target_col]
    usable_text_cols = [c for c in text_cols if df[c].nunique() <= 30]
    skipped_text_cols = [c for c in text_cols if df[c].nunique() > 30]

    if len(numeric_cols) == 0 and len(usable_text_cols) == 0:
        raise ValueError("No usable feature columns found")

    work_df = df[numeric_cols].copy()
    work_df["__target__"] = df[target_col].values
    work_df = work_df.dropna(subset=["__target__"])

    if numeric_cols:
        work_df[numeric_cols] = work_df[numeric_cols].fillna(work_df[numeric_cols].mean())

    encoded_feature_cols = list(numeric_cols)
    if usable_text_cols:
        text_data = df.loc[work_df.index, usable_text_cols].fillna("missing")
        dummies = pd.get_dummies(text_data, prefix=usable_text_cols, drop_first=False)
        work_df = pd.concat([work_df, dummies], axis=1)
        encoded_feature_cols += dummies.columns.tolist()

    feature_cols = encoded_feature_cols
    X = work_df[feature_cols]
    y_raw = work_df["__target__"]

    if len(X) < 20:
        raise ValueError("Not enough rows to train a model (need at least 20)")

    label_encoder = None
    class_names = None

    if is_classification:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)
        class_names = label_encoder.classes_.tolist()
    else:
        y = y_raw

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if is_classification else None
    )

    # Store column means for filling missing values during inference
    col_means = {col: float(work_df[col].mean()) for col in numeric_cols}

    if is_classification:
        model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss"
        )
    else:
        model = XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cv_scoring = "accuracy" if is_classification else "r2"
    cv_scores = cross_val_score(model, X, y, cv=5, scoring=cv_scoring)
    cv_mean = float(cv_scores.mean())
    cv_std = float(cv_scores.std())

    importance = model.feature_importances_
    feature_importance = sorted(zip(feature_cols, importance.tolist()), key=lambda x: x[1], reverse=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    if is_classification and isinstance(shap_values, list):
        mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    elif is_classification and len(np.array(shap_values).shape) == 3:
        mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

    shap_importance = sorted(zip(feature_cols, mean_abs_shap.tolist()), key=lambda x: x[1], reverse=True)

    # Save model and metadata for inference
    if dataset_id is not None:
        model_data = {
            "model": model,
            "label_encoder": label_encoder,
            "feature_cols": feature_cols,
            "numeric_cols": numeric_cols,
            "usable_text_cols": usable_text_cols,
            "col_means": col_means,
        }
        joblib.dump(model_data, get_model_path(dataset_id, target_col))

        mae_val = None
        if not is_classification:
            from sklearn.metrics import mean_absolute_error as _mae
            _y_pred_all = model.predict(X)
            mae_val = round(float(_mae(y, _y_pred_all)), 3)

        meta = {
            "is_classification": is_classification,
            "target_col": target_col,
            "feature_cols": feature_cols,
            "numeric_cols": numeric_cols,
            "class_names": [str(c) for c in class_names] if class_names else None,
            "col_means": col_means,
            "mae": mae_val,
        }
        with open(get_meta_path(dataset_id, target_col), "w") as f:
            json.dump(meta, f)

    result = {
        "is_classification": is_classification,
        "target_column": target_col,
        "feature_columns": feature_cols,
        "numeric_input_columns": numeric_cols,
        "categorical_features_used": usable_text_cols,
        "skipped_high_cardinality_columns": skipped_text_cols,
        "rows_used": len(X),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "cv_mean": round(cv_mean, 3),
        "cv_std": round(cv_std, 3),
        "feature_importance": [{"feature": f, "importance": round(i, 4)} for f, i in feature_importance],
        "shap_importance": [{"feature": f, "impact": round(i, 4)} for f, i in shap_importance],
        "model_saved": dataset_id is not None,
    }

    if is_classification:
        accuracy = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="weighted"))
        result["accuracy"] = round(accuracy, 3)
        result["f1_score"] = round(f1, 3)
        result["class_names"] = [str(c) for c in class_names]

        comparison = []
        for actual, pred in list(zip(y_test.tolist(), y_pred.tolist()))[:50]:
            comparison.append({
                "actual": str(class_names[int(actual)]),
                "predicted": str(class_names[int(pred)]),
                "correct": int(actual) == int(pred)
            })
        result["predictions_sample"] = comparison

        pred_counts = pd.Series([str(class_names[int(p)]) for p in y_pred]).value_counts()
        result["prediction_distribution"] = [{"class": k, "count": int(v)} for k, v in pred_counts.items()]
    else:
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        result["mae"] = round(mae, 3)
        result["r2_score"] = round(r2, 3)

        comparison = []
        for actual, pred in list(zip(y_test.tolist(), y_pred.tolist()))[:50]:
            comparison.append({"actual": round(float(actual), 2), "predicted": round(float(pred), 2)})
        result["actual_vs_predicted"] = comparison

    return result


def run_inference(dataset_id: int, target_col: str, input_values: dict) -> dict:
    model_path = get_model_path(dataset_id, target_col)
    meta_path = get_meta_path(dataset_id, target_col)

    if not os.path.exists(model_path):
        raise ValueError("No trained model found. Please train the model first.")

    model_data = joblib.load(model_path)
    with open(meta_path) as f:
        meta = json.load(f)

    model = model_data["model"]
    label_encoder = model_data["label_encoder"]
    feature_cols = model_data["feature_cols"]
    numeric_cols = model_data["numeric_cols"]
    col_means = model_data["col_means"]

    # Build input row
    row = {}
    for col in numeric_cols:
        if col in input_values and input_values[col] != "" and input_values[col] is not None:
            try:
                row[col] = float(input_values[col])
            except (ValueError, TypeError):
                row[col] = col_means.get(col, 0.0)
        else:
            row[col] = col_means.get(col, 0.0)

    # Fill any encoded dummy columns with 0
    for col in feature_cols:
        if col not in row:
            row[col] = 0.0

    input_df = pd.DataFrame([row])[feature_cols]
    prediction_raw = model.predict(input_df)[0]

    if meta["is_classification"]:
        if label_encoder:
            prediction = str(label_encoder.inverse_transform([int(prediction_raw)])[0])
        else:
            class_names = meta.get("class_names", [])
            prediction = str(class_names[int(prediction_raw)]) if class_names else str(prediction_raw)

        proba = None
        if hasattr(model, "predict_proba"):
            proba_raw = model.predict_proba(input_df)[0]
            class_names = meta.get("class_names", [])
            proba = {str(class_names[i]): round(float(p) * 100, 1) for i, p in enumerate(proba_raw)}

        return {
            "prediction": prediction,
            "type": "classification",
            "probabilities": proba
        }
    else:
        import json as _json
        with open(meta_path) as f:
            meta = _json.load(f)
        mae = meta.get("mae", None)
        prediction_val = round(float(prediction_raw), 3)
        result = {
            "prediction": prediction_val,
            "type": "regression",
            "probabilities": None
        }
        if mae:
            result["confidence_interval"] = {
                "low": round(prediction_val - mae, 3),
                "high": round(prediction_val + mae, 3),
                "mae": mae,
                "explanation": f"Based on test MAE of {mae}, the realistic range is {round(prediction_val - mae, 3)} to {round(prediction_val + mae, 3)}"
            }
        return result


def run_text_classification(filename: str, target_col: str, text_col: str):
    filepath = os.path.join(UPLOAD_DIR, filename)
    df = pd.read_csv(filepath)

    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in dataset")
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in dataset")

    work_df = df[[text_col, target_col]].dropna()

    if len(work_df) < 50:
        raise ValueError("Not enough rows to train a text classifier (need at least 50)")

    if len(work_df) > 5000:
        work_df = work_df.sample(5000, random_state=42)

    unique_classes = work_df[target_col].nunique()
    if unique_classes > 10:
        raise ValueError(f"Target column has {unique_classes} unique classes — too many for classification")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(work_df[target_col])
    class_names = label_encoder.classes_.tolist()

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        work_df[text_col], y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=3000, stop_words="english", ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="weighted"))

    feature_names = vectorizer.get_feature_names_out()
    top_words_by_class = {}

    if len(class_names) == 2:
        coefs = model.coef_[0]
        top_positive_idx = coefs.argsort()[-15:][::-1]
        top_negative_idx = coefs.argsort()[:15]
        top_words_by_class[class_names[1]] = [
            {"word": feature_names[i], "weight": round(float(coefs[i]), 4)} for i in top_positive_idx
        ]
        top_words_by_class[class_names[0]] = [
            {"word": feature_names[i], "weight": round(float(-coefs[i]), 4)} for i in top_negative_idx
        ]
    else:
        for idx, class_name in enumerate(class_names):
            coefs = model.coef_[idx]
            top_idx = coefs.argsort()[-15:][::-1]
            top_words_by_class[class_name] = [
                {"word": feature_names[i], "weight": round(float(coefs[i]), 4)} for i in top_idx
            ]

    sample_results = []
    for text, actual, pred in list(zip(X_test_text.tolist(), y_test.tolist(), y_pred.tolist()))[:20]:
        preview = text[:150] + "..." if len(text) > 150 else text
        sample_results.append({
            "text_preview": preview,
            "actual": class_names[int(actual)],
            "predicted": class_names[int(pred)],
            "correct": int(actual) == int(pred)
        })

    pred_counts = pd.Series([class_names[int(p)] for p in y_pred]).value_counts()
    prediction_distribution = [{"class": k, "count": int(v)} for k, v in pred_counts.items()]

    return {
        "mode": "text_classification",
        "text_column": text_col,
        "target_column": target_col,
        "rows_used": len(work_df),
        "train_rows": len(X_train_text),
        "test_rows": len(X_test_text),
        "accuracy": round(accuracy, 3),
        "f1_score": round(f1, 3),
        "class_names": class_names,
        "top_words_by_class": top_words_by_class,
        "predictions_sample": sample_results,
        "prediction_distribution": prediction_distribution,
        "vocabulary_size": len(feature_names)
    }
