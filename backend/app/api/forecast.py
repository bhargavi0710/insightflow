from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.core.database import get_db
from app.models.dataset import Dataset
from app.services.ml_forecast import run_forecast, run_inference, run_text_classification
import pandas as pd
import os

router = APIRouter()
UPLOAD_DIR = "uploads"


@router.get("/{dataset_id}/forecast")
def forecast_endpoint(dataset_id: int, target_col: str, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    try:
        result = run_forecast(dataset.filename, target_col, dataset_id=dataset_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast error: {str(e)}")


class InferenceRequest(BaseModel):
    target_col: str
    input_values: dict


@router.post("/{dataset_id}/predict")
def predict_endpoint(dataset_id: int, request: InferenceRequest, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    try:
        result = run_inference(dataset_id, request.target_col, request.input_values)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.get("/{dataset_id}/model-status")
def model_status(dataset_id: int, target_col: str):
    import json
    meta_path = os.path.join("models", f"meta_{dataset_id}_{target_col}.json")
    if not os.path.exists(meta_path):
        return {"exists": False}
    with open(meta_path) as f:
        meta = json.load(f)
    return {"exists": True, "meta": meta}


@router.get("/{dataset_id}/text-classify")
def text_classify_endpoint(dataset_id: int, text_col: str, target_col: str, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    try:
        result = run_text_classification(dataset.filename, target_col, text_col)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text classification error: {str(e)}")


@router.get("/{dataset_id}/forecast-targets")
def get_forecast_targets(dataset_id: int, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    filepath = os.path.join(UPLOAD_DIR, dataset.filename)
    df = pd.read_csv(filepath)
    targets = []
    for col in df.columns:
        if df[col].dtype == object:
            if df[col].nunique() <= 20:
                targets.append({"column": col, "type": "classification"})
        else:
            if df[col].nunique() <= 10:
                targets.append({"column": col, "type": "classification"})
            else:
                targets.append({"column": col, "type": "regression"})
    return {"targets": targets}
