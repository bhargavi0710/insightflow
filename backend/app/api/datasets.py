from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.dataset import Dataset
import pandas as pd
import json
import os
import aiofiles

router = APIRouter()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    filepath = os.path.join(UPLOAD_DIR, file.filename)
    async with aiofiles.open(filepath, "wb") as f:
        content = await file.read()
        await f.write(content)

    df = pd.read_csv(filepath)

    dataset = Dataset(
        name=file.filename.replace(".csv", ""),
        filename=file.filename,
        row_count=len(df),
        column_names=json.dumps(list(df.columns))
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    return {
        "id": dataset.id,
        "name": dataset.name,
        "row_count": dataset.row_count,
        "columns": json.loads(dataset.column_names)
    }

@router.get("/")
def list_datasets(db: Session = Depends(get_db)):
    datasets = db.query(Dataset).all()
    return [
        {
            "id": d.id,
            "name": d.name,
            "row_count": d.row_count,
            "columns": json.loads(d.column_names),
            "created_at": d.created_at
        }
        for d in datasets
    ]

@router.get("/{dataset_id}/preview")
def preview_dataset(dataset_id: int, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    filepath = os.path.join(UPLOAD_DIR, dataset.filename)
    df = pd.read_csv(filepath)

    return {
        "columns": list(df.columns),
        "rows": df.head(10).to_dict(orient="records"),
        "dtypes": df.dtypes.astype(str).to_dict()
    }