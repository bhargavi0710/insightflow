from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.core.database import get_db
from app.models.dataset import Dataset
from app.services.ai_chat import chat_with_data
import json

router = APIRouter()

# This defines what the frontend must send in the request body
class ChatRequest(BaseModel):
    question: str
    chat_history: list = []  # list of previous messages for context

@router.post("/{dataset_id}/chat")
def chat_endpoint(
    dataset_id: int,
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """
    POST /api/chat/{dataset_id}/chat
    Body: { "question": "what is the max sales?", "chat_history": [] }
    Returns: { "answer": "The max sales value is 45,230..." }
    """
    # First check the dataset exists
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        answer = chat_with_data(
            filename=dataset.filename,
            user_question=request.question,
            chat_history=request.chat_history
        )
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI error: {str(e)}")