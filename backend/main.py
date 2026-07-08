from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.database import engine, Base
from app.api import datasets, analytics, chat, forecast

Base.metadata.create_all(bind=engine)

app = FastAPI(title="InsightFlow AI", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["analytics"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(forecast.router, prefix="/api/forecast", tags=["forecast"])

@app.get("/")
def root():
    return {"message": "InsightFlow AI v3 is running"}
