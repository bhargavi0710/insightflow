from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "InsightFlow AI"
    DATABASE_URL: str = "sqlite:///./insightflow.db"
    GROQ_API_KEY: str = ""
    SECRET_KEY: str = "change-this-in-production"

    class Config:
        env_file = ".env"

settings = Settings()