from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    workdir: str = "./_repos"
    codebert_model_path: str = "./models"
    quality_model_path: str = "./models/quality_rf.pkl"
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()
