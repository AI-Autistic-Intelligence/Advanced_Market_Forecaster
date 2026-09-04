from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_path: str = "models/best_model.pt"
    scaler_path: str = "models/scaler.joblib"
    seq_length: int = 24
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"

settings = Settings()
