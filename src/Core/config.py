from pydantic_settings import BaseSettings
from typing import Optional, List
import os

class Settings(BaseSettings):
    app_name: str = "Whisper Audio Processing API"
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    
    # Audio configuration
    max_file_size: int = 200 * 1024 * 1024  # 50MB
    allowed_audio_formats: List[str] = [".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".m4a"]
    sample_rate: int = 16000
    channels: int = 1
    
    # Whisper Model configuration
    whisper_model_path: str = "./whisper-tiny-es-filtrado-final"
    vocab_path: Optional[str] = "/home/dvn/Documentos/Desarrollo/Python/PGC/top_2000_spanish.txt"
    device: str = "cpu"  # Valor por defecto, se actualizará después
    
    # Realtime configuration
    chunk_duration: float = 8
    realtime_sample_rate: int = 16000
    
    class Config:
        env_file = ".env"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ahora que torch puede estar disponible, actualizamos el device
        self._update_device()

    def _update_device(self):
        """Actualiza el dispositivo basado en la disponibilidad de CUDA"""
        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            self.device = "cpu"

settings = Settings()