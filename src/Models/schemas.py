from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class AudioResponse(BaseModel):
    status: str
    text: str
    corrected_text: Optional[str] = None
    filename: str
    message: str
    processing_time: Optional[float] = None
    audio_duration: Optional[float] = None
    confidence: Optional[float] = None

class RealtimeTranscript(BaseModel):
    text: str
    corrected_text: Optional[str] = None
    is_final: bool
    timestamp: float
    confidence: Optional[float] = None

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None

class HealthCheck(BaseModel):
    status: str
    version: str
    services: Dict[str, str]
    device: str
    model_loaded: bool