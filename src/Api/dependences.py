from fastapi import HTTPException, UploadFile
from src.Core.config import settings
import os

async def validate_audio_file(file: UploadFile):
    """Valida el archivo de audio"""
    if not file.content_type.startswith('audio/'):
        raise HTTPException(400, "El archivo debe ser de audio")
    
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in settings.allowed_audio_formats:
        raise HTTPException(
            400, 
            f"Formato no soportado. Formatos permitidos: {', '.join(settings.allowed_audio_formats)}"
        )
    
    return file