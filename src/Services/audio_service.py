import time
import logging
from typing import Dict, Any
from src.Services.whisper_service import WhisperService
from src.Utils.audio_utils import AudioUtils
from src.Core.config import settings

logger = logging.getLogger(__name__)

class AudioProcessingService:
    def __init__(self):
        self.whisper_service = WhisperService(
            model_path=settings.whisper_model_path,
            vocab_path=settings.vocab_path,
            device="auto"  # Detección automática
        )
        self.audio_utils = AudioUtils()
        
        # Cargar modelo al inicializar
        self._load_model()

    def _load_model(self):
        """Carga el modelo Whisper al inicializar el servicio"""
        try:
            self.whisper_service.load_model()
            logger.info("Modelo Whisper cargado exitosamente")
        except Exception as e:
            logger.error(f"Error cargando modelo Whisper: {e}")
            raise e

    async def process_audio_file(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Procesa un archivo de audio completo usando Whisper
        """
        start_time = time.time()
        
        try:
            # Validar formato
            file_extension, audio_format = self.audio_utils.validate_audio_format(
                filename, settings.allowed_audio_formats
            )
            
            # Obtener duración del audio
            duration = self.audio_utils.get_audio_duration(file_content, audio_format)
            logger.info(f"Procesando archivo de audio: {filename}, duración: {duration:.2f}s")
            
            # Realizar transcripción con Whisper
            result = self.whisper_service.transcribe_audio(file_content)
            
            processing_time = time.time() - start_time
            
            return {
                "status": "success",
                "text": result["original_text"],
                "corrected_text": result["corrected_text"],
                "filename": filename,
                "message": "Audio procesado exitosamente con Whisper",
                "processing_time": round(processing_time, 2),
                "audio_duration": round(duration, 2),
                "has_correction": result["has_correction"]
            }
            
        except Exception as e:
            logger.error(f"Error procesando archivo de audio: {e}")
            raise e