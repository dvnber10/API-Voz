from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uuid
import logging

from src.Services.audio_service import AudioProcessingService
from src.Services.realtime_service import RealtimeAudioService
from src.Models.schemas import AudioResponse, HealthCheck, RealtimeTranscript
from src.Api.dependences import validate_audio_file
from src.Core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()
audio_service = AudioProcessingService()
realtime_service = RealtimeAudioService()

@router.post("/audio-to-text", response_model=AudioResponse)
async def convert_audio_to_text(file: UploadFile = File(...)):
    """
    Endpoint para procesar archivos de audio completos usando Whisper
    """
    try:
        # Validar archivo
        await validate_audio_file(file)
        
        # Leer contenido del archivo
        contents = await file.read()
        
        # Verificar tamaño del archivo
        if len(contents) > settings.max_file_size:
            raise HTTPException(413, f"Archivo demasiado grande. Máximo: {settings.max_file_size // (1024*1024)}MB")
        
        # Procesar audio con Whisper
        result = await audio_service.process_audio_file(contents, file.filename)
        
        return JSONResponse(content=result, status_code=200)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en endpoint audio-to-text: {e}")
        raise HTTPException(500, f"Error interno del servidor: {str(e)}")

@router.websocket("/realtime-audio")
async def realtime_audio_websocket(websocket: WebSocket):
    """
    WebSocket para procesamiento de audio en tiempo real con Whisper
    """
    client_id = str(uuid.uuid4())
    
    await websocket.accept()
    logger.info(f"Conexión WebSocket aceptada para cliente {client_id}")
    
    try:
        await realtime_service.handle_realtime_connection(websocket, client_id)
    except WebSocketDisconnect:
        logger.info(f"WebSocket desconectado para cliente {client_id}")
    except Exception as e:
        logger.error(f"Error WebSocket para cliente {client_id}: {e}")
        try:
            await websocket.close(code=1011)
        except:
            pass

@router.get("/health", response_model=HealthCheck)
async def health_check():
    """
    Endpoint de health check
    """
    model_status = realtime_service.get_model_status()
    
    return HealthCheck(
        status="healthy",
        version=settings.version,
        services={
            "audio_processing": "active",
            "realtime_processing": "active",
            "whisper_model": "loaded" if model_status["model_loaded"] else "error"
        },
        device=settings.device,
        model_loaded=model_status["model_loaded"]
    )

@router.get("/stats")
async def get_stats():
    """
    Endpoint para obtener estadísticas del servicio
    """
    model_status = realtime_service.get_model_status()
    
    return {
        "active_realtime_connections": realtime_service.get_connection_count(),
        "model_status": model_status,
        "services": {
            "audio_processing": "active",
            "realtime_processing": "active",
            "whisper_model": "loaded" if model_status["model_loaded"] else "error"
        }
    }

@router.get("/model-info")
async def get_model_info():
    """
    Endpoint para obtener información del modelo Whisper
    """
    model_status = realtime_service.get_model_status()
    
    return {
        "model_path": model_status["model_path"],
        "model_loaded": model_status["model_loaded"],
        "vocab_loaded": model_status["vocab_loaded"],
        "device": model_status["device"],
        "allowed_formats": settings.allowed_audio_formats,
        "max_file_size_mb": settings.max_file_size // (1024 * 1024)
    }