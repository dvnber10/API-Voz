from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

from src.Api.routes import router
from src.Core.config import settings

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI(
    title=settings.app_name,
    description="API para procesamiento de audio en tiempo real y batch con Whisper",
    version=settings.version,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "Upgrade", "Connection", "Sec-WebSocket-Key", "Sec-WebSocket-Version"],
)

# Incluir rutas
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "message": "Whisper Audio Processing API",
        "version": settings.version,
        "docs": "/docs",
        "endpoints": {
            "audio_to_text": "POST /api/v1/audio-to-text",
            "realtime_audio": "WebSocket /api/v1/realtime-audio",
            "health": "GET /api/v1/health",
            "stats": "GET /api/v1/stats",
            "model_info": "GET /api/v1/model-info"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )