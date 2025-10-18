# src/Services/realtime_service.py
import asyncio
import logging
import time
import json
from typing import Dict, Any
from src.Services.whisper_service import WhisperService
from src.Core.config import settings

logger = logging.getLogger(__name__)

class RealtimeAudioService:
    def __init__(self):
        self.whisper_service = WhisperService(
            model_path=settings.whisper_model_path,
            vocab_path=settings.vocab_path,
            device="auto"
        )
        self.active_connections: Dict[str, Any] = {}
        self.sample_rate = settings.realtime_sample_rate
        self.chunk_duration = settings.chunk_duration
        self.chunk_size = int(self.sample_rate * 2 * self.chunk_duration)  # 16-bit = 2 bytes
        
        # Cargar modelo
        self._load_model()

    def _load_model(self):
        """Carga el modelo Whisper para tiempo real"""
        try:
            self.whisper_service.load_model()
            logger.info("✅ Modelo Whisper cargado para tiempo real")
        except Exception as e:
            logger.error(f"❌ Error cargando modelo Whisper para tiempo real: {e}")

    async def handle_realtime_connection(self, websocket, client_id: str):
        """
        Maneja una conexión WebSocket para procesamiento en tiempo real con Whisper
        """
        self.active_connections[client_id] = websocket
        logger.info(f"🔌 Cliente {client_id} conectado para procesamiento en tiempo real")
        
        try:
            # Enviar mensaje de bienvenida inmediatamente
            welcome_msg = {
                "type": "connection_established",
                "message": "Conexión establecida para procesamiento en tiempo real con Whisper",
                "client_id": client_id,
                "sample_rate": self.sample_rate,
                "chunk_duration": self.chunk_duration,
                "device": self.whisper_service.device
            }
            await websocket.send_text(json.dumps(welcome_msg))
            logger.info(f"📤 Mensaje de bienvenida enviado a {client_id}")
            
            buffer = b""
            chunk_count = 0
            
            while True:
                # Recibir datos de audio
                message = await websocket.receive()
                
                if message.get("type") == "websocket.disconnect":
                    break
                
                if message.get("type") == "websocket.receive":
                    if "bytes" in message:
                        audio_data = message["bytes"]
                        buffer += audio_data
                        chunk_count += 1
                        
                        logger.debug(f"📥 Audio recibido de {client_id}: {len(audio_data)} bytes, buffer: {len(buffer)} bytes, chunk: #{chunk_count}")
                        
                        # Procesar cuando tenemos suficiente datos
                        while len(buffer) >= self.chunk_size:
                            chunk = buffer[:self.chunk_size]
                            buffer = buffer[self.chunk_size:]
                            
                            logger.info(f"🎯 Procesando chunk de audio #{chunk_count} ({len(chunk)} bytes) para {client_id}")
                            
                            try:
                                # Procesar el chunk con Whisper usando el método de tiempo real
                                result = self.whisper_service.transcribe_realtime_chunk(
                                    chunk, self.sample_rate
                                )
                                
                                logger.info(f"📝 Resultado transcripción para {client_id}: '{result['text']}'")
                                
                                # FILTRO DE TRANSCRIPCIONES
                                if result["text"].strip() and not result["text"].startswith("[Error:"):
                                    # Filtrar transcripciones muy cortas o repetitivas
                                    if len(result["text"]) >= 3 and not self._is_repetitive_text(result["text"]):
                                        response = {
                                            "type": "transcription",
                                            "text": result["text"],
                                            "corrected_text": result["corrected_text"],
                                            "is_final": result["is_final"],
                                            "has_correction": result["has_correction"],
                                            "timestamp": time.time()
                                        }
                                        await websocket.send_text(json.dumps(response))
                                        logger.info(f"📤 Transcripción enviada a {client_id}: {result['text']}")
                                    else:
                                        logger.info(f"🚫 Transcripción filtrada (muy corta/repetitiva): {result['text']}")
                                else:
                                    logger.info(f"📝 Transcripción vacía o con error para {client_id}, no enviando")
                                    
                            except Exception as e:
                                logger.error(f"❌ Error procesando audio para {client_id}: {e}")
                                error_response = {
                                    "type": "error",
                                    "message": f"Error procesando audio: {str(e)}",
                                    "timestamp": time.time()
                                }
                                await websocket.send_text(json.dumps(error_response))
                    
                    elif "text" in message:
                        # Manejar mensajes de control
                        try:
                            control_msg = json.loads(message["text"])
                            logger.info(f"📨 Mensaje de control de {client_id}: {control_msg}")
                            
                            if control_msg.get("type") == "ping":
                                await websocket.send_text(json.dumps({"type": "pong"}))
                            elif control_msg.get("type") == "get_status":
                                await websocket.send_text(json.dumps({
                                    "type": "status",
                                    "model_loaded": self.whisper_service.is_loaded,
                                    "device": self.whisper_service.device,
                                    "buffer_size": len(buffer),
                                    "active_connections": len(self.active_connections),
                                    "chunk_count": chunk_count
                                }))
                        except json.JSONDecodeError:
                            logger.warning(f"❌ Mensaje de control de {client_id} no es JSON válido: {message['text']}")
                        
        except Exception as e:
            logger.error(f"❌ Error en conexión tiempo real {client_id}: {e}")
        finally:
            self.active_connections.pop(client_id, None)
            logger.info(f"🔌 Cliente {client_id} desconectado")

    def _is_repetitive_text(self, text: str, threshold: float = 0.7) -> bool:
        """
        Verifica si el texto es demasiado repetitivo
        """
        words = text.split()
        if len(words) <= 2:
            return False
        
        unique_words = len(set(words))
        repetition_ratio = unique_words / len(words)
        
        logger.debug(f"🔍 Análisis de repetición: {len(words)} palabras, {unique_words} únicas, ratio: {repetition_ratio:.2f}")
        
        return repetition_ratio < threshold

    async def broadcast_to_all(self, message: str):
        """
        Envía un mensaje a todas las conexiones activas
        """
        disconnected = []
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"Error enviando a cliente {client_id}: {e}")
                disconnected.append(client_id)
        
        # Limpiar conexiones desconectadas
        for client_id in disconnected:
            self.active_connections.pop(client_id, None)

    def get_connection_count(self) -> int:
        """Retorna el número de conexiones activas"""
        return len(self.active_connections)

    def get_model_status(self) -> Dict[str, Any]:
        """Retorna el estado del modelo"""
        return {
            "model_loaded": self.whisper_service.is_loaded,
            "model_path": self.whisper_service.model_path,
            "vocab_loaded": self.whisper_service.vocab is not None,
            "device": self.whisper_service.device
        }