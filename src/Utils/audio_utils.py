import io
import logging
from pydub import AudioSegment
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class AudioUtils:
    @staticmethod
    def convert_audio_format(audio_file: bytes, original_format: str, target_format: str = "wav") -> io.BytesIO:
        """
        Convierte audio a formato compatible para Whisper
        """
        try:
            # Convertir formato (ej: 'mp3' -> 'mp3')
            format_name = original_format.replace('.', '').lower()
            
            # Cargar audio desde bytes
            audio = AudioSegment.from_file(io.BytesIO(audio_file), format=format_name)
            
            # Convertir a mono y 16kHz (óptimo para Whisper)
            audio = audio.set_channels(1).set_frame_rate(16000)
            
            # Exportar a formato target
            buffer = io.BytesIO()
            audio.export(buffer, format=target_format)
            buffer.seek(0)
            return buffer
            
        except Exception as e:
            logger.error(f"Error converting audio format: {e}")
            raise ValueError(f"Error al convertir audio: {str(e)}")

    @staticmethod
    def validate_audio_format(filename: str, allowed_formats: list) -> Tuple[str, str]:
        """
        Valida el formato del archivo de audio
        """
        file_extension = '.' + filename.split('.')[-1].lower() if '.' in filename else ''
        if not file_extension or file_extension not in allowed_formats:
            raise ValueError(f"Formato no soportado. Formatos permitidos: {', '.join(allowed_formats)}")
        
        return file_extension, file_extension.replace('.', '')
    
    @staticmethod
    def get_audio_duration(audio_file: bytes, format: str) -> float:
        """
        Obtiene la duración del audio en segundos
        """
        try:
            audio = AudioSegment.from_file(io.BytesIO(audio_file), format=format)
            return len(audio) / 1000.0  # Convertir a segundos
        except Exception as e:
            logger.error(f"Error getting audio duration: {e}")
            return 0.0

    @staticmethod
    def convert_to_whisper_compatible(audio_file: bytes, original_format: str) -> bytes:
        """
        Convierte cualquier formato de audio a WAV compatible con Whisper
        """
        try:
            wav_buffer = AudioUtils.convert_audio_format(audio_file, original_format, "wav")
            return wav_buffer.getvalue()
        except Exception as e:
            logger.error(f"Error converting to Whisper compatible: {e}")
            raise ValueError(f"Error al preparar audio para Whisper: {str(e)}")