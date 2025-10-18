# src/Services/whisper_service.py
import torch
import librosa
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any, List
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import io
import time
import wave
import whisper 
import tempfile
import os
import re

logger = logging.getLogger(__name__)

class WhisperService:
    def __init__(self, model_path: str, vocab_path: Optional[str] = None, device: str = "auto"):
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.device = self._detect_device() if device == "auto" else device
        self.processor = None
        self.model = None
        self.vocab = None
        self.is_loaded = False
        
        # Modelo Whisper de OpenAI para correcciones (Modelo secundario de alta calidad)
        self.whisper_model = None
        self.whisper_loaded = False
        
        # Configuraciones para detección de problemas
        self.silence_threshold = 0.008
        self.min_chunk_size = 16000  # Mínimo 1 segundo
    
    def _detect_device(self):
        """Detecta automáticamente el mejor dispositivo disponible"""
        try:
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        except:
            return "cpu"
        
    def load_model(self):
        """Carga el modelo Whisper y el vocabulario (Tu modelo local)"""
        try:
            logger.info(f"Cargando modelo Whisper desde: {self.model_path}")
            logger.info(f"Usando dispositivo: {self.device}")
            
            self.processor = WhisperProcessor.from_pretrained(self.model_path)
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_path)
            
            self.model.to(self.device)
            self.model.eval()
            
            # Cargar vocabulario si existe
            if self.vocab_path:
                try:
                    with open(self.vocab_path, "r", encoding="utf-8") as f:
                        self.vocab = set([line.strip().lower() for line in f.readlines() if line.strip()])
                    logger.info(f"Vocabulario cargado: {len(self.vocab)} palabras")
                except FileNotFoundError:
                    logger.warning(f"Archivo de vocabulario no encontrado: {self.vocab_path}")
                    self.vocab = None
            
            self.is_loaded = True
            logger.info(f"Modelo Whisper cargado exitosamente en dispositivo: {self.device}")
            
        except Exception as e:
            logger.error(f"Error cargando modelo Whisper: {e}")
            raise e
    
    def load_whisper_model(self):
        """Carga el modelo Whisper de OpenAI (secundario) para correcciones"""
        try:
            logger.info("🔄 Cargando modelo Whisper de OpenAI para correcciones...")
            # Usar el modelo base que es rápido y eficiente
            self.whisper_model = whisper.load_model("base")
            self.whisper_loaded = True
            logger.info("✅ Modelo Whisper de OpenAI cargado para correcciones")
        except Exception as e:
            logger.error(f"❌ Error cargando modelo Whisper de OpenAI: {e}")
            self.whisper_loaded = False
    
    def _call_whisper_correction(self, audio_data: bytes) -> Optional[str]:
        """
        Usa el modelo Whisper de OpenAI para corregir transcripciones problemáticas
        """
        if not self.whisper_loaded:
            self.load_whisper_model()
            if not self.whisper_loaded:
                return None
        
        # --- Lógica de Corrección con Whisper (Modelo Secundario) ---
        try:
            # Crear archivo temporal WAV
            temp_audio_path = None
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                # Convertir a WAV si es PCM crudo
                if len(audio_data) < 100 or audio_data[:4] != b'RIFF':
                    wav_data = self.pcm_to_wav_bytes(audio_data)
                    temp_audio.write(wav_data)
                else:
                    temp_audio.write(audio_data)
                temp_audio_path = temp_audio.name
            
            # Usar Whisper de OpenAI para transcripción
            result = self.whisper_model.transcribe(
                temp_audio_path,
                language="es",
                task="transcribe",
                fp16=False,
                no_speech_threshold=0.6,
                logprob_threshold=-1.0,
                compression_ratio_threshold=2.4
            )
            
            # Limpiar archivo temporal
            os.unlink(temp_audio_path)
            
            transcription = result["text"].strip()
            
            if transcription and len(transcription) > 1:
                logger.info(f"✅ Corrección Whisper exitosa: '{transcription}'")
                return transcription
            else:
                logger.warning("❌ Whisper no produjo transcripción válida")
                return None
            
        except Exception as e:
            logger.error(f"❌ Error en Whisper corrección: {e}")
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
            return None
    
    # --- Funciones de Detección de Alucinaciones/Problemas (Real-Time) ---

    def _should_use_whisper_correction(self, local_transcription: str, audio_rms: float) -> bool:
        """Decide si es necesario usar Whisper para corregir la transcripción local."""
        if not local_transcription or len(local_transcription.strip()) < 2:
            return False

        conditions = [
            self._contains_problematic_patterns(local_transcription),
            self._sounds_like_hallucination(local_transcription),
            self._has_excessive_repetition(local_transcription),
            self._has_nonsense_words(local_transcription),
            self._has_repetitive_patterns(local_transcription),
        ]

        should_correct = any(conditions)

        if should_correct:
            logger.warning(f"🚨 ACTIVANDO CORRECCIÓN WHISPER - Razón: '{local_transcription}'")

        return should_correct
    
    def _contains_problematic_patterns(self, text: str) -> bool:
        """Detecta patrones problemáticos comunes en las alucinaciones."""
        problematic_patterns = [
            r'\b(?:civilizado|civilizativa|civilización)\b',
            r'\b(?:liberaron|liberó|liberar)\b.*\b(?:años|año)\b',
            r'\b(?:últimos|primeros)\b.*\b(?:años|año)\b',
            r'\b(?:permite|permitir)\b.*\b(?:terminar|finalizar)\b',
            r'\b(?:palabra)\b.*\b(?:palabra)\b',
            r'\b(?:deceso|decesos)\b',
            r'\b(?:se permite|se liberó|se liberaron)\b',
            r'\b(?:chotó|choto)\b',
        ]
        
        text_lower = text.lower()
        for pattern in problematic_patterns:
            if re.search(pattern, text_lower):
                logger.warning(f"🚫 Patrón problemático detectado: {pattern}")
                return True
        return False
    
    def _has_excessive_repetition(self, text: str) -> bool:
        """Detecta repeticiones excesivas que indican alucinación."""
        words = text.lower().split()
        if len(words) < 3:
            return False
        
        word_counts = {}
        for word in words:
            if len(word) > 2:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        max_repetitions = max(word_counts.values()) if word_counts else 0
        if max_repetitions > len(words) * 0.4:
            logger.warning(f"🚫 Repetición excesiva detectada: {max_repetitions}/{len(words)}")
            return True
        return False
    
    def _has_repetitive_patterns(self, text: str) -> bool:
        """Detecta patrones repetitivos como 'y luego', 'y entonces'."""
        repetitive_patterns = [
            r'(y luego\s*){2,}',
            r'(y entonces\s*){2,}',
            r'(el deceso\s*){2,}',
        ]
        
        for pattern in repetitive_patterns:
            if re.search(pattern, text.lower()):
                logger.warning(f"🚫 Patrón repetitivo detectado: {pattern}")
                return True
        return False
    
    def _has_nonsense_words(self, text: str) -> bool:
        """Detecta palabras que probablemente son alucinaciones."""
        nonsense_words = {
            'chotó', 'choto', 'deceso', 'civilizativa', 'permite', 'liberaron', 
            'liberó', 'terminar', 'finalizar', 'palabra', 'decesos'
        }
        
        words = text.lower().split()
        if not words:
            return False
            
        nonsense_count = sum(1 for word in words if word in nonsense_words)
        
        if nonsense_count > len(words) * 0.25:
            logger.warning(f"🚫 Palabras sin sentido detectadas: {nonsense_count}/{len(words)}")
            return True
        return False
    
    def _detect_silence(self, audio_array: np.ndarray, threshold: float = 0.008) -> bool:
        """Detecta si el audio es principalmente silencio."""
        if len(audio_array) == 0:
            return True
            
        rms = np.sqrt(np.mean(audio_array**2))
        return rms < threshold
    
    def _sounds_like_hallucination(self, text: str) -> bool:
        """Detector más agresivo de alucinaciones basado en ejemplos reales."""
        if not text:
            return False

        text_lower = text.lower()
        hallucination_indicators = [
            r'\bplenamento\b', r'\bcompilión\b', r'\búnica\b.*\bcompilión\b',
            r'\brompió\b.*\bplenamento\b', r'\bser única\b.*\bser en el\b',
            r'\bpara ser en el\b', r'\bde usted$', r'\bdel? luz\b',
            r'\bno hay que ser\b.*\bpara ser\b.*\bde usted\b',
        ]

        for pattern in hallucination_indicators:
            if re.search(pattern, text_lower):
                logger.warning(f"🚨 Patrón de alucinación detectado: '{pattern}' en '{text}'")
                return True
        return False

    def _get_common_spanish_words(self):
        """Lista de palabras comunes en español para detectar alucinaciones."""
        return {
            'que', 'de', 'el', 'la', 'en', 'y', 'a', 'los', 'las', 'un', 'una', 'unos', 'unas',
            'con', 'del', 'al', 'lo', 'te', 'se', 'su', 'tu', 'mi', 'es', 'son', 'soy', 'eres',
            'está', 'están', 'estoy', 'estás', 'hay', 'tiene', 'tienen', 'tengo', 'tienes',
            'para', 'por', 'como', 'más', 'pero', 'o', 'si', 'no', 'sí', 'ya', 'cuando',
            'muy', 'sin', 'sobre', 'entre', 'hasta', 'desde', 'hacia', 'ante', 'bajo'
        }
    
    # --- Utilidades de Audio ---

    def pcm_to_wav_bytes(self, pcm_data: bytes, sample_rate: int = 16000) -> bytes:
        """
        Convierte datos PCM crudos a formato WAV
        """
        try:
            wav_buffer = io.BytesIO()
            
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm_data)
            
            wav_buffer.seek(0)
            return wav_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Error convirtiendo PCM a WAV: {e}")
            raise ValueError(f"Error al convertir audio PCM: {str(e)}")
    
    def load_and_preprocess_audio(self, audio_data: bytes, sample_rate: int = 16000) -> torch.Tensor:
        """
        Preprocesa audio (WAV, MP3, etc.) para tu modelo local.
        """
        try:
            # Si es PCM crudo o formato no estándar, lo convierte a WAV en memoria
            if len(audio_data) > 44 and audio_data[:4] == b'RIFF':
                audio_array, sr = librosa.load(io.BytesIO(audio_data), sr=sample_rate, mono=True)
            else:
                wav_data = self.pcm_to_wav_bytes(audio_data, sample_rate)
                audio_array, sr = librosa.load(io.BytesIO(wav_data), sr=sample_rate, mono=True)
            
            audio_array = librosa.util.normalize(audio_array)
            
            input_features = self.processor(
                audio_array, 
                sampling_rate=sr, 
                return_tensors="pt"
            ).input_features
            
            return input_features.to(self.device)
            
        except Exception as e:
            logger.error(f"❌ Error en preprocesamiento de audio: {e}")
            raise ValueError(f"Error al preprocesar audio: {str(e)}")
    
    def load_and_preprocess_audio_with_rms(self, audio_data: bytes, sample_rate: int = 16000) -> Tuple[Optional[torch.Tensor], float]:
        """
        Preprocesa audio y retorna features + RMS para detección de silencio (Solo Real-Time).
        """
        try:
            # Si es PCM crudo o formato no estándar, lo convierte a WAV en memoria
            if len(audio_data) > 44 and audio_data[:4] == b'RIFF':
                audio_array, sr = librosa.load(io.BytesIO(audio_data), sr=sample_rate, mono=True)
            else:
                wav_data = self.pcm_to_wav_bytes(audio_data, sample_rate)
                audio_array, sr = librosa.load(io.BytesIO(wav_data), sr=sample_rate, mono=True)

            rms = np.sqrt(np.mean(audio_array**2))
            
            if self._detect_silence(audio_array, self.silence_threshold):
                return None, rms
            
            audio_array = librosa.util.normalize(audio_array)
            
            input_features = self.processor(
                audio_array, 
                sampling_rate=sr, 
                return_tensors="pt"
            ).input_features
            
            return input_features.to(self.device), rms
            
        except Exception as e:
            logger.error(f"❌ Error en preprocesamiento de audio: {e}")
            return None, 0.0

    # --- NUEVA Lógica de Segmentación para Archivos Largos ---
    
    def _split_audio_for_transcription(self, audio_data: bytes, segment_length_sec: int = 30) -> List[bytes]:
        """
        Carga el audio y lo divide en segmentos más pequeños para evitar OOM (Out Of Memory).
        Retorna una lista de datos WAV para cada segmento.
        """
        logger.info(f"🔄 Dividiendo audio en segmentos de {segment_length_sec} segundos...")
        
        try:
            audio_array, sr = librosa.load(io.BytesIO(audio_data), sr=16000, mono=True)
        except Exception as e:
            logger.error(f"Error cargando audio para segmentación: {e}")
            raise ValueError("Formato de audio no compatible o corrupto.")

        segment_samples = segment_length_sec * 16000
        total_samples = len(audio_array)
        segments_data = []
        
        # Iterar y dividir
        for i in range(0, total_samples, segment_samples):
            segment_array = audio_array[i:i + segment_samples]
            
            # Convertir el segmento de float32 a PCM 16-bit
            int16_array = np.int16(segment_array * 32767)
            pcm_data = int16_array.tobytes()
            
            # Convertir el segmento PCM a bytes WAV (para re-uso de load_and_preprocess_audio)
            wav_segment_bytes = self.pcm_to_wav_bytes(pcm_data, sample_rate=16000)
            segments_data.append(wav_segment_bytes)

        logger.info(f"✅ Audio dividido en {len(segments_data)} segmentos.")
        return segments_data

    # --- Transcripción de Archivos (Modificada con Segmentación) ---

    def transcribe_audio(self, audio_data: bytes, language: str = "es") -> Dict[str, Any]:
        """
        Transcribe archivos de audio, ahora con segmentación para soportar audios largos.
        """
        if not self.is_loaded:
            self.load_model()
        
        start_time = time.time()
        
        # 1. Dividir el audio en segmentos
        try:
            segments = self._split_audio_for_transcription(audio_data, segment_length_sec=30)
        except Exception as e:
            logger.error(f"Error en segmentación del audio: {e}")
            raise ValueError(f"Error en segmentación del audio: {str(e)}")

        full_original_text = []
        full_corrected_text = []
        has_any_correction = False
        
        # 2. Procesar cada segmento
        for i, segment_bytes in enumerate(segments):
            logger.info(f"🎙️ Transcribiendo segmento {i + 1}/{len(segments)}...")
            
            try:
                # Preprocesar segmento
                input_features = self.load_and_preprocess_audio(segment_bytes)
                
                # Generar transcripción (con tu modelo local)
                with torch.no_grad():
                    predicted_ids = self.model.generate(
                        input_features, 
                        language=language, 
                        task="transcribe", 
                        num_beams=5,
                        return_timestamps=False
                    )
                
                # Decodificar resultado
                transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                transcription = self._clean_transcription(transcription)

                # Aplicar corrección de vocabulario
                corrected_transcription = self._apply_vocabulary_correction(transcription)
                
                full_original_text.append(transcription)
                full_corrected_text.append(corrected_transcription)
                
                if corrected_transcription != transcription:
                    has_any_correction = True
                    
            except Exception as e:
                logger.error(f"Error transcribiendo segmento {i + 1}: {e}. Segmento omitido.")
                full_original_text.append("[ERROR_SEGMENTO]")
                full_corrected_text.append("[ERROR_SEGMENTO]")

        # 3. Concatenar resultados
        final_original_text = " ".join(full_original_text).strip()
        final_corrected_text = " ".join(full_corrected_text).strip()
        
        processing_time = time.time() - start_time
        
        return {
            "original_text": final_original_text,
            "corrected_text": final_corrected_text,
            "processing_time": processing_time,
            "has_correction": has_any_correction
        }

    # --- Transcripción en Tiempo Real (Modificada con Doble Modelo) ---
    
    def transcribe_realtime_chunk(self, audio_chunk: bytes, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Procesa un chunk de audio en tiempo real con corrección Whisper (Doble Modelo).
        """
        if not self.is_loaded:
            self.load_model()
        
        start_time = time.time()
        
        try:
            if len(audio_chunk) < self.min_chunk_size:
                return {"text": "", "corrected_text": "", "is_final": False, "has_correction": False, "source": "local"}
            
            input_features, audio_rms = self.load_and_preprocess_audio_with_rms(audio_chunk, sample_rate)
            
            if input_features is None:
                return {"text": "", "corrected_text": "", "is_final": False, "has_correction": False, "source": "local"}
            
            generation_config = {
                "language": "es", "task": "transcribe", "num_beams": 1, "max_length": 64,
                "temperature": 0.0, "do_sample": False, "return_timestamps": False, "suppress_tokens": [-1],
            }
            
            # 1. Transcripción Local (Modelo Rápido)
            with torch.no_grad():
                predicted_ids = self.model.generate(input_features, **generation_config)
            
            local_transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            local_transcription = self._clean_transcription(local_transcription)
            
            # 2. Verificación y Corrección (Modelo de Alta Calidad)
            should_correct = False

            if not self._is_valid_transcription(local_transcription) or len(local_transcription.strip()) < 3:
                should_correct = True
            
            if not should_correct and self._should_use_whisper_correction(local_transcription, audio_rms):
                should_correct = True
            
            final_transcription = local_transcription
            used_whisper_correction = False
            
            if should_correct:
                whisper_transcription = self._call_whisper_correction(audio_chunk)
                
                if whisper_transcription and whisper_transcription.strip():
                    final_transcription = whisper_transcription
                    used_whisper_correction = True
                elif not self._is_valid_transcription(local_transcription) or len(local_transcription.strip()) < 3:
                    final_transcription = "" # Usar vacío si ambos fallan

            # 3. Limpieza final y retorno
            corrected_transcription = self._apply_vocabulary_correction(final_transcription)
            
            if not final_transcription.strip():
                return {"text": "", "corrected_text": "", "is_final": False, "has_correction": False, "source": "local"}

            processing_time = time.time() - start_time
            
            return {
                "text": final_transcription,
                "corrected_text": corrected_transcription,
                "is_final": len(final_transcription.strip()) > 0,
                "has_correction": corrected_transcription != final_transcription,
                "source": "whisper" if used_whisper_correction else "local",
                "processing_time": processing_time,
                "local_original": local_transcription if used_whisper_correction else None
            }
            
        except Exception as e:
            logger.error(f"❌ Error en transcripción en tiempo real: {e}")
            return {"text": "", "corrected_text": "", "is_final": False, "has_correction": False, "source": "local"}
    
    # --- Funciones de Limpieza y Corrección ---

    def _clean_transcription(self, text: str) -> str:
        """Limpia la transcripción de tokens especiales y contenido no deseado."""
        if not text: return ""
        
        special_tokens = ["[música]", "[MUSIC]", "[music]", "[Música]", "[ruido]", "[NOISE]", "[noise]", "[Ruido]", "[aplausos]", "[APPLAUSE]", "[applause]", "[Aplausos]", "[silbidos]", "[WHISTLING]", "[whistling]", "[Silbidos]"]
        
        cleaned_text = text
        for token in special_tokens:
            cleaned_text = cleaned_text.replace(token, "")
        
        cleaned_text = re.sub(r'\[.*?\]', '', cleaned_text)
        cleaned_text = re.sub(r'(.)\1{3,}', r'\1', cleaned_text)
        cleaned_text = ' '.join(cleaned_text.split())
        
        return cleaned_text.strip()
    
    def _is_valid_transcription(self, text: str) -> bool:
        """Verifica si la transcripción es válida."""
        if not text or len(text.strip()) < 2: return False
        
        import string
        letters_and_spaces = sum(1 for c in text if c.isalpha() or c in string.whitespace)
        special_chars = sum(1 for c in text if c in '!¡¿?*[]')
        
        if special_chars > len(text) * 0.3: return False
        
        words = text.split()
        if len(words) > 3:
            unique_words = len(set(words))
            if unique_words / len(words) < 0.3: return False
        
        return True
    
    def _apply_vocabulary_correction(self, text: str) -> str:
        """Aplica corrección de vocabulario usando el archivo de palabras."""
        if not self.vocab or not text.strip(): return text
        
        try:
            words = text.lower().split()
            corrected_words = []
            
            for word in words:
                clean_word = ''.join(c for c in word if c.isalnum())
                
                if clean_word and clean_word not in self.vocab:
                    best_match = min(self.vocab, key=lambda x: self._word_similarity(clean_word, x))
                    corrected_words.append(best_match)
                else:
                    corrected_words.append(word)
            
            return " ".join(corrected_words)
            
        except Exception as e:
            logger.warning(f"Error en corrección de vocabulario: {e}")
            return text
    
    def _word_similarity(self, word1: str, word2: str) -> int:
        """Calcula similitud entre palabras (simple métrica)."""
        if not word1 or not word2:
            return max(len(word1), len(word2))
        
        return abs(len(word1) - len(word2)) + len(set(word1) ^ set(word2))

    def update_silence_threshold(self, threshold: float):
        """Permite ajustar el umbral de silencio dinámicamente."""
        self.silence_threshold = threshold

    def update_min_chunk_size(self, min_size: int):
        """Permite ajustar el tamaño mínimo del chunk."""
        self.min_chunk_size = min_size