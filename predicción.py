from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
import librosa
import numpy as np

# Ruta al modelo entrenado y al archivo de audio
MODEL_PATH = "./whisper-tiny-es-filtrado-final"
AUDIO_PATH = "audioprueba.ogg"  # Cambia esto a la ruta de tu archivo de audio

# 1. Cargar el procesador y el modelo
processor = WhisperProcessor.from_pretrained(MODEL_PATH)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)

# Configurar para CPU
model.to("cpu")
model.eval()  # Modo evaluación

# 2. Cargar y preprocesar el audio
def load_and_preprocess_audio(audio_path, sample_rate=16000):
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    # Normalizar audio
    audio = librosa.util.normalize(audio)
    # Convertir a espectrogramas Log-Mel
    input_features = processor(audio, sampling_rate=sample_rate, return_tensors="pt").input_features
    return input_features

# 3. Hacer la predicción
input_features = load_and_preprocess_audio(AUDIO_PATH)
with torch.no_grad():
    predicted_ids = model.generate(input_features, language="es", task="transcribe")
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

# 4. Opcional: Post-procesamiento con vocabulario
try:
    with open("/home/dvn/Documentos/Desarrollo/Python/PGC/top_2000_spanish.txt", "r", encoding="utf-8") as f:
        vocab = set([line.strip().lower() for line in f.readlines() if line.strip()])
    words = transcription.lower().split()
    corrected_words = [word if word in vocab else min(vocab, key=lambda x: len(set(word) ^ set(x))) for word in words]
    corrected_transcription = " ".join(corrected_words)
except FileNotFoundError:
    corrected_transcription = transcription

# 5. Imprimir resultados
print(f"📜 Transcripción original: {transcription}")
if corrected_transcription != transcription:
    print(f"📜 Transcripción corregida: {corrected_transcription}")