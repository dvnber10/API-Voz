# test_websocket.py
import asyncio
import websockets
import json
import pyaudio
import time

async def test_websocket():
    uri = "ws://127.0.0.1:8000/api/v1/realtime-audio"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Conectado al WebSocket")
            
            # Recibir mensaje de conexión
            welcome = await websocket.recv()
            print(f"📨 Mensaje de bienvenida: {welcome}")
            
            # Configurar audio
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=8000
            )
            
            print("🎤 Grabando audio... (presiona Ctrl+C para detener)")
            
            try:
                while True:
                    # Leer audio del micrófono
                    audio_data = stream.read(8000, exception_on_overflow=False)
                    
                    # Enviar audio al WebSocket
                    await websocket.send(audio_data)
                    
                    # Intentar recibir transcripción
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                        data = json.loads(response)
                        if data.get('type') == 'transcription':
                            print(f"📝 Transcripción: {data.get('corrected_text')}")
                    except asyncio.TimeoutError:
                        # No hay transcripción disponible aún
                        pass
                        
            except KeyboardInterrupt:
                print("\n🛑 Deteniendo prueba...")
            finally:
                stream.stop_stream()
                stream.close()
                p.terminate()
                
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())