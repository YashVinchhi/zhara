from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import requests
import os
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from TTS.api import TTS
from pydub import AudioSegment
import io
import re
from scipy.signal import resample

app = FastAPI()

# --- Models ---
class InputText(BaseModel):
    text: str

# --- STT Model ---
import torch
# Use GPU if available for Whisper
whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = WhisperModel("base", device=whisper_device, compute_type="float16" if whisper_device=="cuda" else "int8")

# --- TTS Model ---
# Use GPU if available for TTS
tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=torch.cuda.is_available())

# --- API Endpoints ---
@app.post("/ask")
def ask_zhara(data: InputText):
    user_prompt = data.text
    # Query Ollama
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "qwen2.5-coder:14b", "prompt": user_prompt + "\n\nRespond in clear, simple, and well-punctuated sentences suitable for text-to-speech. Avoid code blocks, markdown, and special characters. Only output plain English sentences.", "stream": False},
            timeout=900
        )
        response.raise_for_status()
        result = response.json()
        reply = result.get("response", "Sorry, I couldn't get a response from the model.")
        # Clean up reply for TTS: remove markdown/code blocks if any
        reply = re.sub(r'```.*?```', '', reply, flags=re.DOTALL)  # Remove code blocks
        reply = re.sub(r'`+', '', reply)  # Remove inline code
        reply = re.sub(r'[*_#\[\]()>~]', '', reply)  # Remove markdown special chars
        reply = reply.strip()
    except Exception as e:
        reply = f"Error: {str(e)}"
        print(f"[Zhāra Error] {reply}")
        return {"reply": reply}
    print(f"Zhāra: {reply}")
    # TTS
    audio_path = "response.wav"
    tts.tts_to_file(text=reply, file_path=audio_path)
    return {"reply": reply, "audio_url": "/audio"}

@app.post("/stt")
async def stt(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    try:
        # Try to decode as wav first
        audio, samplerate = sf.read(io.BytesIO(audio_bytes))
        # If stereo, convert to mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        # Resample to 16000 Hz if needed
        if samplerate != 16000:
            num_samples = int(len(audio) * 16000 / samplerate)
            audio = resample(audio, num_samples)
            samplerate = 16000
    except Exception:
        # If that fails, try to decode as webm/ogg/opus using pydub
        try:
            audio_seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
            audio_seg = audio_seg.set_channels(1).set_frame_rate(16000)
            audio = np.array(audio_seg.get_array_of_samples()).astype(np.float32) / (2**15)
            samplerate = audio_seg.frame_rate
        except Exception as e:
            return {"text": f"Audio decode error: {e}"}
    # Ensure float32 and correct shape for whisper
    if not isinstance(audio, np.ndarray):
        audio = np.array(audio, dtype=np.float32)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    segments, _ = whisper_model.transcribe(audio, language="en")
    text = " ".join([seg.text for seg in segments])
    return {"text": text}

@app.get("/audio")
def get_audio():
    return FileResponse("response.wav", media_type="audio/wav")

# Serve static files
static_dir = os.path.dirname(os.path.abspath(__file__))
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
