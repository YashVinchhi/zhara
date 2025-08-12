import os
import io
import re
import uuid
import subprocess
import time
from datetime import datetime
from typing import Optional
import torch
import numpy as np
import soundfile as sf
import requests
from scipy.signal import resample
from pydub import AudioSegment
from faster_whisper import WhisperModel
from TTS.api import TTS
from apscheduler.schedulers.background import BackgroundScheduler

# FastAPI and Pydantic
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Import config and constants
from config import (
    MAX_AUDIO_DURATION,
    MAX_TEXT_LENGTH,
    MAX_FILE_AGE,
    MAX_AUDIO_SIZE,
    LLM_TIMEOUT_SECONDS,
    Rhubarb_TIMEOUT_SECONDS
)

# Define base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
STATIC_DIR = os.path.join(BASE_DIR, "static")
AUDIO_DIR = os.path.join(STORAGE_DIR, "audio")
VISEME_DIR = os.path.join(STORAGE_DIR, "visemes")

# Print directory paths for debugging
print("Base directory:", BASE_DIR)
print("Storage directory:", STORAGE_DIR)
print("Static directory:", STATIC_DIR)
print("Audio directory:", AUDIO_DIR)
print("Viseme directory:", VISEME_DIR)

# Create necessary directories at startup
def create_required_directories():
    """Create all required directories for the application"""
    required_dirs = [
        STATIC_DIR,
        STORAGE_DIR,
        AUDIO_DIR,
        VISEME_DIR
    ]
    for directory in required_dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"Ensured directory exists: {directory}")

# Create directories before setting up anything else
create_required_directories()

# --- Models Setup ---
def get_whisper_model():
    """Provides a singleton-like instance of the Whisper model."""
    whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
    return WhisperModel("base", device=whisper_device, compute_type="float16" if whisper_device == "cuda" else "int8")

def get_tts_model():
    """Provides a singleton-like instance of the TTS model."""
    return TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=torch.cuda.is_available())

# Initialize models
whisper_model = get_whisper_model()
tts = get_tts_model()
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Background tasks
from apscheduler.schedulers.background import BackgroundScheduler

# Models and utilities
import torch
from faster_whisper import WhisperModel
from TTS.api import TTS
from pydub import AudioSegment

# Configuration file
import config

# --- Models ---
class ChatRequest(BaseModel):
    text: str
    model: str = "default"  # The selected model from dropdown

# --- Helper Functions ---
def cleanup_old_files():
    """Remove files older than MAX_FILE_AGE hours."""
    current_time = time.time()
    for directory in [AUDIO_DIR, VISEME_DIR]:
        if not os.path.exists(directory):
            continue
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.getmtime(filepath) < current_time - (config.MAX_FILE_AGE * 3600):
                try:
                    os.remove(filepath)
                    print(f"Cleaned up old file: {filepath}")
                except OSError as e:
                    print(f"Error cleaning up {filepath}: {e}")

def decode_audio_file(audio_bytes: bytes):
    """
    Decodes audio bytes from various formats (WAV, OGG, WebM) and returns 
    a mono, 16kHz numpy array.
    """
    try:
        # Try to decode as wav first
        audio, samplerate = sf.read(io.BytesIO(audio_bytes))
        # If stereo, convert to mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        # Resample if not 16kHz
        if samplerate != 16000:
            audio = np.array(resample(audio, int(len(audio) * 16000 / samplerate)), dtype=np.float32)
        return audio, 16000
    except Exception:
        # If that fails, try to decode with pydub
        try:
            audio_seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
            audio_seg = audio_seg.set_channels(1).set_frame_rate(16000)
            audio = np.array(audio_seg.get_array_of_samples()).astype(np.float32) / (2**15)
            return audio, audio_seg.frame_rate
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Audio decode error: {str(e)}")

# --- FastAPI App Setup ---
app = FastAPI()

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories before setting up routes
create_required_directories()

# Set up API routes first (before static files)
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/ask")
async def ask_zhara(data: ChatRequest = Body(...)):
    if len(data.text) > config.MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Text exceeds maximum length of {config.MAX_TEXT_LENGTH} characters"
        )
    
    unique_id = str(uuid.uuid4())
    audio_path = os.path.join(AUDIO_DIR, f"response_{unique_id}.wav")
    viseme_path = os.path.join(VISEME_DIR, f"viseme_{unique_id}.json")
    
    try:
        response_data = requests.post(
            f"{config.OLLAMA_HOST}/api/generate",
            json={
                "model": "qwen2:0.5b",
                "prompt": data.text + "\n\nRespond in clear, simple, and well-punctuated sentences suitable for text-to-speech. Avoid code blocks, markdown, and special characters. Only output plain English sentences.",
                "stream": False
            },
            timeout=config.LLM_TIMEOUT_SECONDS
        ).json()
        reply = response_data.get("response", "Sorry, I couldn't get a response from the model.")
        
        reply = re.sub(r'```.*?```', '', reply, flags=re.DOTALL).strip()
        reply = re.sub(r'`+|[*_#\[\]()>~]', '', reply).strip()
        
        print(f"Zhāra: {reply}")
        
        tts.tts_to_file(text=reply, file_path=audio_path)
        
        viseme_generated = False
        try:
            subprocess.run(
                ["rhubarb", "-f", "json", "-o", viseme_path, audio_path],
                check=True,
                timeout=config.Rhubarb_TIMEOUT_SECONDS
            )
            viseme_generated = True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Rhubarb error: {e}. Viseme generation skipped.")
        
        if not os.path.exists(audio_path):
            raise RuntimeError("Failed to generate audio file.")
        
        viseme_url = f"/viseme/{os.path.basename(viseme_path)}" if viseme_generated and os.path.exists(viseme_path) else None
        
        return {
            "reply": reply,
            "audio_url": f"/audio/{os.path.basename(audio_path)}",
            "viseme_url": viseme_url
        }

    except requests.exceptions.RequestException as e:
        print(f"LLM request error: {e}")
        raise HTTPException(status_code=503, detail="The language model is currently unavailable.")
    except Exception as e:
        # Clean up files on error
        for p in [audio_path, viseme_path]:
            if os.path.exists(p):
                os.remove(p)
        print(f"[Zhāra Error] {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

# Set up static file serving
# First mount specific directories
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")
app.mount("/viseme", StaticFiles(directory=VISEME_DIR), name="viseme")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Finally mount the root directory for serving index.html
app.mount("/", StaticFiles(directory=BASE_DIR, html=True), name="root")

# --- Models Setup ---
def get_whisper_model():
    """Provides a singleton-like instance of the Whisper model."""
    whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
    return WhisperModel("base", device=whisper_device, compute_type="float16" if whisper_device == "cuda" else "int8")

def get_tts_model():
    """Provides a singleton-like instance of the TTS model."""
    return TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=torch.cuda.is_available())

whisper_model = get_whisper_model()
tts = get_tts_model()

# --- Background Scheduler for Cleanup ---
scheduler = BackgroundScheduler()
scheduler.add_job(cleanup_old_files, 'interval', hours=config.MAX_FILE_AGE)
scheduler.start()

# --- API Endpoints ---
@app.post("/stt")
async def stt(file: UploadFile = File(...)):
    file_content = await file.read()
    if not file_content:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    if len(file_content) > config.MAX_AUDIO_SIZE:
        raise HTTPException(status_code=400, detail=f"File size exceeds maximum of {config.MAX_AUDIO_SIZE / 1024 / 1024:.2f}MB.")
    
    try:
        audio_data, _ = decode_audio_file(file_content)
        
        duration = len(audio_data) / 16000
        if duration > config.MAX_AUDIO_DURATION:
            raise HTTPException(
                status_code=400,
                detail=f"Audio duration ({duration:.1f}s) exceeds maximum of {config.MAX_AUDIO_DURATION}s."
            )

        segments, _ = whisper_model.transcribe(audio_data, language="en")
        text = " ".join([seg.text for seg in segments])
        
        return {"text": text}
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"STT processing error: {e}")
        raise HTTPException(status_code=500, detail="Error processing audio for transcription.")

# Create necessary directories at startup
def create_required_directories():
    """Create all required directories for the application"""
    required_dirs = [
        STATIC_DIR,
        STORAGE_DIR,
        AUDIO_DIR,
        VISEME_DIR
    ]
    for directory in required_dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"Ensured directory exists: {directory}")

# --- Main Entry Point ---
if __name__ == "__main__":
    import uvicorn
    
    # Create required directories before starting the server
    create_required_directories()
    
    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=8000)