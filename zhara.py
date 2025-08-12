from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse, JSONResponse
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
import uuid
import subprocess
import time
from datetime import datetime
from typing import Optional

# Define directories for storing audio and viseme files
STORAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "storage")
AUDIO_DIR = os.path.join(STORAGE_DIR, "audio")
VISEME_DIR = os.path.join(STORAGE_DIR, "visemes")

print(f"Audio directory: {AUDIO_DIR}")
print(f"Viseme directory: {VISEME_DIR}")

# Constants
MAX_AUDIO_DURATION = 300  # Maximum audio duration in seconds
MAX_TEXT_LENGTH = 1000    # Maximum text length for TTS
MAX_FILE_AGE = 24        # Maximum age of files in hours
MAX_AUDIO_SIZE = 10 * 1024 * 1024  # 10MB maximum file size

# Create directories if they don't exist and ensure they're accessible
def ensure_directory(path):
    os.makedirs(path, exist_ok=True)
    # Create a test file to verify write permissions
    test_file = os.path.join(path, "test.txt")
    try:
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except Exception as e:
        print(f"Error: Cannot write to directory {path}: {e}")
        raise

ensure_directory(AUDIO_DIR)
ensure_directory(VISEME_DIR)

def cleanup_old_files():
    """Remove files older than MAX_FILE_AGE hours"""
    current_time = time.time()
    for directory in [AUDIO_DIR, VISEME_DIR]:
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.getmtime(filepath) < current_time - (MAX_FILE_AGE * 3600):
                try:
                    os.remove(filepath)
                except OSError as e:
                    print(f"Error cleaning up {filepath}: {e}")

app = FastAPI()

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

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
async def ask_zhara(data: InputText):
    # Clean up old files
    cleanup_old_files()
    
    if len(data.text) > MAX_TEXT_LENGTH:
        raise HTTPException(status_code=400, detail=f"Text exceeds maximum length of {MAX_TEXT_LENGTH} characters")

    user_prompt = data.text
    # Generate a unique filename for this session
    unique_id = str(uuid.uuid4())
    audio_filename = f"response_{unique_id}.wav"
    viseme_filename = f"viseme_{unique_id}.json"
    audio_path = os.path.join(AUDIO_DIR, audio_filename)
    viseme_path = os.path.join(VISEME_DIR, viseme_filename)
    
    print(f"Generating audio file: {audio_path}")  # Debug output

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
        error_msg = f"Error communicating with language model: {str(e)}"
        print(f"[Zhāra Error] {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

    print(f"Zhāra: {reply}")
    
    # TTS
    try:
        tts.tts_to_file(text=reply, file_path=audio_path)
    except Exception as e:
        error_msg = f"Text-to-speech generation failed: {str(e)}"
        print(f"[Zhāra Error] {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

    # Rhubarb Viseme Generation
    try:
        # Run Rhubarb CLI command
        subprocess.run(
            ["rhubarb", "-f", "json", "-o", viseme_path, audio_path],
            check=True,
            timeout=60
        )
        viseme_generated = True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Rhubarb error: {e}. Viseme generation skipped.")
        viseme_generated = False
    
    # Debug: Check if files exist
    # Verify files exist and are accessible
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=500, detail="Failed to generate audio file")
    
    if viseme_generated and not os.path.exists(viseme_path):
        raise HTTPException(status_code=500, detail="Failed to generate viseme file")
    
    # Return URLs with correct format
    base_url = "http://localhost:8000"  # You can make this configurable later
    response_data = {
        "reply": reply,
        "audio_url": f"{base_url}/audio/{audio_filename}",
        "viseme_url": f"{base_url}/viseme/{viseme_filename}" if viseme_generated else None
    }
    print(f"Returning response: {response_data}")  # Debug output
    return response_data

@app.post("/stt")
async def stt(file: UploadFile = File(...)):
    # Check file size
    file_size = 0
    file_content = bytearray()
    
    async for chunk in file.stream():
        file_size += len(chunk)
        if file_size > MAX_AUDIO_SIZE:
            raise HTTPException(status_code=400, detail=f"File size exceeds maximum of {MAX_AUDIO_SIZE/1024/1024}MB")
        file_content.extend(chunk)
    
    audio_bytes = bytes(file_content)
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    try:
        # Try to decode as wav first
        audio, samplerate = sf.read(io.BytesIO(audio_bytes))
        # If stereo, convert to mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
    except Exception:
        # If that fails, try to decode as webm/ogg/opus using pydub
        try:
            audio_seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
            audio_seg = audio_seg.set_channels(1).set_frame_rate(16000)
            audio = np.array(audio_seg.get_array_of_samples()).astype(np.float32) / (2**15)
            samplerate = audio_seg.frame_rate
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Audio decode error: {str(e)}")
    
    # Check audio duration
    duration = len(audio) / samplerate
    if duration > MAX_AUDIO_DURATION:
        raise HTTPException(
            status_code=400, 
            detail=f"Audio duration ({duration:.1f}s) exceeds maximum of {MAX_AUDIO_DURATION}s"
        )
    # Ensure float32 and correct shape for whisper
    if not isinstance(audio, np.ndarray):
        audio = np.array(audio, dtype=np.float32)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    segments, _ = whisper_model.transcribe(audio, language="en")
    text = " ".join([seg.text for seg in segments])
    return {"text": text}

# File serving endpoints are now handled by StaticFiles mounts

# Serve static files
static_dir = os.path.dirname(os.path.abspath(__file__))
# Mount audio and viseme directories first (higher priority)
app.mount("/audio", StaticFiles(directory=AUDIO_DIR, html=False), name="audio")
app.mount("/viseme", StaticFiles(directory=VISEME_DIR, html=False), name="viseme")

# Mount the main static files last (lowest priority)
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
