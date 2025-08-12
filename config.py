import os

# --- Application Constants ---
MAX_AUDIO_DURATION = int(os.getenv('MAX_AUDIO_DURATION', 300))  # Maximum audio duration in seconds
MAX_TEXT_LENGTH = int(os.getenv('MAX_TEXT_LENGTH', 1000))  # Maximum text length for TTS
MAX_FILE_AGE = int(os.getenv('MAX_FILE_AGE', 24))  # Maximum age of files in hours
MAX_AUDIO_SIZE = int(os.getenv('MAX_AUDIO_SIZE', 10 * 1024 * 1024))  # 10MB maximum file size
LLM_TIMEOUT_SECONDS = int(os.getenv('LLM_TIMEOUT_SECONDS', 900))  # Timeout for LLM request
Rhubarb_TIMEOUT_SECONDS = int(os.getenv('Rhubarb_TIMEOUT_SECONDS', 60))  # Timeout for Rhubarb command
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')  # Ollama server host