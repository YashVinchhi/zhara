# Zhara

Zhara is a simple API server for speech-to-text (STT) and text-to-speech (TTS) using FastAPI, Whisper, and Coqui TTS.

## Features
- Speech-to-text (STT) endpoint using Whisper
- Text-to-speech (TTS) endpoint using Coqui TTS
- API endpoint for interacting with Ollama (LLM)
- Audio file serving

## Requirements
- Python 3.8+
- All dependencies listed in `requirements.txt`

## Installation
1. Clone this repository or copy the files to your project directory.
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

## Usage
1. Start the API server:
   ```powershell
   python zhara.py
   ```
2. The server will run on `http://localhost:8000`.

### Endpoints
- `POST /ask`  
  Send a JSON payload `{ "text": "your question" }` to get a response and TTS audio.
- `POST /stt`  
  Upload an audio file (wav, webm, ogg, opus) to get the transcribed text.
- `GET /audio`  
  Download the latest generated TTS audio as `response.wav`.

## Example Usage
### Ask Endpoint
```bash
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d "{\"text\": \"Hello, Zhara!\"}"
```

### STT Endpoint
```bash
curl -X POST "http://localhost:8000/stt" -F "file=@your_audio.wav"
```

### Audio Endpoint
```bash
curl -O "http://localhost:8000/audio"
```

## Notes
- Make sure Ollama is running locally if you want to use the LLM endpoint.
- The static files are served from the project directory root.

## License
MIT
