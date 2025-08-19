"""
API Router for Zhara AI Assistant
Centralized endpoint definitions with proper separation of concerns
"""

import asyncio
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import platform  # Add

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
import soundfile as sf
import numpy as np
import aiohttp

from session_manager import SessionManager
from tts_service import get_tts_service
from utils import TextProcessor, MemoryManager, RateLimiter
import config
import logging
from chroma_memory import ChromaMemory

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Initialize rate limiter
rate_limiter = RateLimiter(max_requests=60, time_window=60)

# Request/Response Models
class ChatRequest(BaseModel):
    text: str
    model: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    audio_file: str
    viseme_file: str
    session_id: str
    model_used: str
    # New URL fields for frontend compatibility
    audio_url: Optional[str] = None
    viseme_url: Optional[str] = None

class SessionInfo(BaseModel):
    session_id: str
    title: str
    created_at: str
    last_updated: str
    message_count: int

class CreateSessionRequest(BaseModel):
    title: Optional[str] = None

class UpdateSessionRequest(BaseModel):
    title: str

class ModelInfo(BaseModel):
    name: str
    size: Optional[str] = None
    modified_at: Optional[str] = None

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "default"

# Dependency Functions
async def get_rate_limit():
    """Rate limiting dependency"""
    # In a real implementation, you'd extract client IP or user ID
    client_id = "default"  # Placeholder

    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )
    return True

async def get_session_manager() -> SessionManager:
    """Get session manager instance"""
    if not hasattr(get_session_manager, '_instance'):
        get_session_manager._instance = SessionManager()
    return get_session_manager._instance

SKIP_CHROMA = platform.system() == "Windows"  # Skip semantic memory on Windows

async def get_chroma_memory() -> ChromaMemory:
    """Get ChromaMemory singleton instance"""
    if SKIP_CHROMA:
        return None
    if not hasattr(get_chroma_memory, '_instance'):
        # Persist in project storage directory
        persist_dir = str((Path(config.STORAGE_DIR) / 'chroma_db').resolve()) if hasattr(config, 'STORAGE_DIR') else './chroma_db'
        get_chroma_memory._instance = ChromaMemory(persist_directory=persist_dir)
    return get_chroma_memory._instance

# Audio Processing Utilities
def decode_audio_file(file_content: bytes) -> tuple[np.ndarray, int]:
    """Decode audio file content to numpy array"""
    try:
        import io
        audio_data, sample_rate = sf.read(io.BytesIO(file_content))

        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            from scipy.signal import resample
            audio_data = resample(audio_data, int(len(audio_data) * 16000 / sample_rate))

        return audio_data, 16000
    except Exception as e:
        logger.error(f"Error decoding audio file: {e}")
        raise HTTPException(status_code=400, detail="Invalid audio file format")

# AI Response Generation
async def generate_ai_response(text: str, model: str, session_id: Optional[str] = None) -> str:
    """Generate AI response using the specified model with session-aware context"""
    try:
        session_mgr = await get_session_manager()
        memory = None
        if not SKIP_CHROMA:
            try:
                memory = await get_chroma_memory()
            except Exception as mem_err:
                logger.warning(f"Chroma memory unavailable, continuing without semantic memory: {mem_err}")
                memory = None

        # Recent conversation context (last 8 exchanges)
        recent_msgs = []
        try:
            if session_id:
                msgs = session_mgr.get_session_messages(session_id, limit=16)
                # Format as simple plain text turns (avoid markdown)
                for m in msgs:
                    recent_msgs.append(f"User: {m.user_message}")
                    recent_msgs.append(f"Assistant: {m.ai_response}")
        except Exception:
            recent_msgs = []

        # Semantic memories relevant to current input
        similar = {"documents": []}
        try:
            if memory:
                similar = memory.query_memory(text, n_results=5, session_id=session_id, max_age_hours=24*7)
        except Exception:
            pass
        similar_docs = []
        try:
            docs = similar.get('documents') or []
            if docs and isinstance(docs[0], list):
                similar_docs = docs[0]
            elif isinstance(docs, list):
                similar_docs = docs
        except Exception:
            similar_docs = []

        # Build context block (keep modest size)
        ctx_parts = []
        if recent_msgs:
            ctx_parts.append("Recent conversation:\n" + "\n".join(recent_msgs[-12:]))
        if similar_docs:
            ctx_parts.append("Relevant memory:\n" + "\n".join(similar_docs[:5]))
        context_block = ("\n\n".join(ctx_parts)).strip()
        if len(context_block) > 1500:
            context_block = context_block[:1500]

        # Skip offline model handling
        if model == "offline" or config.OFFLINE_MODE:
            logger.info("Using offline mode for response generation")
            return get_offline_response(text)

        # Plain-text instruction to minimize markdown/special characters in output
        system_plain_text_instruction = (
            "You are a helpful assistant. Respond in plain English text only. "
            "Do not use markdown, code blocks, or any special characters beyond standard punctuation. "
            "Avoid headings, lists, emojis, hashes (#), asterisks (*), underscores (_), backticks (`), tildes (~), "
            "brackets [], braces {}, angle brackets <>, pipes |, or inline HTML. "
            "Write complete sentences as normal prose."
        )

        # Compose final prompt with instruction + context
        final_prompt = (
            f"{system_plain_text_instruction}\n\n" +
            (f"Context:\n{context_block}\n\n" if context_block else "") +
            f"User message:\n{text}\n\nAnswer:"
        ).strip()

        # Try to get response from Ollama
        logger.info(f"Generating response with model: {model}")
        timeout_total = getattr(config, 'LLM_TIMEOUT_SECONDS', 45) or 45
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_total)) as session:
            try:
                payload = {
                    "model": model,
                    # Include instruction both as system and in prompt for broader model compatibility
                    "system": system_plain_text_instruction,
                    "prompt": final_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 1000
                    }
                }

                async with session.post(
                    f"{config.OLLAMA_BASE_URL}/api/generate",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        ai_response = result.get('response', '').strip()

                        if ai_response:
                            logger.info(f"Successfully generated response using {model}")
                            return ai_response
                        else:
                            logger.warning("Empty response from Ollama")
                            return get_offline_response(text)
                    else:
                        error_text = await response.text()
                        logger.warning(f"Ollama API error {response.status}: {error_text}")
                        return get_offline_response(text)

            except asyncio.TimeoutError:
                logger.warning(f"Timeout calling Ollama API with model {model}")
                return get_offline_response(text)
            except Exception as e:
                logger.warning(f"Ollama API failed with model {model}: {e}")
                return get_offline_response(text)

    except Exception as e:
        logger.error(f"Error generating AI response: {e}")
        return get_offline_response(text)

def get_offline_response(text: str) -> str:
    """Generate offline response when API is unavailable"""
    responses = [
        "I understand you're asking about that. Let me think about it for a moment.",
        "That's an interesting question. Based on what I know, here's what I think.",
        "I hear what you're saying. Let me provide you with some thoughts on that.",
        "Thank you for sharing that with me. Here's my perspective on the matter."
    ]

    # Simple keyword-based responses
    text_lower = text.lower()

    if any(word in text_lower for word in ['hello', 'hi', 'hey']):
        return "Hello! How can I help you today?"
    elif any(word in text_lower for word in ['thank', 'thanks']):
        return "You're welcome! Is there anything else I can help you with?"
    elif any(word in text_lower for word in ['bye', 'goodbye']):
        return "Goodbye! Have a great day!"
    else:
        import random
        return random.choice(responses)

# API Endpoints
@router.post("/ask", response_model=ChatResponse)
@router.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    session_manager: SessionManager = Depends(get_session_manager),
    rate_limit: None = Depends(get_rate_limit)
):
    """
    Main chat endpoint with async TTS processing
    Supports both /ask (legacy) and /api/chat endpoints
    """
    try:
        # Input validation
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Empty message")

        if len(request.text) > config.MAX_MESSAGE_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Message too long (max {config.MAX_MESSAGE_LENGTH} characters)"
            )

        # Get or create session
        session_id = request.session_id
        if not session_id:
            session_id = session_manager.create_session()
        elif not session_manager.get_session(session_id):
            session_id = session_manager.create_session()

        # Generate response using AI model
        reply = await generate_ai_response(request.text, request.model, session_id)

        # Persist to memory (user + assistant turns) using effective session_id
        if not SKIP_CHROMA:
            try:
                memory = await get_chroma_memory()
                if session_id and memory:
                    memory.add_memory(request.text, metadata={"role": "user", "model": request.model}, session_id=session_id)
                    memory.add_memory(reply, metadata={"role": "assistant", "model": request.model}, session_id=session_id)
            except Exception as e:
                logger.warning(f"Could not persist to Chroma memory: {e}")

        # Clean text for TTS
        cleaned_reply = TextProcessor.clean_text_for_tts(reply)

        # Generate unique filenames
        response_id = str(uuid.uuid4())
        audio_filename = f"response_{response_id}.wav"
        viseme_filename = f"visemes_{response_id}.json"

        audio_path = Path(config.AUDIO_OUTPUT_DIR) / audio_filename
        viseme_path = Path(config.VISEME_OUTPUT_DIR) / viseme_filename

        # Submit TTS job to background service
        logger.info(f"Starting TTS processing for text: {cleaned_reply[:50]}...")

        # Compute wait timeout based on text length (buffer > worker timeout)
        text_len = len(cleaned_reply)
        if text_len > 1500:
            wait_timeout = 180  # must exceed worker 150s
        elif text_len > 1000:
            wait_timeout = 140  # exceed worker 120s
        elif text_len > 500:
            wait_timeout = 110  # exceed worker 90s
        else:
            wait_timeout = 60   # exceed worker 45s

        tts_result = None
        try:
            tts_service = get_tts_service()
            logger.info(f"TTS service stats before processing: {tts_service.get_stats()}")

            tts_result = await tts_service.process_text_to_speech(
                cleaned_reply,
                str(audio_path),
                timeout=wait_timeout
            )

            if tts_result and tts_result.success:
                logger.info(f"TTS successful: {audio_filename}, processing time: {tts_result.processing_time:.2f}s")
                # Verify the file actually exists and has content
                if audio_path.exists() and audio_path.stat().st_size > 0:
                    logger.info(f"Audio file verified: {audio_path} ({audio_path.stat().st_size} bytes)")
                else:
                    logger.warning(f"Audio file missing or empty after TTS success: {audio_path}")
                    audio_filename = await create_placeholder_audio(audio_path, cleaned_reply)
            else:
                error_msg = tts_result.error_message if tts_result else "TTS service returned None"
                logger.error(f"TTS failed: {error_msg}")
                audio_filename = await create_placeholder_audio(audio_path, cleaned_reply)

        except Exception as tts_error:
            logger.error(f"TTS service error: {tts_error}", exc_info=True)
            audio_filename = await create_placeholder_audio(audio_path, cleaned_reply)

        # Generate visemes (if TTS was successful)
        visemes = tts_result.visemes if tts_result and tts_result.success else []
        await save_visemes(visemes, viseme_path)

        # Save message to session
        session_manager.add_message(
            session_id=session_id,
            user_message=request.text,
            ai_response=reply,
            model_used=request.model,
            audio_file=audio_filename,
            viseme_file=viseme_filename
        )

        # Build URLs for frontend playback
        audio_url = f"/audio/{audio_filename}"
        viseme_url = f"/visemes/{viseme_filename}"

        return ChatResponse(
            reply=reply,
            audio_file=audio_filename,
            viseme_file=viseme_filename,
            session_id=session_id,
            model_used=request.model,
            audio_url=audio_url,
            viseme_url=viseme_url
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/api/tts")
@router.post("/tts")
async def text_to_speech(
    request: TTSRequest,
    rate_limit: None = Depends(get_rate_limit)
):
    """Standalone TTS endpoint"""
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Empty text")

        # Generate unique filename
        response_id = str(uuid.uuid4())
        audio_filename = f"tts_{response_id}.wav"
        audio_path = Path(config.AUDIO_OUTPUT_DIR) / audio_filename

        # Process with TTS service
        tts_service = get_tts_service()
        result = await tts_service.process_text_to_speech(
            request.text,
            str(audio_path)
        )

        if not result.success:
            raise HTTPException(status_code=500, detail=result.error_message)

        # Build URL for frontend playback
        audio_url = f"/audio/{audio_filename}"

        return {"audio_file": audio_filename, "audio_url": audio_url, "processing_time": result.processing_time}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS endpoint error: {e}")
        raise HTTPException(status_code=500, detail="TTS processing failed")

@router.post("/api/stt")
@router.post("/stt")
async def speech_to_text(
    file: UploadFile = File(...),
    rate_limit: None = Depends(get_rate_limit)
):
    """Speech-to-text endpoint with proper async handling"""
    try:
        # Validate file
        file_content = await file.read()
        if not file_content:
            raise HTTPException(status_code=400, detail="Empty audio file")

        if len(file_content) > config.MAX_AUDIO_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds maximum of {config.MAX_AUDIO_SIZE / 1024 / 1024:.2f}MB"
            )

        # Decode audio
        audio_data, sample_rate = decode_audio_file(file_content)

        # Check duration
        duration = len(audio_data) / sample_rate
        if duration > config.MAX_AUDIO_DURATION:
            raise HTTPException(
                status_code=400,
                detail=f"Audio duration ({duration:.1f}s) exceeds maximum of {config.MAX_AUDIO_DURATION}s"
            )

        # Process with Whisper (assuming whisper_model is available globally)
        # This would need to be adapted to your specific Whisper model setup
        text = await process_speech_to_text(audio_data)

        return {"text": text}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"STT endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Speech recognition failed")

# Session Management Endpoints
@router.post("/api/sessions", response_model=SessionInfo)
async def create_session(
    request: CreateSessionRequest,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """Create a new chat session"""
    session_id = session_manager.create_session(request.title)
    session = session_manager.get_session(session_id)

    return SessionInfo(
        session_id=session.session_id,
        title=session.title,
        created_at=session.created_at,
        last_updated=session.last_updated,
        message_count=session.message_count
    )

@router.get("/api/sessions", response_model=List[SessionInfo])
async def list_sessions(
    limit: Optional[int] = 50,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """List all chat sessions"""
    sessions = session_manager.list_sessions(limit)

    return [
        SessionInfo(
            session_id=session.session_id,
            title=session.title,
            created_at=session.created_at,
            last_updated=session.last_updated,
            message_count=session.message_count
        )
        for session in sessions
    ]

@router.get("/api/sessions/{session_id}")
async def get_session_details(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """Get detailed session information including messages"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = session_manager.get_session_messages(session_id)

    return {
        "session": SessionInfo(
            session_id=session.session_id,
            title=session.title,
            created_at=session.created_at,
            last_updated=session.last_updated,
            message_count=session.message_count
        ),
        "messages": messages
    }

@router.put("/api/sessions/{session_id}")
async def update_session(
    session_id: str,
    request: UpdateSessionRequest,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """Update session title"""
    success = session_manager.update_session(session_id, title=request.title)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"message": "Session updated successfully"}

@router.delete("/api/sessions/{session_id}")
async def delete_session(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """Delete a session"""
    success = session_manager.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"message": "Session deleted successfully"}

# System and Model Endpoints
@router.get("/api/models")
async def get_available_models():
    """Get available AI models"""
    try:
        # This would integrate with your model discovery logic
        models = await fetch_available_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        return {"models": [{"name": "offline"}], "error": str(e)}

@router.get("/api/stats")
async def get_system_stats(
    session_manager: SessionManager = Depends(get_session_manager)
):
    """Get system statistics"""
    try:
        # TTS service stats
        tts_service = get_tts_service()
        tts_stats = tts_service.get_stats()

        # Session stats
        session_stats = session_manager.get_session_stats()

        # Memory stats
        memory_stats = MemoryManager.get_memory_usage()

        return {
            "tts": tts_stats,
            "sessions": session_stats,
            "memory": memory_stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {"error": str(e)}

# File Serving Endpoints
@router.get("/audio/{filename}")
async def serve_audio(filename: str):
    """Serve audio files"""
    file_path = Path(config.AUDIO_OUTPUT_DIR) / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(
        file_path,
        media_type="audio/wav",
        headers={"Cache-Control": "public, max-age=3600"}
    )

@router.get("/visemes/{filename}")
async def serve_visemes(filename: str):
    """Serve viseme files"""
    file_path = Path(config.VISEME_OUTPUT_DIR) / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Viseme file not found")

    return FileResponse(
        file_path,
        media_type="application/json",
        headers={"Cache-Control": "public, max-age=3600"}
    )

# Helper Functions - Proper implementations
async def fetch_available_models() -> List[Dict[str, Any]]:
    """Fetch available models from Ollama"""
    try:
        if config.OFFLINE_MODE:
            return [{"name": "offline", "size": "N/A"}]

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.get(f"{config.OLLAMA_BASE_URL}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = []

                    for model in data.get('models', []):
                        model_info = {
                            "name": model.get('name', 'unknown'),
                            "size": format_bytes(model.get('size', 0)),
                            "modified_at": model.get('modified_at', '')
                        }
                        models.append(model_info)

                    logger.info(f"Found {len(models)} Ollama models")
                    return models
                else:
                    logger.warning(f"Ollama API returned status {response.status}")
                    return [{"name": "offline", "size": "N/A"}]

    except Exception as e:
        logger.error(f"Error fetching Ollama models: {e}")
        return [{"name": "offline", "size": "N/A"}]

def format_bytes(bytes_val: int) -> str:
    """Format bytes to human readable format"""
    if bytes_val == 0:
        return "0 B"

    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} PB"

async def process_speech_to_text(audio_data: np.ndarray) -> str:
    """Process audio with Whisper - basic implementation"""
    try:
        # This is a placeholder - you would integrate with your actual Whisper model
        # For now, return a basic transcription message
        logger.info("Processing speech-to-text (placeholder implementation)")
        return "Transcribed: Audio processing not fully implemented yet"
    except Exception as e:
        logger.error(f"Speech-to-text error: {e}")
        return "Error: Could not transcribe audio"

async def save_visemes(visemes: List[Dict], output_path: Path):
    """Save visemes to file"""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        import json
        with open(output_path, 'w') as f:
            json.dump(visemes, f, indent=2)
        logger.debug(f"Saved visemes to {output_path}")
    except Exception as e:
        logger.error(f"Error saving visemes: {e}")

async def create_placeholder_audio(output_path: Path, text: str) -> str:
    """Create a simple placeholder audio file when TTS fails"""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate a simple beep sound - 800Hz tone for 0.5 seconds
        sample_rate = 16000
        frequency = 800  # 800Hz beep
        duration = 0.5  # 0.5 seconds
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

        # Create a simple beep with fade in/out to avoid clicks
        waveform = 0.3 * np.sin(2 * np.pi * frequency * t)
        fade_samples = int(0.05 * sample_rate)  # 50ms fade
        waveform[:fade_samples] *= np.linspace(0, 1, fade_samples)
        waveform[-fade_samples:] *= np.linspace(1, 0, fade_samples)

        # Write to WAV file
        sf.write(output_path, waveform, sample_rate)

        logger.info(f"Created placeholder audio file: {output_path}")
        return str(output_path.name)

    except Exception as e:
        logger.error(f"Error creating placeholder audio: {e}")
        # Return the filename anyway so the frontend doesn't break
        return str(output_path.name)
