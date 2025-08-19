"""
TTS Service with Async Processing and Queue Management
Handles text-to-speech generation with background workers and proper error handling
"""

import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
from queue import Queue, Empty
import threading
import time
import logging

from TTS.api import TTS
import torch
from pydub import AudioSegment

from utils import MemoryManager, TextProcessor  # Removed unused FileManager

logger = logging.getLogger(__name__)

@dataclass
class TTSJob:
    """TTS job structure for queue processing"""
    job_id: str
    text: str
    output_path: str
    model_name: str
    priority: int = 1  # Lower number = higher priority
    created_at: float = None
    callback: Optional[callable] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

@dataclass
class TTSResult:
    """TTS processing result"""
    job_id: str
    success: bool
    audio_path: Optional[str] = None
    error_message: Optional[str] = None
    processing_time: float = 0
    visemes: Optional[List[Dict]] = None

class TTSService:
    """
    Async TTS service with background worker and queue management
    Decouples TTS processing from API endpoints to prevent timeouts
    """

    def __init__(self, model_path: str, gpu_enabled: bool = True, max_workers: int = 2):
        self.model_path = model_path
        self.gpu_enabled = gpu_enabled and torch.cuda.is_available()
        self.max_workers = max_workers

        # Queue and worker management
        self.job_queue = Queue()
        self.result_store: Dict[str, TTSResult] = {}
        self.workers: List[threading.Thread] = []
        self.shutdown_event = threading.Event()

        # TTS model instances (one per worker to avoid threading issues)
        self.models: Dict[int, TTS] = {}

        # Statistics
        self.stats = {
            "jobs_processed": 0,
            "jobs_failed": 0,
            "average_processing_time": 0,
            "queue_size": 0
        }

        self._initialize_workers()

    def _initialize_workers(self):
        """Initialize background worker threads"""
        for worker_id in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(worker_id,),
                daemon=True,
                name=f"TTSWorker-{worker_id}"
            )
            worker.start()
            self.workers.append(worker)

        logger.info(f"Initialized {self.max_workers} TTS workers")

    def _get_model(self, worker_id: int) -> TTS:
        """Get or create TTS model for specific worker"""
        if worker_id not in self.models:
            try:
                logger.info(f"Loading TTS model for worker {worker_id}: {self.model_path}")

                # Try with the configured model first
                try:
                    self.models[worker_id] = TTS(
                        self.model_path,
                        gpu=self.gpu_enabled
                    )
                    logger.info(f"TTS model loaded successfully for worker {worker_id}")
                    return self.models[worker_id]
                except Exception as model_error:
                    logger.warning(f"Primary TTS model failed for worker {worker_id}: {model_error}")

                # Try fallback models
                fallback_models = [
                    "tts_models/en/ljspeech/tacotron2-DDC",
                    "tts_models/en/ljspeech/fast_pitch",
                    "tts_models/en/ljspeech/glow-tts"
                ]

                for fallback_model in fallback_models:
                    try:
                        logger.info(f"Trying fallback TTS model: {fallback_model}")
                        self.models[worker_id] = TTS(fallback_model, gpu=False)  # Use CPU for fallback
                        logger.info(f"Fallback TTS model {fallback_model} loaded successfully for worker {worker_id}")
                        return self.models[worker_id]
                    except Exception as fallback_error:
                        logger.warning(f"Fallback model {fallback_model} failed: {fallback_error}")
                        continue

                # If all models fail, raise an error
                raise RuntimeError(f"Could not load any TTS model for worker {worker_id}")

            except Exception as e:
                logger.error(f"All TTS model loading failed for worker {worker_id}: {e}")
                raise RuntimeError(f"Could not load any TTS model for worker {worker_id}")

        return self.models[worker_id]

    def _worker_loop(self, worker_id: int):
        """Main worker loop for processing TTS jobs"""
        logger.info(f"TTS Worker {worker_id} started")

        while not self.shutdown_event.is_set():
            try:
                # Get job from queue with timeout
                job = self.job_queue.get(timeout=1.0)

                if job is None:  # Shutdown signal
                    break

                self._process_job(worker_id, job)
                self.job_queue.task_done()

            except Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

        logger.info(f"TTS Worker {worker_id} stopped")

    def _split_for_tts(self, text: str, max_chars: int = 300) -> List[str]:
        """Split text into chunks not exceeding max_chars, respecting sentence boundaries."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        chunks: List[str] = []
        current = []
        current_len = 0
        for s in sentences:
            if not s:
                continue
            s = s.strip()
            if current_len + len(s) + (1 if current else 0) <= max_chars:
                current.append(s)
                current_len += len(s) + (1 if current else 0)
            else:
                if current:
                    chunks.append(' '.join(current))
                # If a single sentence is too long, hard-split it
                if len(s) > max_chars:
                    for i in range(0, len(s), max_chars):
                        chunks.append(s[i:i+max_chars])
                    current = []
                    current_len = 0
                else:
                    current = [s]
                    current_len = len(s)
        if current:
            chunks.append(' '.join(current))
        return chunks

    def _synthesize_chunks_and_merge(self, tts_model: TTS, text: str, output_path: Path, timeout_per_chunk: int = 30):
        """Synthesize long text in chunks and merge into a single wav file."""
        tmp_files: List[Path] = []
        try:
            chunks = self._split_for_tts(text, max_chars=300)
            logger.info(f"TTS chunking: {len(chunks)} chunks")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Generate each chunk
            for idx, chunk in enumerate(chunks):
                part_path = output_path.parent / f"{output_path.stem}_part{idx}.wav"
                tmp_files.append(part_path)

                def call_chunk():
                    # Use named parameters to ensure file_path is correct and avoid misinterpreting as speaker
                    tts_model.tts_to_file(text=chunk, file_path=str(part_path))
                    return True

                with ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(call_chunk)
                    fut.result(timeout=timeout_per_chunk)

                if not part_path.exists() or part_path.stat().st_size == 0:
                    raise RuntimeError(f"Chunk {idx} synthesis failed")

                # Cleanup GPU between chunks
                MemoryManager.cleanup_gpu_memory()

            # Merge chunks
            combined = None
            for fp in tmp_files:
                seg = AudioSegment.from_file(fp, format='wav')
                combined = seg if combined is None else combined + seg
            if combined is None:
                raise RuntimeError("No audio generated from chunks")
            combined.export(str(output_path), format='wav')

        finally:
            # Remove tmp parts
            for fp in tmp_files:
                try:
                    if fp.exists():
                        fp.unlink()
                except Exception:
                    pass

    def _process_job(self, worker_id: int, job: TTSJob):
        """Process a single TTS job"""
        start_time = time.time()

        try:
            # Clean memory before processing
            MemoryManager.cleanup_gpu_memory()

            # Get model for this worker
            tts_model = self._get_model(worker_id)

            # Clean text for TTS
            cleaned_text = TextProcessor.clean_text_for_tts(job.text)

            if not cleaned_text:
                raise ValueError("Empty text after cleaning")

            # Ensure output directory exists
            output_path = Path(job.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Generate TTS with timeout protection
            with ThreadPoolExecutor(max_workers=1) as executor:
                # Use the simplest possible TTS call to avoid parameter issues
                logger.info(f"Generating TTS for text length: {len(cleaned_text)} characters")

                def safe_tts_call():
                    """Safe TTS call that handles both single and multi-speaker models; falls back to chunked synthesis for long text."""
                    # For long texts, do chunked synthesis to avoid model timeouts
                    if len(cleaned_text) > 600:
                        self._synthesize_chunks_and_merge(tts_model, cleaned_text, output_path)
                        return True
                    # Use named parameters to avoid misinterpreting file_path as speaker
                    tts_model.tts_to_file(text=cleaned_text, file_path=str(output_path))
                    return True

                future = executor.submit(safe_tts_call)

                try:
                    # Increase timeout for longer texts - calculate based on text length
                    text_length = len(cleaned_text)
                    if text_length > 1500:
                        timeout = 150  # allow more time due to chunk merging
                    elif text_length > 1000:
                        timeout = 120
                    elif text_length > 500:
                        timeout = 90
                    else:
                        timeout = 45

                    logger.info(f"Using {timeout}s timeout for text of {text_length} characters")
                    future.result(timeout=timeout)
                    logger.info(f"TTS generation completed successfully")
                except FutureTimeoutError:
                    future.cancel()
                    raise TimeoutError(f"TTS generation timed out after {timeout} seconds")

            # Verify output file
            if not output_path.exists() or output_path.stat().st_size == 0:
                raise RuntimeError("TTS file generation failed - empty or missing file")

            processing_time = time.time() - start_time

            # Generate visemes if needed (placeholder for now)
            visemes = self._generate_visemes(cleaned_text)

            result = TTSResult(
                job_id=job.job_id,
                success=True,
                audio_path=str(output_path),
                processing_time=processing_time,
                visemes=visemes
            )

            # Update statistics
            self.stats["jobs_processed"] += 1
            self.stats["average_processing_time"] = (
                (self.stats["average_processing_time"] * (self.stats["jobs_processed"] - 1) +
                 processing_time) / self.stats["jobs_processed"]
            )

            logger.info(f"TTS job {job.job_id} completed in {processing_time:.2f}s")

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)

            logger.error(f"TTS job {job.job_id} failed: {error_msg}")

            result = TTSResult(
                job_id=job.job_id,
                success=False,
                error_message=error_msg,
                processing_time=processing_time
            )

            self.stats["jobs_failed"] += 1

            # Clean up partial files
            try:
                if Path(job.output_path).exists():
                    Path(job.output_path).unlink()
            except Exception:
                pass

        finally:
            # Store result and call callback if provided
            self.result_store[job.job_id] = result

            if job.callback:
                try:
                    job.callback(result)
                except Exception as e:
                    logger.error(f"Error in TTS job callback: {e}")

            # Cleanup memory after processing
            MemoryManager.cleanup_gpu_memory()

    def _generate_visemes(self, text: str) -> List[Dict]:
        """Generate visemes for text (placeholder implementation)"""
        # This would integrate with actual viseme generation logic
        # For now, return empty list
        return []

    async def submit_job(self, text: str, output_path: str,
                        model_name: str = "default",
                        priority: int = 1) -> str:
        """Submit a TTS job and return job ID"""
        job_id = str(uuid.uuid4())

        job = TTSJob(
            job_id=job_id,
            text=text,
            output_path=output_path,
            model_name=model_name,
            priority=priority
        )

        self.job_queue.put(job)
        self.stats["queue_size"] = self.job_queue.qsize()

        logger.info(f"Submitted TTS job {job_id}, queue size: {self.stats['queue_size']}")
        return job_id

    async def get_result(self, job_id: str, timeout: float = 60) -> Optional[TTSResult]:
        """Get result for a job ID with timeout"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            if job_id in self.result_store:
                result = self.result_store.pop(job_id)
                return result

            await asyncio.sleep(0.1)  # Check every 100ms

        # Timeout reached
        logger.warning(f"Timeout waiting for TTS job {job_id}")
        return None

    async def process_text_to_speech(self, text: str, output_path: str,
                                   timeout: float = 60) -> TTSResult:
        """
        High-level async interface for TTS processing
        Submit job and wait for result
        """
        job_id = await self.submit_job(text, output_path)
        result = await self.get_result(job_id, timeout)

        if result is None:
            return TTSResult(
                job_id=job_id,
                success=False,
                error_message="TTS processing timed out"
            )

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        stats = self.stats.copy()
        stats["queue_size"] = self.job_queue.qsize()
        stats["active_workers"] = len([w for w in self.workers if w.is_alive()])
        return stats

    def shutdown(self):
        """Shutdown the TTS service gracefully"""
        logger.info("Shutting down TTS service...")

        # Signal workers to stop
        self.shutdown_event.set()

        # Add None jobs to wake up workers
        for _ in range(self.max_workers):
            self.job_queue.put(None)

        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5)

        # Clean up models
        for model in self.models.values():
            try:
                del model
            except Exception:
                pass

        MemoryManager.cleanup_gpu_memory()
        logger.info("TTS service shutdown complete")

# Global TTS service instance
_tts_service: Optional[TTSService] = None

def get_tts_service() -> TTSService:
    """Get global TTS service instance"""
    global _tts_service
    if _tts_service is None:
        raise RuntimeError("TTS service not initialized")
    return _tts_service

def initialize_tts_service(model_path: str, gpu_enabled: bool = True,
                          max_workers: int = 2) -> TTSService:
    """Initialize global TTS service"""
    global _tts_service
    if _tts_service is not None:
        _tts_service.shutdown()

    _tts_service = TTSService(model_path, gpu_enabled, max_workers)
    return _tts_service
