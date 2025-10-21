import asyncio
import json
import os
import numpy as np
from fastapi import FastAPI, WebSocket, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
import uvicorn
import logging
from prompts import PROMPTS
from openai_realtime_client import OpenAIRealtimeAudioTextClient
from starlette.websockets import WebSocketState
import wave
import datetime
import scipy.signal
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel, Field
from typing import Generator
from llm_processor import get_llm_processor
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic models for request and response schemas
class ReadabilityRequest(BaseModel):
    text: str = Field(..., description="The text to improve readability for.")
    api_key: str = Field(None, description="Optional OpenAI API key")
    llm_provider: str = Field("compatible", description="LLM provider: 'openai' or 'compatible'")
    openai_model: str = Field("gpt-4o", description="OpenAI model name")
    compatible_base_url: str = Field(None, description="Optional compatible API base URL")
    compatible_model: str = Field(None, description="Optional compatible model name")
    compatible_api_key: str = Field(None, description="Optional compatible API key")

class ReadabilityResponse(BaseModel):
    enhanced_text: str = Field(..., description="The text with improved readability.")

class CorrectnessRequest(BaseModel):
    text: str = Field(..., description="The text to check for factual correctness.")
    api_key: str = Field(None, description="Optional OpenAI API key")
    llm_provider: str = Field("compatible", description="LLM provider: 'openai' or 'compatible'")
    openai_model: str = Field("gpt-4o", description="OpenAI model name")
    compatible_base_url: str = Field(None, description="Optional compatible API base URL")
    compatible_model: str = Field(None, description="Optional compatible model name")
    compatible_api_key: str = Field(None, description="Optional compatible API key")

class CorrectnessResponse(BaseModel):
    analysis: str = Field(..., description="The factual correctness analysis.")

class AskAIRequest(BaseModel):
    text: str = Field(..., description="The question to ask AI.")
    api_key: str = Field(None, description="Optional OpenAI API key")
    llm_provider: str = Field("compatible", description="LLM provider: 'openai' or 'compatible'")
    openai_model: str = Field("gpt-4o", description="OpenAI model name")
    compatible_base_url: str = Field(None, description="Optional compatible API base URL")
    compatible_model: str = Field(None, description="Optional compatible model name")
    compatible_api_key: str = Field(None, description="Optional compatible API key")

class AskAIResponse(BaseModel):
    answer: str = Field(..., description="AI's answer to the question.")

app = FastAPI()

# Make API key optional from environment variables
# 配置 LLM - 优先使用 GLM-4.6-FP8（成本更低）
USE_GLM = os.getenv("USE_GLM", "true").lower() == "true"  # 默认使用 GLM
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if USE_GLM:
    try:
        logger.info("Using GLM-4.6-FP8 from config.yml for text processing")
        llm_processor = get_llm_processor("GLM-4.6-FP8")
        logger.info("GLM processor initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize GLM processor: {e}")
        if OPENAI_API_KEY:
            logger.info("Fallback to OpenAI GPT-4o")
            llm_processor = get_llm_processor("gpt-4o")
        else:
            logger.warning("No LLM available. API key must be provided by user.")
            llm_processor = None
elif OPENAI_API_KEY:
    logger.info("Using OpenAI GPT-4o from environment variables")
    llm_processor = get_llm_processor("gpt-4o")
else:
    logger.warning("No LLM configured. API key must be provided by user.")
    llm_processor = None

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_realtime_page(request: Request):
    return FileResponse("static/realtime.html")

class AudioProcessor:
    def __init__(self, target_sample_rate=24000):
        self.target_sample_rate = target_sample_rate
        self.source_sample_rate = 48000  # Most common sample rate for microphones
        
    def process_audio_chunk(self, audio_data):
        # Convert binary audio data to Int16 array
        pcm_data = np.frombuffer(audio_data, dtype=np.int16)
        
        # Convert to float32 for better precision during resampling
        float_data = pcm_data.astype(np.float32) / 32768.0
        
        # Resample from 48kHz to 24kHz
        resampled_data = scipy.signal.resample_poly(
            float_data, 
            self.target_sample_rate, 
            self.source_sample_rate
        )
        
        # Convert back to int16 while preserving amplitude
        resampled_int16 = (resampled_data * 32768.0).clip(-32768, 32767).astype(np.int16)
        return resampled_int16.tobytes()

    def save_audio_buffer(self, audio_buffer, filename):
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)  # Mono audio
            wf.setsampwidth(2)  # 2 bytes per sample (16-bit)
            wf.setframerate(self.target_sample_rate)
            wf.writeframes(b''.join(audio_buffer))
        logger.info(f"Saved audio buffer to {filename}")

@app.websocket("/api/v1/ws")
async def websocket_endpoint(websocket: WebSocket):
    logger.info("New WebSocket connection attempt")
    await websocket.accept()
    logger.info("WebSocket connection accepted")

    # Add initial status update here
    await websocket.send_text(json.dumps({
        "type": "status",
        "status": "idle"  # Set initial status to idle (blue)
    }))

    client = None
    audio_processor = AudioProcessor()
    audio_buffer = []
    recording_stopped = asyncio.Event()
    openai_ready = asyncio.Event()
    pending_audio_chunks = []
    # Add synchronization for audio sending operations
    pending_audio_operations = 0
    audio_send_lock = asyncio.Lock()
    all_audio_sent = asyncio.Event()
    all_audio_sent.set()  # Initially set since no audio is pending
    marker_prefix = "下面是语音识别转录结果：\n\n"
    max_prefix_deltas = 20
    response_buffer = []
    marker_seen = False
    delta_counter = 0
    # Store user's API key
    user_api_key = None

    async def initialize_openai():
        nonlocal client
        try:
            # Clear the ready flag while initializing
            openai_ready.clear()

            # Use user's API key if provided, otherwise fall back to environment variable
            api_key = user_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("No API key provided. Please set your OpenAI API key.")

            client = OpenAIRealtimeAudioTextClient(api_key)
            await client.connect()
            logger.info("Successfully connected to OpenAI client")
            
            # Register handlers after client is initialized
            client.register_handler("session.updated", lambda data: handle_generic_event("session.updated", data))
            client.register_handler("input_audio_buffer.cleared", lambda data: handle_generic_event("input_audio_buffer.cleared", data))
            client.register_handler("input_audio_buffer.speech_started", lambda data: handle_generic_event("input_audio_buffer.speech_started", data))
            client.register_handler("rate_limits.updated", lambda data: handle_generic_event("rate_limits.updated", data))
            client.register_handler("response.output_item.added", lambda data: handle_generic_event("response.output_item.added", data))
            client.register_handler("conversation.item.created", lambda data: handle_generic_event("conversation.item.created", data))
            client.register_handler("response.content_part.added", lambda data: handle_generic_event("response.content_part.added", data))
            client.register_handler("response.text.done", lambda data: handle_generic_event("response.text.done", data))
            client.register_handler("response.content_part.done", lambda data: handle_generic_event("response.content_part.done", data))
            client.register_handler("response.output_item.done", lambda data: handle_generic_event("response.output_item.done", data))
            client.register_handler("response.done", lambda data: handle_response_done(data))
            client.register_handler("error", lambda data: handle_error(data))
            client.register_handler("response.text.delta", lambda data: handle_text_delta(data))
            client.register_handler("response.created", lambda data: handle_response_created(data))
            
            openai_ready.set()  # Set ready flag after successful initialization
            await websocket.send_text(json.dumps({
                "type": "status",
                "status": "connected"
            }))
            return True
        except Exception as e:
            logger.error(f"Failed to connect to OpenAI: {e}")
            openai_ready.clear()  # Ensure flag is cleared on failure
            await websocket.send_text(json.dumps({
                "type": "error",
                "content": "Failed to initialize OpenAI connection"
            }))
            return False

    # Move the handler definitions here (before initialize_openai)
    async def emit_text_delta(content: str):
        if content and websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text(json.dumps({
                "type": "text",
                "content": content,
                "isNewResponse": False
            }))

    async def flush_buffer(with_warning: bool = False):
        nonlocal response_buffer
        if not response_buffer:
            return
        buffered_text = "".join(response_buffer)
        response_buffer = []
        if buffered_text.startswith(marker_prefix):
            buffered_text = buffered_text[len(marker_prefix):]
        if with_warning and not buffered_text:
            logger.warning("Buffered text discarded after removing marker prefix.")
        await emit_text_delta(buffered_text)

    async def handle_text_delta(data):
        nonlocal response_buffer, marker_seen, delta_counter
        try:
            if websocket.client_state != WebSocketState.CONNECTED:
                return

            delta = data.get("delta", "")

            if marker_seen:
                await emit_text_delta(delta)
                logger.info("Handled response.text.delta (passthrough)")
                return

            if delta:
                response_buffer.append(delta)

            if delta:
                delta_counter += 1

            joined = "".join(response_buffer)
            marker_index = joined.find(marker_prefix)

            if marker_index != -1:
                marker_seen = True
                remaining = joined[marker_index + len(marker_prefix):]
                response_buffer = []
                await emit_text_delta(remaining)
                logger.info("Handled response.text.delta (marker detected)")
                return

            if delta_counter >= max_prefix_deltas:
                marker_seen = True
                await flush_buffer(with_warning=True)
                logger.warning("Marker prefix not detected after max deltas; emitted buffered text.")
            else:
                logger.info("Handled response.text.delta (buffering)")
        except Exception as e:
            logger.error(f"Error in handle_text_delta: {str(e)}", exc_info=True)

    async def handle_response_created(data):
        nonlocal response_buffer, marker_seen, delta_counter
        response_buffer = []
        marker_seen = False
        delta_counter = 0
        await websocket.send_text(json.dumps({
            "type": "text",
            "content": "",
            "isNewResponse": True
        }))
        logger.info("Handled response.created")

    async def handle_error(data):
        error_msg = data.get("error", {}).get("message", "Unknown error")
        logger.error(f"OpenAI error: {error_msg}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "content": error_msg
        }))
        logger.info("Handled error message from OpenAI")

    async def handle_response_done(data):
        nonlocal client, response_buffer, marker_seen
        logger.info("Handled response.done")
        if not marker_seen and response_buffer:
            await flush_buffer()
            marker_seen = True
        recording_stopped.set()
        
        if client:
            try:
                await client.close()
                client = None
                openai_ready.clear()
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "status": "idle"
                }))
                logger.info("Connection closed after response completion")
            except Exception as e:
                logger.error(f"Error closing client after response done: {str(e)}")

    async def handle_generic_event(event_type, data):
        logger.info(f"Handled {event_type} with data: {json.dumps(data, ensure_ascii=False)}")

    # Create a queue to handle incoming audio chunks
    audio_queue = asyncio.Queue()

    async def receive_messages():
        nonlocal client
        
        try:
            while True:
                if websocket.client_state == WebSocketState.DISCONNECTED:
                    logger.info("WebSocket client disconnected")
                    openai_ready.clear()
                    break
                    
                try:
                    # Add timeout to prevent infinite waiting
                    data = await asyncio.wait_for(websocket.receive(), timeout=30.0)
                    
                    if "bytes" in data:
                        processed_audio = audio_processor.process_audio_chunk(data["bytes"])
                        if not openai_ready.is_set():
                            logger.debug("OpenAI not ready, buffering audio chunk")
                            pending_audio_chunks.append(processed_audio)
                        elif client:
                            # Track pending audio operations
                            async with audio_send_lock:
                                nonlocal pending_audio_operations
                                pending_audio_operations += 1
                                all_audio_sent.clear()  # Clear the event since we have pending operations
                            
                            try:
                                await client.send_audio(processed_audio)
                                await websocket.send_text(json.dumps({
                                    "type": "status",
                                    "status": "connected"
                                }))
                                logger.debug(f"Sent audio chunk, size: {len(processed_audio)} bytes")
                            finally:
                                # Mark operation as complete
                                async with audio_send_lock:
                                    pending_audio_operations -= 1
                                    if pending_audio_operations == 0:
                                        all_audio_sent.set()  # Set event when all operations complete
                        else:
                            logger.warning("Received audio but client is not initialized")
                            
                    elif "text" in data:
                        msg = json.loads(data["text"])

                        if msg.get("type") == "set_api_key":
                            # Handle API key setting
                            nonlocal user_api_key
                            user_api_key = msg.get("api_key")
                            logger.info("API key received from user")
                            await websocket.send_text(json.dumps({
                                "type": "api_key_status",
                                "status": "saved"
                            }))

                        elif msg.get("type") == "start_recording":
                            # Update status to connecting while initializing OpenAI
                            await websocket.send_text(json.dumps({
                                "type": "status",
                                "status": "connecting"
                            }))
                            if not await initialize_openai():
                                continue
                            recording_stopped.clear()
                            pending_audio_chunks.clear()
                            
                            # Send any buffered chunks
                            if pending_audio_chunks and client:
                                logger.info(f"Sending {len(pending_audio_chunks)} buffered chunks")
                                for chunk in pending_audio_chunks:
                                    # Track each buffered chunk operation
                                    async with audio_send_lock:
                                        pending_audio_operations += 1
                                        all_audio_sent.clear()
                                    
                                    try:
                                        await client.send_audio(chunk)
                                    finally:
                                        async with audio_send_lock:
                                            pending_audio_operations -= 1
                                            if pending_audio_operations == 0:
                                                all_audio_sent.set()
                                pending_audio_chunks.clear()
                            
                        elif msg.get("type") == "stop_recording":
                            if client:
                                # CRITICAL FIX: Wait for all pending audio operations to complete
                                # before committing to prevent data loss
                                logger.info("Stop recording received, waiting for all audio to be sent...")
                                
                                # Wait for any pending audio chunks to be sent (with timeout for safety)
                                try:
                                    await asyncio.wait_for(all_audio_sent.wait(), timeout=5.0)
                                    logger.info("All pending audio operations completed")
                                except asyncio.TimeoutError:
                                    logger.warning("Timeout waiting for audio operations to complete, proceeding anyway")
                                    # Reset the pending counter to prevent deadlock
                                    async with audio_send_lock:
                                        pending_audio_operations = 0
                                        all_audio_sent.set()
                                
                                # Add a small buffer to ensure network operations complete
                                await asyncio.sleep(0.1)
                                
                                logger.info("All audio sent, committing audio buffer...")
                                await client.commit_audio()
                                await client.start_response(PROMPTS['paraphrase-gpt-realtime-enhanced'])
                                await recording_stopped.wait()
                                # Don't close the client here, let the disconnect timer handle it
                                # Update client status to connected (waiting for response)
                                await websocket.send_text(json.dumps({
                                    "type": "status",
                                    "status": "connected"
                                }))

                except asyncio.TimeoutError:
                    logger.debug("No message received for 30 seconds")
                    continue
                except Exception as e:
                    logger.error(f"Error in receive_messages loop: {str(e)}", exc_info=True)
                    break
                
        finally:
            # Cleanup when the loop exits
            if client:
                try:
                    await client.close()
                except Exception as e:
                    logger.error(f"Error closing client in receive_messages: {str(e)}")
            logger.info("Receive messages loop ended")

    async def send_audio_messages():
        while True:
            try:
                processed_audio = await audio_queue.get()
                if processed_audio is None:
                    break
                
                # Add validation
                if len(processed_audio) == 0:
                    logger.warning("Empty audio chunk received, skipping")
                    continue
                
                # Append the processed audio to the buffer
                audio_buffer.append(processed_audio)

                await client.send_audio(processed_audio)
                logger.info(f"Audio chunk sent to OpenAI client, size: {len(processed_audio)} bytes")
                
            except Exception as e:
                logger.error(f"Error in send_audio_messages: {str(e)}", exc_info=True)
                break

        # After processing all audio, set the event
        recording_stopped.set()

    # Start concurrent tasks for receiving and sending
    receive_task = asyncio.create_task(receive_messages())
    send_task = asyncio.create_task(send_audio_messages())

    try:
        # Wait for both tasks to complete
        await asyncio.gather(receive_task, send_task)
    finally:
        if client:
            await client.close()
            logger.info("OpenAI client connection closed")

@app.post(
    "/api/v1/readability",
    response_model=ReadabilityResponse,
    summary="Enhance Text Readability",
    description="Improve the readability of the provided text using GPT-4."
)
async def enhance_readability(request: ReadabilityRequest):
    prompt = PROMPTS.get('readability-enhance')
    if not prompt:
        raise HTTPException(status_code=500, detail="Readability prompt not found.")

    try:
        # 根据用户选择使用 OpenAI 或 Compatible
        if request.llm_provider == "compatible":
            # Use OpenAI Compatible API (GLM, vLLM, Ollama, etc.)
            processor = get_llm_processor(request.compatible_model or "GLM-4.6-FP8",
                                         glm_host=request.compatible_base_url,
                                         glm_api_key=request.compatible_api_key)
            logger.info(f"Using compatible API with model {request.compatible_model} for readability enhancement")
        else:
            # Use OpenAI Official
            api_key = request.api_key or OPENAI_API_KEY
            if not api_key:
                raise HTTPException(status_code=400, detail="OpenAI API key is required.")
            model = request.openai_model or "gpt-4o"
            processor = get_llm_processor(model, api_key=api_key)
            logger.info(f"Using OpenAI {model} for readability enhancement")

        async def text_generator():
            async for part in processor.process_text(request.text, prompt):
                yield part

        return StreamingResponse(text_generator(), media_type="text/plain")

    except Exception as e:
        logger.error(f"Error enhancing readability: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing readability enhancement.")

@app.post(
    "/api/v1/ask_ai",
    response_model=AskAIResponse,
    summary="Ask AI a Question",
    description="Ask AI to provide insights using GLM or O1-mini model."
)
async def ask_ai(request: AskAIRequest):
    prompt = PROMPTS.get('ask-ai')
    if not prompt:
        raise HTTPException(status_code=500, detail="Ask AI prompt not found.")

    try:
        # 根据用户选择使用 OpenAI 或 Compatible
        if request.llm_provider == "compatible":
            # Use OpenAI Compatible API (GLM, vLLM, Ollama, etc.)
            processor = get_llm_processor(request.compatible_model or "GLM-4.6-FP8",
                                         glm_host=request.compatible_base_url,
                                         glm_api_key=request.compatible_api_key)
            logger.info(f"Using compatible API with model {request.compatible_model} for AI question")
        else:
            # Use OpenAI Official
            api_key = request.api_key or OPENAI_API_KEY
            if not api_key:
                raise HTTPException(status_code=400, detail="OpenAI API key is required.")
            model = request.openai_model or "gpt-4o"
            processor = get_llm_processor(model, api_key=api_key)
            logger.info(f"Using OpenAI {model} for AI question")

        async def text_generator():
            async for part in processor.process_text(request.text, prompt):
                yield part

        return StreamingResponse(text_generator(), media_type="text/plain")

    except Exception as e:
        logger.error(f"Error processing AI question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing AI question.")

@app.post(
    "/api/v1/correctness",
    response_model=CorrectnessResponse,
    summary="Check Factual Correctness",
    description="Analyze the text for factual accuracy using GPT-4o."
)
async def check_correctness(request: CorrectnessRequest):
    prompt = PROMPTS.get('correctness-check')
    if not prompt:
        raise HTTPException(status_code=500, detail="Correctness prompt not found.")

    try:
        # 根据用户选择使用 OpenAI 或 Compatible
        if request.llm_provider == "compatible":
            # Use OpenAI Compatible API (GLM, vLLM, Ollama, etc.)
            processor = get_llm_processor(request.compatible_model or "GLM-4.6-FP8",
                                         glm_host=request.compatible_base_url,
                                         glm_api_key=request.compatible_api_key)
            logger.info(f"Using compatible API with model {request.compatible_model} for correctness check")
        else:
            # Use OpenAI Official
            api_key = request.api_key or OPENAI_API_KEY
            if not api_key:
                raise HTTPException(status_code=400, detail="OpenAI API key is required.")
            model = request.openai_model or "gpt-4o"
            processor = get_llm_processor(model, api_key=api_key)
            logger.info(f"Using OpenAI {model} for correctness check")

        async def text_generator():
            async for part in processor.process_text(request.text, prompt):
                yield part

        return StreamingResponse(text_generator(), media_type="text/plain")

    except Exception as e:
        logger.error(f"Error checking correctness: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing correctness check.")

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=3005)
