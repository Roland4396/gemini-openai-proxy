#!/usr/bin/env python3
"""
Gemini OpenAI Proxy - Convert OpenAI API format to Gemini CLI
- Receives OpenAI format requests
- Calls gemini-cli command line tool
- Returns OpenAI compatible responses
- Supports streaming and non-streaming modes
- Auto-retry on 429 errors
"""

import os
import sys
import json
import uuid
import time
import asyncio
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()

# Configuration
TIMEOUT = int(os.environ.get('TIMEOUT', '600'))
DEBUG = os.environ.get('DEBUG', 'false').lower() == 'true'
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
GOOGLE_GEMINI_BASE_URL = os.environ.get('GOOGLE_GEMINI_BASE_URL', 'https://generativelanguage.googleapis.com')

# Retry configuration for 429 errors
MAX_RETRIES = int(os.environ.get('MAX_RETRIES', '5'))
RETRY_DELAY = int(os.environ.get('RETRY_DELAY', '1'))


def log(msg: str):
    if DEBUG:
        print(f"[GEMINI-PROXY] {msg}", file=sys.stderr, flush=True)


def build_prompt_from_messages(messages: list) -> str:
    """Convert OpenAI format messages to a single prompt"""
    parts = []
    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')

        # Handle multimodal content (array format)
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get('type') == 'text':
                    text_parts.append(item.get('text', ''))
                elif isinstance(item, str):
                    text_parts.append(item)
            content = '\n'.join(text_parts)

        if role == 'system':
            parts.append(f"[System]\n{content}")
        elif role == 'user':
            parts.append(f"[User]\n{content}")
        elif role == 'assistant':
            parts.append(f"[Assistant]\n{content}")

    return "\n\n".join(parts)


def clear_gemini_sessions():
    """Clear gemini CLI session cache to avoid reusing old responses"""
    import shutil
    gemini_tmp = os.path.expanduser("~/.gemini/tmp")
    if os.path.exists(gemini_tmp):
        for item in os.listdir(gemini_tmp):
            item_path = os.path.join(gemini_tmp, item)
            if os.path.isdir(item_path):
                chats_path = os.path.join(item_path, "chats")
                if os.path.exists(chats_path):
                    try:
                        shutil.rmtree(chats_path)
                        log(f"Cleared session cache: {chats_path}")
                    except Exception as e:
                        log(f"Failed to clear cache: {e}")


async def call_gemini_cli_stream(prompt: str, model: str, retry_count: int = 0) -> AsyncGenerator[str, None]:
    """
    Stream call to Gemini CLI with chunk reading to avoid truncation
    Supports auto-retry on 429 errors
    Yields: content chunks (None for heartbeat)
    """
    import tempfile

    # Clear old sessions
    clear_gemini_sessions()

    env = os.environ.copy()
    env['GEMINI_API_KEY'] = GEMINI_API_KEY
    env['GOOGLE_GEMINI_BASE_URL'] = GOOGLE_GEMINI_BASE_URL

    log(f"Calling gemini-cli: model={model}, prompt_len={len(prompt)}, retry={retry_count}")

    # Write to temp file to avoid command line length limits
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(prompt)
        prompt_file = f.name

    process = None
    try:
        cmd = ['gemini', '-m', model, '--output-format', 'json']
        log(f"Command: {' '.join(cmd)}")

        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_content = f.read()

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )
        log(f"Process started, pid={process.pid}")

        # Concurrent stdin write to avoid deadlock
        async def write_stdin():
            try:
                data = prompt_content.encode('utf-8')
                total_len = len(data)
                written = 0
                WRITE_CHUNK = 8192

                while written < total_len:
                    chunk = data[written:written + WRITE_CHUNK]
                    process.stdin.write(chunk)
                    await process.stdin.drain()
                    written += len(chunk)

                process.stdin.close()
                await process.stdin.wait_closed()
                log(f"Stdin: DONE, total written={written} bytes")
            except Exception as e:
                log(f"Stdin write error: {e}")

        stdin_task = asyncio.create_task(write_stdin())

        # Concurrent stderr reading
        async def read_stderr():
            stderr_data = b''
            while True:
                try:
                    chunk = await asyncio.wait_for(process.stderr.read(1024), timeout=1)
                    if not chunk:
                        break
                    stderr_data += chunk
                    if DEBUG:
                        log(f"Stderr: {chunk.decode('utf-8', errors='replace')[:200]}")
                except asyncio.TimeoutError:
                    continue
                except:
                    break
            return stderr_data

        stderr_task = asyncio.create_task(read_stderr())

        # Chunk reading with buffer handling
        buffer = ""
        pending_bytes = b''
        chunk_count = 0
        CHUNK_SIZE = 4096
        READ_TIMEOUT = 10
        total_wait = 0

        while True:
            try:
                chunk = await asyncio.wait_for(
                    process.stdout.read(CHUNK_SIZE),
                    timeout=READ_TIMEOUT
                )
                total_wait = 0
            except asyncio.TimeoutError:
                total_wait += READ_TIMEOUT
                if total_wait >= TIMEOUT:
                    log("Gemini CLI total timeout")
                    break
                # Send heartbeat signal
                yield None
                continue

            if not chunk:
                log(f"gemini-cli: EOF reached after {chunk_count} chunks")
                break

            chunk_count += 1
            chunk = pending_bytes + chunk
            pending_bytes = b''

            # Handle UTF-8 multi-byte character truncation
            try:
                chunk_str = chunk.decode('utf-8')
            except UnicodeDecodeError:
                for i in range(1, 4):
                    try:
                        chunk_str = chunk[:-i].decode('utf-8')
                        pending_bytes = chunk[-i:]
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    chunk_str = chunk.decode('utf-8', errors='replace')
                    pending_bytes = b''
            buffer += chunk_str

        # Parse complete JSON output
        log(f"gemini-cli: parsing buffer: {len(buffer)} chars")

        # Detect 429 error
        is_429_error = '429' in buffer or 'exhausted' in buffer.lower() or 'quota' in buffer.lower()

        if is_429_error and retry_count < MAX_RETRIES:
            log(f"gemini-cli: detected 429 error, retrying in {RETRY_DELAY}s (attempt {retry_count + 1}/{MAX_RETRIES})")
            if process and process.returncode is None:
                process.kill()
                await process.wait()
            try:
                os.unlink(prompt_file)
            except:
                pass
            await asyncio.sleep(RETRY_DELAY)
            async for chunk in call_gemini_cli_stream(prompt, model, retry_count + 1):
                yield chunk
            return

        try:
            data = json.loads(buffer.strip())
            if 'response' in data:
                content = data['response']
                log(f"gemini-cli: extracted response, length={len(content)}")
                if content:
                    yield content
            else:
                log(f"gemini-cli: no 'response' field in JSON, keys={list(data.keys())}")
        except json.JSONDecodeError as e:
            log(f"gemini-cli: JSON parse error: {e}")
            log(f"gemini-cli: raw buffer: {buffer[:1000]}")

    except asyncio.TimeoutError:
        log("Gemini CLI timeout")
        if process:
            process.kill()
        raise Exception("Gemini CLI timeout")
    finally:
        if process and process.returncode is None:
            process.kill()
            await process.wait()
        try:
            os.unlink(prompt_file)
        except:
            pass


async def call_gemini_cli(prompt: str, model: str) -> str:
    """
    Non-streaming call to Gemini CLI (internally uses streaming)
    Returns: complete content
    """
    log(f"Gemini CLI request (non-stream): model={model}")

    full_content = ""

    try:
        async for chunk in call_gemini_cli_stream(prompt, model):
            if chunk is None:
                continue
            if chunk:
                full_content += chunk
    except Exception as e:
        log(f"Gemini CLI error: {e}")
        raise

    return full_content


async def gemini_stream_generator(
    prompt: str,
    model: str,
    response_id: str,
    created: int
) -> AsyncGenerator[bytes, None]:
    """Generate Gemini streaming response in OpenAI format with heartbeat"""
    try:
        async for chunk in call_gemini_cli_stream(prompt, model):
            if chunk is None:
                yield b": heartbeat\n\n"
                log("Sent heartbeat")
                continue

            if chunk:
                resp_chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(resp_chunk)}\n\n".encode()

        # Send end marker
        end_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(end_chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    except Exception as e:
        log(f"Gemini stream error: {e}")
        error_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": f"\n\n[Error: {str(e)}]"},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(error_chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"


@app.post('/v1/chat/completions')
async def chat_completions(request: Request):
    try:
        data = await request.json()
        model = data.get('model', 'gemini-2.5-flash')
        stream = data.get('stream', False)
        messages = data.get('messages', [])

        if not messages:
            return JSONResponse(
                {"error": {"message": "messages is required"}},
                status_code=400
            )

        prompt = build_prompt_from_messages(messages)
        response_id = f"chatcmpl-{uuid.uuid4()}"
        created = int(time.time())

        log(f"Request: model={model}, stream={stream}, prompt_len={len(prompt)}")

        if stream:
            return StreamingResponse(
                gemini_stream_generator(prompt, model, response_id, created),
                media_type='text/event-stream'
            )
        else:
            try:
                response_text = await call_gemini_cli(prompt, model)

                return JSONResponse({
                    "id": response_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_text
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": len(prompt) // 4,
                        "completion_tokens": len(response_text) // 4,
                        "total_tokens": (len(prompt) + len(response_text)) // 4
                    }
                })
            except Exception as e:
                log(f"Gemini CLI error: {e}")
                return JSONResponse(
                    {"error": {"message": str(e), "type": "gemini_error"}},
                    status_code=500
                )

    except Exception as e:
        log(f"Exception: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            {"error": {"message": str(e), "type": "proxy_error"}},
            status_code=500
        )


@app.get('/v1/models')
async def models():
    """Return supported Gemini models"""
    return JSONResponse({
        "object": "list",
        "data": [
            {"id": "gemini-2.5-pro", "object": "model", "owned_by": "google"},
            {"id": "gemini-2.5-flash", "object": "model", "owned_by": "google"},
            {"id": "gemini-2.0-flash", "object": "model", "owned_by": "google"},
            {"id": "gemini-1.5-pro", "object": "model", "owned_by": "google"},
            {"id": "gemini-1.5-flash", "object": "model", "owned_by": "google"},
        ]
    })


@app.get('/health')
async def health():
    return JSONResponse({"status": "ok"})


if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('PORT', '3000'))
    uvicorn.run(app, host='0.0.0.0', port=port)
