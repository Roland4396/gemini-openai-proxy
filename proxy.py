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
DEBUG = os.environ.get('DEBUG', 'true').lower() == 'true'
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

    # 保存发送给CLI的完整prompt（不做任何处理）
    raw_save_dir = "/app/logs"
    os.makedirs(raw_save_dir, exist_ok=True)
    prompt_filename = f"{raw_save_dir}/sent_prompt_{int(time.time())}_{retry_count}.txt"
    try:
        with open(prompt_filename, 'w', encoding='utf-8') as f:
            f.write(prompt)
        log(f"[PROMPT SAVED] {prompt_filename} ({len(prompt)} chars)")
    except Exception as e:
        log(f"[PROMPT SAVE ERROR] {e}")

    # Write to temp file to avoid command line length limits
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(prompt)
        prompt_file = f.name

    # 用临时文件存储输出，避免管道缓冲区限制
    output_file = f"/tmp/gemini_output_{int(time.time())}_{retry_count}.json"

    process = None
    try:
        # 使用shell重定向，输出到文件，stderr丢弃
        cmd = f'gemini -m {model} --output-format json < "{prompt_file}" > "{output_file}" 2>/dev/null'
        log(f"Command: {cmd}")

        process = await asyncio.create_subprocess_shell(
            cmd,
            env=env
        )
        log(f"Process started, pid={process.pid}")

        # 等待进程完成
        try:
            await asyncio.wait_for(process.wait(), timeout=TIMEOUT)
        except asyncio.TimeoutError:
            log("Gemini CLI timeout")
            process.kill()
            raise Exception("Gemini CLI timeout")

        log(f"[EXIT] code={process.returncode}")

        # 从文件读取完整输出
        with open(output_file, 'rb') as f:
            raw_bytes = f.read()
        buffer = raw_bytes.decode('utf-8', errors='replace')

        log(f"[OUTPUT FILE] {output_file} ({len(raw_bytes)} bytes)")

        # Parse complete JSON output
        log(f"gemini-cli: parsing buffer: {len(buffer)} chars")

        # 保存CLI返回的完整原始字节（不做任何处理）
        raw_save_dir = "/app/logs"
        os.makedirs(raw_save_dir, exist_ok=True)
        raw_filename = f"{raw_save_dir}/raw_bytes_{int(time.time())}_{retry_count}.txt"
        try:
            with open(raw_filename, 'wb') as f:
                f.write(raw_bytes)
            log(f"[RAW BYTES SAVED] {raw_filename} ({len(raw_bytes)} bytes)")
        except Exception as e:
            log(f"[RAW SAVE ERROR] {e}")

        # 统一重试函数
        async def do_retry(reason: str):
            delay = RETRY_DELAY * (retry_count + 1)
            log(f"gemini-cli: {reason}, retrying in {delay}s (attempt {retry_count + 1}/{MAX_RETRIES})")
            try:
                os.unlink(prompt_file)
            except:
                pass
            await asyncio.sleep(delay)
            async for c in call_gemini_cli_stream(prompt, model, retry_count + 1):
                yield c

        # 1. 检查进程退出码
        if process.returncode != 0 and process.returncode is not None:
            log(f"gemini-cli: process failed with exit code {process.returncode}")
            if retry_count < MAX_RETRIES:
                async for c in do_retry(f"exit code {process.returncode}"):
                    yield c
                return

        # 2. 检测 429 错误
        is_429_error = '429' in buffer or 'exhausted' in buffer.lower() or 'quota' in buffer.lower()
        if is_429_error and retry_count < MAX_RETRIES:
            async for c in do_retry("429 error"):
                yield c
            return

        # 3. 解析 JSON
        try:
            data = json.loads(buffer.strip())
            if 'response' in data:
                content = data['response']
                log(f"gemini-cli: extracted response, length={len(content)}")
                if content:
                    yield content
                elif retry_count < MAX_RETRIES:
                    async for c in do_retry("empty response"):
                        yield c
                    return
            else:
                log(f"gemini-cli: no 'response' field in JSON, keys={list(data.keys())}")
                if retry_count < MAX_RETRIES:
                    async for c in do_retry("no response field"):
                        yield c
                    return
        except json.JSONDecodeError as e:
            log(f"gemini-cli: JSON parse error: {e}")
            log(f"gemini-cli: raw buffer: {buffer[:1000]}")
            if retry_count < MAX_RETRIES:
                async for c in do_retry("JSON parse error"):
                    yield c
                return

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
        try:
            os.unlink(output_file)
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
            {"id": "gemini-3-pro-preview", "object": "model", "owned_by": "google"},
            {"id": "gemini-3-flash-preview", "object": "model", "owned_by": "google"},
            {"id": "gemini-2.5-pro", "object": "model", "owned_by": "google"},
            {"id": "gemini-2.5-flash", "object": "model", "owned_by": "google"},
            {"id": "gemini-2.0-flash", "object": "model", "owned_by": "google"},
        ]
    })


@app.get('/health')
async def health():
    return JSONResponse({"status": "ok"})


if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('PORT', '3003'))
    uvicorn.run(app, host='0.0.0.0', port=port)
