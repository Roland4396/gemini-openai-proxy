#!/usr/bin/env python3
"""
Stream Proxy - 统一入口，路由分发
- 普通模型 → New API
- Gemini 模型 → gemini-proxy
- 非流转流（内部流式收集）
- 早停机制
"""

import os
import sys
import json
import uuid
import time
import threading
from typing import AsyncGenerator
from collections import deque
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
import docker

app = FastAPI()

# 上游服务配置
UPSTREAM_URL = os.environ.get('UPSTREAM_URL', 'http://127.0.0.1:3001')
GEMINI_PROXY_URL = os.environ.get('GEMINI_PROXY_URL', 'http://gemini-proxy:3003')
TIMEOUT = int(os.environ.get('TIMEOUT', '600'))
DEBUG = os.environ.get('DEBUG', 'true').lower() == 'true'
# 固定 API Key（连接 new-api 用）
FIXED_API_KEY = os.environ.get('FIXED_API_KEY', '')

# Docker 客户端
docker_client = None

def get_docker_client():
    """获取 Docker 客户端（懒加载）"""
    global docker_client
    if docker_client is None:
        try:
            docker_client = docker.from_env()
        except Exception as e:
            log(f"Failed to connect to Docker: {e}")
    return docker_client

def restart_sillytavern():
    """重启 SillyTavern 容器"""
    try:
        client = get_docker_client()
        if client:
            container = client.containers.get('sillytavern')
            container.restart(timeout=5)
            log("SillyTavern restarted")
    except Exception as e:
        log(f"Failed to restart SillyTavern: {e}")

def schedule_delayed_restart(delay: float = 2.0):
    """延迟重启 SillyTavern"""
    log(f"Scheduling restart in {delay}s")
    threading.Timer(delay, restart_sillytavern).start()

# 早停标签
STOP_TAGS = ["<!--ST0P_PROXY_", "<disclaimer>"]

# 请求/响应日志目录
LOG_DIR = "/app/logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 日志计数器（环形，保留最近10条）
LOG_COUNTER_FILE = os.path.join(LOG_DIR, "counter.txt")


def get_next_log_id() -> int:
    """获取下一个日志ID（1-10循环）"""
    try:
        with open(LOG_COUNTER_FILE, 'r') as f:
            counter = int(f.read().strip())
    except:
        counter = 0
    next_id = (counter % 10) + 1
    with open(LOG_COUNTER_FILE, 'w') as f:
        f.write(str(next_id))
    return next_id


def save_request_log(model: str, messages: list, response: str, stream: bool):
    """完整保存请求和响应到文件"""
    log_id = get_next_log_id()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存输入
    input_file = os.path.join(LOG_DIR, f"{log_id:02d}_input.json")
    with open(input_file, 'w', encoding='utf-8') as f:
        json.dump({
            "time": timestamp,
            "model": model,
            "stream": stream,
            "messages": messages
        }, f, ensure_ascii=False, indent=2)

    # 保存输出
    output_file = os.path.join(LOG_DIR, f"{log_id:02d}_output.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Time: {timestamp}\n")
        f.write(f"Model: {model}\n")
        f.write(f"Stream: {stream}\n")
        f.write(f"Length: {len(response)}\n")
        f.write("="*50 + "\n")
        f.write(response)

    log(f"Saved log #{log_id}: input={len(json.dumps(messages))} bytes, output={len(response)} bytes")


def find_stop_tag(text: str) -> int:
    """查找最早出现的停止标签位置，返回 -1 表示没找到"""
    positions = []
    for tag in STOP_TAGS:
        pos = text.find(tag)
        if pos != -1:
            positions.append(pos)
    return min(positions) if positions else -1


def has_stop_tag(text: str) -> bool:
    """检测文本是否包含任一停止标签"""
    return any(tag in text for tag in STOP_TAGS)


def log(msg: str):
    if DEBUG:
        print(f"[STREAM-PROXY] {msg}", file=sys.stderr, flush=True)


def is_gemini_model(model_name: str) -> bool:
    """判断是否是 Gemini 模型"""
    if not model_name:
        return False
    model_lower = model_name.lower()
    return model_lower.startswith('gemini-') or 'gemini' in model_lower


async def stream_with_early_stop(
    response_iter,
    parse_content_func
) -> AsyncGenerator[bytes, None]:
    """
    通用流式转发，带早停检测
    parse_content_func: 从 SSE data 中提取 content 的函数
    """
    content_buffer = ""

    async for line in response_iter:
        if not line:
            yield b'\n'
            continue

        line_str = line if isinstance(line, str) else line.decode('utf-8')

        if line_str.startswith('data: '):
            json_str = line_str[6:].strip()
            if json_str and json_str != '[DONE]':
                try:
                    data = json.loads(json_str)
                    content = parse_content_func(data)

                    if content:
                        content_buffer += content

                        # 早停检测
                        if has_stop_tag(content_buffer):
                            log(f"Stream: detected STOP_TAG")
                            prefix_pos = find_stop_tag(content_buffer)
                            prev_len = len(content_buffer) - len(content)

                            if prefix_pos >= prev_len:
                                keep_len = prefix_pos - prev_len
                                clean = content[:keep_len]
                            else:
                                clean = ''

                            if clean:
                                # 修改 data 中的 content
                                if 'choices' in data and data['choices']:
                                    if 'delta' in data['choices'][0]:
                                        data['choices'][0]['delta']['content'] = clean
                                    elif 'message' in data['choices'][0]:
                                        data['choices'][0]['message']['content'] = clean
                                yield f"data: {json.dumps(data)}\n\n".encode()

                            yield b"data: [DONE]\n\n"
                            return

                    # 检测 finish_reason
                    if 'choices' in data and data['choices']:
                        if data['choices'][0].get('finish_reason'):
                            yield f"{line_str}\n".encode()
                            return

                except json.JSONDecodeError:
                    pass

        yield f"{line_str}\n".encode()


def extract_openai_content(data: dict) -> str:
    """从 OpenAI 格式响应中提取 content"""
    if 'choices' in data and data['choices']:
        choice = data['choices'][0]
        delta = choice.get('delta', {})
        return delta.get('content', '')
    return ''


async def forward_stream(
    url: str,
    request_data: dict,
    headers: dict,
    enable_early_stop: bool = True,
    model: str = "",
    messages: list = None
) -> AsyncGenerator[bytes, None]:
    """流式转发请求，带早停检测（仅 Claude 需要，Gemini 跳过）"""
    content_buffer = ""
    full_response = ""  # 收集完整响应用于日志

    # 使用 Timeout 配置：connect=30秒，read=None（无限等待，由 gemini-proxy 心跳保活）
    timeout_config = httpx.Timeout(connect=30.0, read=None, write=30.0, pool=30.0)
    try:
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            async with client.stream(
                'POST',
                url,
                json=request_data,
                headers=headers
            ) as response:
                if response.status_code != 200:
                    content = await response.aread()
                    yield content
                    return

                async for line in response.aiter_lines():
                    if not line:
                        yield b'\n'
                        continue

                    if line.startswith('data: '):
                        json_str = line[6:].strip()
                        if json_str and json_str != '[DONE]':
                            try:
                                data = json.loads(json_str)
                                content = extract_openai_content(data)

                                if content:
                                    content_buffer += content
                                    full_response += content  # 收集完整响应

                                    # 早停检测（仅 Claude 需要）
                                    if enable_early_stop and has_stop_tag(content_buffer):
                                        log(f"Stream: detected STOP_TAG")
                                        prefix_pos = find_stop_tag(content_buffer)
                                        prev_len = len(content_buffer) - len(content)

                                        if prefix_pos >= prev_len:
                                            keep_len = prefix_pos - prev_len
                                            clean = content[:keep_len]
                                        else:
                                            clean = ''

                                        if clean:
                                            data['choices'][0]['delta']['content'] = clean
                                            full_response = full_response[:len(full_response)-len(content)+len(clean)]
                                            yield f"data: {json.dumps(data)}\n\n".encode()

                                        yield b"data: [DONE]\n\n"
                                        # 保存日志
                                        if messages is not None:
                                            save_request_log(model, messages, full_response, stream=True)
                                        return

                                # 检测 finish_reason
                                if 'choices' in data and data['choices']:
                                    if data['choices'][0].get('finish_reason'):
                                        yield f"{line}\n".encode()
                                        # 保存日志
                                        if messages is not None:
                                            save_request_log(model, messages, full_response, stream=True)
                                        return

                            except json.JSONDecodeError:
                                pass

                    yield f"{line}\n".encode()
    except Exception as e:
        log(f"Stream forward error: {e}")
        # 重启 SillyTavern 清除前端卡住状态
        restart_sillytavern()


async def collect_stream(
    client: httpx.AsyncClient,
    url: str,
    request_data: dict,
    headers: dict,
    enable_early_stop: bool = True
) -> tuple[str, str, dict]:
    """
    收集流式响应，返回完整内容（非流转流）
    返回: (content, model, usage)
    """
    full_content = ""
    model_name = ""
    usage = {}
    finish_reason = None

    # 强制流式请求
    request_data = request_data.copy()
    request_data['stream'] = True

    async with client.stream(
        'POST',
        url,
        json=request_data,
        headers=headers
    ) as response:
        if response.status_code != 200:
            content = await response.aread()
            raise Exception(f"Upstream error: {response.status_code} - {content.decode()}")

        async for line in response.aiter_lines():
            if not line:
                continue

            if line.startswith('data: '):
                line = line[6:]

            if line == '[DONE]':
                break

            try:
                data = json.loads(line)

                if 'model' in data:
                    model_name = data['model']

                if 'choices' in data and data['choices']:
                    choice = data['choices'][0]

                    # 提取 delta content
                    delta = choice.get('delta', {})
                    if 'content' in delta:
                        full_content += delta['content']

                        # 早停检测（仅 Claude 需要，Gemini 跳过）
                        if enable_early_stop and has_stop_tag(full_content):
                            log(f"Collect: detected STOP_TAG")
                            full_content = full_content[:find_stop_tag(full_content)]
                            break

                    # 提取 message content（某些情况）
                    message = choice.get('message', {})
                    if 'content' in message and message['content']:
                        full_content += message['content']
                        if enable_early_stop and has_stop_tag(full_content):
                            full_content = full_content[:find_stop_tag(full_content)]
                            break

                    if choice.get('finish_reason'):
                        finish_reason = choice['finish_reason']
                        break

                if 'usage' in data:
                    usage = data['usage']

            except json.JSONDecodeError:
                continue

    return full_content, model_name, usage


@app.post('/v1/chat/completions')
async def chat_completions(request: Request):
    try:
        data = await request.json()
        model = data.get('model', '')
        stream = data.get('stream', False)

        # 使用固定 API Key
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {FIXED_API_KEY}'
        }

        # 决定转发目标
        if is_gemini_model(model):
            target_url = f"{GEMINI_PROXY_URL}/v1/chat/completions"
            log(f"Route to Gemini: {model}")
        else:
            target_url = f"{UPSTREAM_URL}/v1/chat/completions"
            log(f"Route to NewAPI: {model}")

        if stream:
            # 流式请求，透传（Gemini 不需要早停，Claude 需要）
            data['stream'] = True
            enable_early_stop = not is_gemini_model(model)
            return StreamingResponse(
                forward_stream(target_url, data, headers, enable_early_stop, model, data.get('messages', [])),
                media_type='text/event-stream'
            )

        # 非流式请求
        timeout_config = httpx.Timeout(connect=30.0, read=None, write=30.0, pool=30.0)
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            try:
                # 使用流式收集（Gemini 不需要早停，Claude 需要）
                enable_early_stop = not is_gemini_model(model)
                full_content, model_name, usage = await collect_stream(
                    client, target_url, data, headers, enable_early_stop
                )

                # 保存日志
                save_request_log(model, data.get('messages', []), full_content, stream=False)

                return JSONResponse({
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_name or model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": full_content
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": usage or {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                })
            except Exception as e:
                log(f"Collect error: {e}")
                # 重启 SillyTavern 清除前端卡住状态
                restart_sillytavern()
                return JSONResponse(
                    {"error": {"message": str(e), "type": "upstream_error"}},
                    status_code=502
                )

    except httpx.TimeoutException:
        log("Request timeout")
        return JSONResponse(
            {"error": {"message": "Upstream timeout", "type": "timeout"}},
            status_code=504
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
async def models(request: Request):
    """合并上游和 Gemini 模型列表"""
    headers = {'Authorization': f'Bearer {FIXED_API_KEY}'}

    gemini_models = [
        {"id": "gemini-3-pro-preview", "object": "model", "owned_by": "google"},
        {"id": "gemini-3-flash-preview", "object": "model", "owned_by": "google"},
    ]

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{UPSTREAM_URL}/v1/models",
                headers=headers,
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                if 'data' in data:
                    data['data'] = gemini_models + data['data']
                return JSONResponse(data)
            else:
                return JSONResponse({"object": "list", "data": gemini_models})
    except Exception:
        return JSONResponse({"object": "list", "data": gemini_models})


@app.get('/health')
async def health():
    return JSONResponse({"status": "ok"})


if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('PORT', '3002'))
    uvicorn.run(app, host='0.0.0.0', port=port)
