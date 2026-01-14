# Gemini OpenAI Proxy

A proxy solution that converts OpenAI API format to Google Gemini API, with unified routing and stream conversion capabilities.

## Components

This project includes two proxy components:

| Component | File | Port | Description |
|-----------|------|------|-------------|
| **Gemini Proxy** | `proxy.py` | 3000 | Direct Gemini CLI wrapper with OpenAI-compatible API |
| **Stream Proxy** | `stream-proxy.py` | 3002 | Unified router with non-stream to stream conversion |

## Features

### Gemini Proxy (`proxy.py`)
- **OpenAI-compatible API** - Drop-in replacement for OpenAI API endpoints
- **Gemini CLI Backend** - Uses official Google Gemini CLI for requests
- **Streaming Support** - Server-sent events (SSE) streaming responses
- **Auto Retry** - Automatic retry on 429 (rate limit) errors
- **Heartbeat** - Connection keepalive for long-running requests

### Stream Proxy (`stream-proxy.py`)
- **Unified Router** - Route requests to different backends based on model
- **Non-Stream to Stream** - Convert non-streaming requests to streaming internally
- **Early Stop** - Detect stop tags and terminate response early
- **Multi-Backend** - Gemini models → Gemini Proxy, Other models → New-API
- **Request Logging** - Save request/response logs for debugging

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌───────────────┐     ┌─────────────┐
│   Client    │────▶│   Stream Proxy   │────▶│ Gemini Proxy  │────▶│ Gemini CLI  │
│ (SillyTavern)│     │   (Port 3002)    │     │  (Port 3000)  │     │             │
└─────────────┘     └──────────────────┘     └───────────────┘     └─────────────┘
                            │
                            │ (non-gemini models)
                            ▼
                    ┌───────────────┐
                    │    New-API    │
                    │  (Port 3001)  │
                    └───────────────┘
```

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Roland4396/gemini-openai-proxy.git
cd gemini-openai-proxy
```

### 2. Set environment variables

```bash
# For Gemini Proxy
export GEMINI_API_KEY="your-gemini-api-key"

# For Stream Proxy (optional)
export FIXED_API_KEY="your-new-api-key"
export UPSTREAM_URL="http://new-api:3001"
export GEMINI_PROXY_URL="http://gemini-proxy:3000"
```

### 3. Run with Docker Compose

```bash
docker compose up -d
```

## Configuration

### Gemini Proxy

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `GEMINI_API_KEY` | (required) | Your Gemini API key |
| `GOOGLE_GEMINI_BASE_URL` | `https://generativelanguage.googleapis.com` | Gemini API base URL |
| `PORT` | `3000` | Server port |
| `TIMEOUT` | `600` | Request timeout in seconds |
| `MAX_RETRIES` | `5` | Max retry attempts on 429 errors |
| `RETRY_DELAY` | `1` | Delay between retries in seconds |
| `DEBUG` | `false` | Enable debug logging |

### Stream Proxy

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `UPSTREAM_URL` | `http://127.0.0.1:3001` | New-API endpoint for non-Gemini models |
| `GEMINI_PROXY_URL` | `http://gemini-proxy:3003` | Gemini Proxy endpoint |
| `FIXED_API_KEY` | (required) | API key for New-API authentication |
| `TIMEOUT` | `600` | Request timeout in seconds |
| `DEBUG` | `true` | Enable debug logging |

## Supported Models

- `gemini-3-pro-preview` (Latest)
- `gemini-3-flash-preview` (Latest)
- `gemini-2.5-pro`
- `gemini-2.5-flash`
- `gemini-2.0-flash`

Plus any models available through your New-API instance.

## Usage

### Direct Gemini Proxy

```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-2.5-flash",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

### Via Stream Proxy (Recommended)

```bash
curl http://localhost:3002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-2.5-flash",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

The Stream Proxy will automatically:
1. Route Gemini models to Gemini Proxy
2. Route other models to New-API
3. Convert non-stream requests to stream internally
4. Handle early stopping on special tags

## Integration Examples

### With New-API

Use [new-api](https://github.com/Calcium-Ion/new-api) as a backend for non-Gemini models:

1. Deploy new-api and both proxy components
2. Configure Stream Proxy environment:
   - `UPSTREAM_URL`: Your new-api endpoint
   - `GEMINI_PROXY_URL`: Your gemini-proxy endpoint
   - `FIXED_API_KEY`: Your new-api access key
3. Point your client to Stream Proxy (port 3002)

### SillyTavern

1. Go to API Connections
2. Select "Chat Completion" API type
3. Set Custom Endpoint: `http://localhost:3002/v1` (Stream Proxy)
4. Select model: `gemini-2.5-flash` or any supported model

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    api_key="dummy",  # Not used for Gemini, required for SDK
    base_url="http://localhost:3002/v1"
)

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

## Docker Compose Example

```yaml
services:
  gemini-proxy:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}

  stream-proxy:
    build:
      context: .
      dockerfile: Dockerfile.stream
    ports:
      - "3002:3002"
    environment:
      - UPSTREAM_URL=http://new-api:3001
      - GEMINI_PROXY_URL=http://gemini-proxy:3000
      - FIXED_API_KEY=${FIXED_API_KEY}
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
```

## Why This Architecture?

1. **Non-Stream to Stream** - Many clients send non-streaming requests, but streaming internally prevents timeout issues on long responses
2. **Unified Entry Point** - Single endpoint for all models, automatic routing
3. **Gemini CLI Backend** - Official Google tooling with better compatibility
4. **Early Stop** - Detect special tags and stop generation early to save tokens
5. **Heartbeat** - Keep connections alive during long Gemini CLI processing

## License

MIT License
