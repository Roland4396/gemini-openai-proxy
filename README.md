# Gemini OpenAI Proxy

A proxy server that converts OpenAI API format to Google Gemini API, using the official Gemini CLI for better compatibility and fewer content restrictions.

## Features

- **OpenAI-compatible API** - Drop-in replacement for OpenAI API endpoints
- **Gemini CLI Backend** - Uses official Google Gemini CLI for requests
- **Streaming Support** - Server-sent events (SSE) streaming responses
- **Auto Retry** - Automatic retry on 429 (rate limit) errors
- **Heartbeat** - Connection keepalive for long-running requests
- **Docker Ready** - Easy deployment with Docker Compose

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Roland4396/gemini-openai-proxy.git
cd gemini-openai-proxy
```

### 2. Set environment variables

```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

### 3. Run with Docker Compose

```bash
docker compose up -d
```

The proxy will be available at `http://localhost:3000`

## Usage

### Chat Completions

```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-2.5-flash",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "stream": false
  }'
```

### Streaming

```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-2.5-flash",
    "messages": [
      {"role": "user", "content": "Tell me a story"}
    ],
    "stream": true
  }'
```

### List Models

```bash
curl http://localhost:3000/v1/models
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `GEMINI_API_KEY` | (required) | Your Gemini API key |
| `GOOGLE_GEMINI_BASE_URL` | `https://generativelanguage.googleapis.com` | Gemini API base URL |
| `PORT` | `3000` | Server port |
| `TIMEOUT` | `600` | Request timeout in seconds |
| `MAX_RETRIES` | `5` | Max retry attempts on 429 errors |
| `RETRY_DELAY` | `1` | Delay between retries in seconds |
| `DEBUG` | `false` | Enable debug logging |

## Supported Models

- `gemini-3-pro-preview` (Latest)
- `gemini-3-flash-preview` (Latest)
- `gemini-2.5-pro`
- `gemini-2.5-flash`
- `gemini-2.0-flash`

You can use any model name supported by the Gemini CLI.

## Integration Examples

### SillyTavern

1. Go to API Connections
2. Select "Chat Completion" API type
3. Set Custom Endpoint: `http://localhost:3000/v1`
4. Select model: `gemini-2.5-flash`

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    api_key="dummy",  # Not used, but required
    base_url="http://localhost:3000/v1"
)

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

## Why Gemini CLI?

This proxy uses the official Gemini CLI instead of direct HTTP calls because:

1. **Better Compatibility** - CLI handles authentication and request formatting automatically
2. **Fewer Restrictions** - CLI requests may have different content filtering policies
3. **Official Support** - Uses Google's official tooling

## License

MIT License
