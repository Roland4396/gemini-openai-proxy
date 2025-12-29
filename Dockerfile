FROM python:3.11-slim

WORKDIR /app

# Install Node.js and Gemini CLI
RUN apt-get update && apt-get install -y curl coreutils && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Gemini CLI
RUN npm install -g @google/gemini-cli

# Install Python dependencies
RUN pip3 install --no-cache-dir fastapi uvicorn httpx

COPY proxy.py .

EXPOSE 3000

CMD ["uvicorn", "proxy:app", "--host", "0.0.0.0", "--port", "3000"]
