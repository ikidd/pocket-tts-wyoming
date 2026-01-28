# Pocket-TTS Wyoming Protocol Server (Streaming Optimized)

A high-performance, **low-latency streaming** Wyoming protocol server for [Pocket-TTS](https://github.com/kyutai-labs/pocket-tts). This version is optimized to start playback almost immediately by streaming audio chunks to Home Assistant as they are generated, rather than waiting for the full sentence to complete.

Built with `uv` for fast dependency management and optimized Docker layers.

## ✨ Key Features

- **Real-time Streaming:** Sends audio chunks immediately (Chunked Transfer).
- **Reduced Latency:** Optimized prefix trimming logic (`...` handling) to minimize wait time.
- **Smart Caching:** Uses HuggingFace cache to prevent re-downloads.
- **Home Assistant Auto-Discovery:** Supports Zeroconf/mDNS.

## 🚀 Quick Start with Docker Compose

Create a `docker-compose.yml` file in the same directory as the `Dockerfile` and `wyoming_tts_server.py`:

```yaml
services:
  pocket-tts-wyoming:
    build: .
    container_name: pocket-tts-wyoming
    network_mode: host
    environment:
      - WYOMING_PORT=10201
      - DEFAULT_VOICE=alba
      - MODEL_VARIANT=b6369a24
      - ZEROCONF=pocket-tts
      
      # --- Performance Tuning ---
      # Force unbuffered logs to see streaming in real-time
      - PYTHONUNBUFFERED=1
      # Aggressiveness of the "Sacrificial Prefix" trimming (lower = faster start)
      - PREFIX_MIN_DURATION=0.1
      
      # Cache location (matches internal Docker path)
      - HF_HOME=/data/hf

    restart: unless-stopped
    volumes:
      - pocket-tts-hf-cache:/data/hf

volumes:
  pocket-tts-hf-cache:
    driver: local
Build and Run
Since this version uses custom optimizations, you need to build the image locally:
code
Bash
docker compose up -d --build
To view the logs and confirm streaming is working (look for DEBUG STREAM entries):
code
Bash
docker logs -f pocket-tts-wyoming
⚙️ Configuration
You can customize the following environment variables in docker-compose.yml:
Variable	Default	Description
WYOMING_PORT	10201	The port the server listens on.
DEFAULT_VOICE	alba	Default voice if none specified.
MODEL_VARIANT	b6369a24	The Pocket-TTS model checkpoint.
PYTHONUNBUFFERED	1	Important: Keeps logs real-time. Essential for debugging streaming.
PREFIX_MIN_DURATION	0.1	Min seconds to wait before trimming the "..." prefix. Lower = faster start, but risk of hearing artifacts.
DEBUG_WAV	false	Set to true to save generated audio files to /output for debugging.
HF_HOME	/data/hf	Internal path for model caching.
🗣️ Available Voices
alba, marius, javert, jean, fantine, cosette, eponine, azelma
🏠 Home Assistant Integration
Go to Settings -> Devices & Services -> Add Integration.
Search for Wyoming Protocol.
Auto-Discovery: It should appear automatically if on the same network.
Manual: Enter tcp://<YOUR_IP>:10201 (use the IP of your Docker host).
Configure your Voice Assistant Pipeline to use this TTS service.
🛠️ How it works (The Streaming Logic)
Standard Pocket-TTS implementations often generate the full audio for a sentence, trim the silence, and then send it. This causes a delay of 1-3 seconds per sentence.
This implementation:
Generates audio in a continuous stream.
Buffers only the first few milliseconds to detect and remove the sacrificial prefix (...).
Immediately pushes subsequent audio chunks to Home Assistant via the Wyoming protocol.
Result: Audio starts playing while the GPU/CPU is still generating the rest of the sentence.
🐛 Troubleshooting
Logs don't show streaming: Ensure PYTHONUNBUFFERED=1 is set in your compose file.
Start is too slow: Try lowering PREFIX_MIN_DURATION to 0.08 or 0.05.
First word sounds cut off: Increase PREFIX_MIN_DURATION to 0.15 or 0.2.
Permission errors: If you cannot write to volumes, ensure the container has root access or correct PUID/PGID (default runs as root inside the container).
🏗️ Manual Build (without Compose)
code
Bash
docker build -t pocket-tts-streaming .
docker run -d \
  --name pocket-tts \
  --net=host \
  -e PYTHONUNBUFFERED=1 \
  -v pocket-tts-hf-cache:/data/hf \
  pocket-tts-streaming
