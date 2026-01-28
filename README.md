# Pocket-TTS Wyoming Protocol Server (Streaming Optimized)

Wyoming protocol server for [Pocket-TTS](https://github.com/kyutai-labs/pocket-tts), enabling Home Assistant integration with voice selection support and **ultra-low latency streaming**.

This fork modifies the original implementation to support real-time audio streaming. Instead of waiting for a full sentence to be generated, audio chunks are sent to Home Assistant as they are created, significantly reducing the "Time to First Audio".

## ✨ Key Enhancements (Streaming Edition)

- **Real-time Audio Streaming:** Audio playback starts almost immediately.
- **`uv` Integration:** Uses the ultra-fast Python package installer for faster builds and cleaner environments.
- **Optimized Prefix Trimming:** Tuned logic for the sacrificial prefix (`...`) to prevent "swallowed" words without sacrificing streaming speed.
- **Unbuffered Logging:** Real-time visibility into the streaming process.

## 🚀 Quick Start with Docker Compose

To use the streaming features, you must build the image locally using the included `Dockerfile` and `wyoming_tts_server.py`.

```yaml
services:
  pocket-tts-wyoming:
    build: .
    container_name: pocket-tts-wyoming
    network_mode: host
    environment:
      - WYOMING_PORT=10201
      - WYOMING_HOST=0.0.0.0
      - DEFAULT_VOICE=alba
      - MODEL_VARIANT=b6369a24
      - ZEROCONF=pocket-tts
      - PYTHONUNBUFFERED=1
      # Tuning for faster response
      - PREFIX_MIN_DURATION=0.1
    restart: unless-stopped
    volumes:
      - pocket-tts-hf-cache:/data/hf

volumes:
  pocket-tts-hf-cache:
    driver: local
```

### Build and Start:

```bash
docker compose up -d --build
```

## ⚙️ Configuration

You can customize the following environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `WYOMING_PORT` | `10201` | The port the Wyoming protocol server listens on. |
| `DEFAULT_VOICE` | `alba` | The default voice used when none is specified. |
| `MODEL_VARIANT` | `b6369a24` | The Pocket-TTS model variant to use. |
| `ZEROCONF` | `pocket-tts` | Service name for mDNS/Zeroconf discovery. |
| `PYTHONUNBUFFERED` | `1` | Forces real-time log output (essential for monitoring streaming). |
| `PREFIX_MIN_DURATION` | `0.1` | Min. seconds to wait before trimming the "..." prefix. Lower = faster start. |

## 🗣️ Available Voices

`alba`, `marius`, `javert`, `jean`, `fantine`, `cosette`, `eponine`, `azelma`

## 🏠 Home Assistant Integration

The server supports Zeroconf/mDNS for automatic discovery.

1. Start the Docker container.
2. Go to **Settings** -> **Devices & Services** -> **Add Integration**.
3. Search for **Wyoming Protocol**.
4. The server should appear in the "Discovered" section, or enter `tcp://<server-ip>:10201` manually.
5. Configure a Voice Assistant pipeline to use the TTS service and select a voice.

## 🛠️ Streaming & Debug Mode

This implementation uses a continuous audio generator. It buffers just enough audio to detect the sacrificial prefix (`...`) and then immediately starts pushing chunks to the Wyoming client.

### Background on "Sacrificial Prefix"
Audio-prompt based models like Pocket-TTS can "swallow" the first word. We prepend `"..."` to all text and trim it from the result. This fork performs this trimming **on the fly**.

### Timing Tunables for Experts

| Variable | Default | Description |
|----------|---------|-------------|
| `PREFIX_MIN_DURATION` | `0.1` | Minimum seconds before looking for the pause after the prefix. |
| `PREFIX_MAX_DURATION` | `1.0` | Maximum seconds to search for the prefix end. |
| `PREFIX_SILENCE_GAP` | `0.08` | Minimum silence duration to identify the gap after the prefix. |

## 🔍 Troubleshooting

- **Slow startup:** First run downloads ~500MB of model weights. Ensure the volume mount for `/data/hf` is working.
- **Audio cuts off:** If the beginning of the sentence is missing, increase `PREFIX_MIN_DURATION` to `0.15` or `0.2`.
- **Latency issues:** Ensure your hardware (CPU/GPU) can generate audio faster than real-time. Check logs for `Generated: X ms of audio in Y ms`.
- **Logs not showing:** Check with `docker logs -f pocket-tts-wyoming`. You should see `DEBUG STREAM: Sending chunk...` during synthesis.

---
