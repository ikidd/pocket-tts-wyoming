# Pocket-TTS Wyoming Protocol Server

Wyoming protocol server for [Pocket-TTS](https://github.com/kyutai-labs/pocket-tts), enabling Home Assistant integration with voice selection support.

## Changes in this fork

This fork of [ikidd/pocket-tts-wyoming](https://github.com/ikidd/pocket-tts-wyoming) contains two fixes:

- **Streaming audio to Home Assistant.** The upstream server drained the entire Pocket-TTS generator and concatenated the waveform in memory before emitting any `AudioChunk` events. On CPUs without NNPACK, longer inputs could take tens of seconds, during which Home Assistant's Wyoming client timed out and dropped the socket (`ConnectionResetError` / `BrokenPipeError`), so long text appeared to "not generate audio." The server now buffers only the short head needed to resolve the sacrificial `...` prefix-trim window (`PREFIX_MAX_DURATION`, ~1s), then forwards chunks to HA as they are produced. First chunk now arrives in under a second even for multi-sentence inputs.
- **Pinned upstream Pocket-TTS commit.** `kyutai-labs/pocket-tts` renamed `DEFAULT_VARIANT` to `DEFAULT_LANGUAGE` and reshaped the model-loading API, which breaks a fresh build of the upstream Dockerfile with `ImportError`. The Dockerfile now pins `pocket-tts` to the last commit this shim was written against (`119ca2e`) via the `POCKET_TTS_REF` build arg. Override with `--build-arg POCKET_TTS_REF=<sha>` if you want to test a newer commit.

## Quick Start with Docker Compose

Use the included `docker-compose.yml` file:

```yaml
services:
  pocket-tts-wyoming:
    image: ghcr.io/ikidd/pocket-tts-wyoming:latest
    container_name: pocket-tts-wyoming
    network_mode: host
    environment:
      - WYOMING_PORT=10201
      - WYOMING_HOST=0.0.0.0
      - DEFAULT_VOICE=alba
      - MODEL_VARIANT=b6369a24
      - ZEROCONF=pocket-tts
    restart: unless-stopped
    volumes:
      - pocket-tts-hf-cache:/root/.cache/huggingface
      - pocket-tts-cache:/root/.cache/pocket_tts

volumes:
  pocket-tts-hf-cache:
    driver: local
  pocket-tts-cache:
    driver: local
```

### Configuration

You can customize the following environment variables in the compose file before starting:

| Variable | Default | Description |
|----------|---------|-------------|
| `WYOMING_PORT` | `10201` | The port the Wyoming protocol server listens on. Change if you have a conflict with another service. |
| `WYOMING_HOST` | `0.0.0.0` | The network interface to bind to. `0.0.0.0` accepts connections from any interface. |
| `DEFAULT_VOICE` | `alba` | The default voice used when none is specified. See [Available Voices](#available-voices) for options. |
| `MODEL_VARIANT` | `b6369a24` | The Pocket-TTS model variant to use. This corresponds to a specific model checkpoint. |
| `ZEROCONF` | `pocket-tts` | Service name for mDNS/Zeroconf discovery. Home Assistant uses this to auto-discover the TTS server. Set to empty string to disable. |

Pull and start:

```bash
docker compose pull
docker compose up -d
```

The pre-built image is automatically updated via GitHub Actions when changes are pushed to the repository or when the upstream [Pocket-TTS](https://github.com/kyutai-labs/pocket-tts) repository is updated.

## Using the Pre-built Image

The image is available on GitHub Container Registry:

```bash
docker pull ghcr.io/ikidd/pocket-tts-wyoming:latest
```

## Running with Docker

```bash
docker run -d \
  --name pocket-tts-wyoming \
  --network host \
  -e DEFAULT_VOICE=alba \
  -e MODEL_VARIANT=b6369a24 \
  -e ZEROCONF=pocket-tts \
  -v pocket-tts-hf-cache:/root/.cache/huggingface \
  -v pocket-tts-cache:/root/.cache/pocket_tts \
  ghcr.io/ikidd/pocket-tts-wyoming:latest
```

The volume mounts are recommended to cache model files and avoid re-downloads on restart.

## Building the Docker Image Manually

If you prefer to build the image locally instead of using the pre-built image:

```bash
docker build -t pocket-tts-wyoming .
```

To build against a different Pocket-TTS commit, pass the build arg:

```bash
docker build --build-arg POCKET_TTS_REF=<sha> -t pocket-tts-wyoming .
```

Then update `docker-compose.yml` to use `build: .` instead of `image: ghcr.io/ikidd/pocket-tts-wyoming:latest`.


## Available Voices

alba, marius, javert, jean, fantine, cosette, eponine, azelma

## Home Assistant Integration

The server supports Zeroconf/mDNS for automatic discovery.

1. Start the Docker container
2. Go to Settings -> Devices & Services -> Add Integration
3. Search for "Wyoming Protocol"
4. The server should appear in the "Discovered" section, or enter `tcp://<server-ip>:10201` manually
5. Configure a Voice Assistant pipeline to use the TTS service and select a voice

## Debug Mode

Debug mode writes WAV files for each synthesis request and exposes timing tunables for diagnosing audio issues (such as the first word being cut off).

To run in debug mode, include the debug overlay file:

```bash
docker compose -f docker-compose.yml -f docker-compose.debug.yml up -d --build
```

This enables:
- **WAV file output**: Each synthesis writes a debug WAV file to the project directory
- **Timing tunables**: Environment variables to adjust the sacrificial prefix trimming

### Background

Audio-prompt based TTS models like Pocket-TTS can "swallow" the first word into a blend region when transitioning from the voice prompt. To prevent this, a sacrificial prefix (`"..."`) is prepended to all text and then trimmed from the resulting audio. Debug mode lets you tune this trimming.

### Timing Tunables

| Variable | Default | Description |
|----------|---------|-------------|
| `PREFIX_MIN_DURATION` | `0.15` | Minimum seconds before looking for the pause after the prefix |
| `PREFIX_MAX_DURATION` | `1.0` | Maximum seconds to search for the prefix end |
| `PREFIX_SILENCE_GAP` | `0.08` | Minimum silence duration (seconds) to identify the gap after the prefix |

**Tuning tips:**
- If you hear part of the "..." prefix, decrease `PREFIX_SILENCE_GAP` to catch shorter pauses
- If the first syllable is still being cut, increase `PREFIX_MIN_DURATION`
- Different voices speak at different speeds, so optimal values may vary

## Troubleshooting

- **Slow startup**: First run downloads ~500MB of model weights. Use volume mounts to persist the cache.
- **Connection issues**: Verify port 10201 is open and check logs with `docker compose logs pocket-tts-wyoming` or `docker logs pocket-tts-wyoming`
- **Voice not found**: Ensure the voice name matches one of the 8 predefined voices listed above.
- **Image pull issues**: If you encounter authentication issues pulling from GHCR, ensure you're logged in: `docker login ghcr.io`
- **Outdated image**: Pull the latest image with `docker compose pull` or `docker pull ghcr.io/ikidd/pocket-tts-wyoming:latest`
- **First word cut off**: Run in debug mode and check the WAV files. Adjust the timing tunables as needed.
- **Long text produces no audio / socket errors in logs**: Fixed in this fork — see [Changes in this fork](#changes-in-this-fork). If you are still on the upstream image, rebuild from this fork or pull the local image built from this repo.
- **`ImportError: cannot import name 'DEFAULT_VARIANT'` on build**: Upstream Pocket-TTS moved on. This fork pins a known-good commit; rebuild from this repo, or pass `--build-arg POCKET_TTS_REF=<sha>` to test a different pin.

- **⏳ Last Build On**: Never
- **🔄 Last Run**: 2026-01-20 00:30:58 UTC
- **Last Upstream SHA**: 6f9dd250c24ee85cecc5587902a684f0d82b2a0d 
## 📅 Release Status
- **⏳ Last Build On**: 2026-04-29 00:54:56 UTC
- **🔄 Last Run**: 2026-04-29 00:54:56 UTC
- **Last Upstream SHA**: d529606ced20a8de5afc8740a170169858a8afb3
