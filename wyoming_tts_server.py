#!/usr/bin/env python3
"""
Wyoming Protocol TTS Server for Pocket-TTS

Implements Wyoming protocol TTS server that wraps pocket-tts,
exposing available voices to Home Assistant for selection.
"""

import argparse
import asyncio
import logging
import os
import wave
from datetime import datetime
from functools import partial
from typing import Optional

import numpy

from pocket_tts import TTSModel
from pocket_tts.default_parameters import DEFAULT_VARIANT
from pocket_tts.utils.utils import PREDEFINED_VOICES
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncServer, AsyncTcpServer
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
)

_LOGGER = logging.getLogger(__name__)

DEFAULT_PORT = int(os.environ.get("WYOMING_PORT", "10201"))
DEFAULT_VOICE = os.environ.get("DEFAULT_VOICE", "alba")
MODEL_VARIANT = os.environ.get("MODEL_VARIANT", DEFAULT_VARIANT)
DEBUG_WAV = os.environ.get("DEBUG_WAV", "").lower() in ("true", "1", "yes")

# Prefix trimming tunables (in seconds)
# Minimum time before looking for the pause after the sacrificial prefix
PREFIX_MIN_DURATION = float(os.environ.get("PREFIX_MIN_DURATION", "0.15"))
# Maximum time to search for the prefix end
PREFIX_MAX_DURATION = float(os.environ.get("PREFIX_MAX_DURATION", "1.0"))
# Minimum silence duration to consider it the gap after the prefix
PREFIX_SILENCE_GAP = float(os.environ.get("PREFIX_SILENCE_GAP", "0.08"))

_VOICE_STATES: dict[str, dict] = {}
_VOICE_LOCK = asyncio.Lock()


class PocketTTSEventHandler(AsyncEventHandler):
    """Event handler for Pocket-TTS Wyoming server."""

    def __init__(
        self,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        tts_model: TTSModel,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.tts_model = tts_model
        self.is_streaming: Optional[bool] = None
        self._synthesize: Optional[Synthesize] = None
        self._stream_text: str = ""

    async def handle_event(self, event: Event) -> bool:
        """Handle incoming Wyoming protocol events."""
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        try:
            if Synthesize.is_type(event.type):
                if self.is_streaming:
                    # Ignore since this is only sent for compatibility reasons.
                    # For streaming, we expect:
                    # [synthesize-start] -> [synthesize-chunk]+ -> [synthesize]? -> [synthesize-stop]
                    return True

                synthesize = Synthesize.from_event(event)
                await self._handle_synthesize(synthesize, send_start=True, send_stop=True)
                return True

            if SynthesizeStart.is_type(event.type):
                stream_start = SynthesizeStart.from_event(event)
                self.is_streaming = True
                self._stream_text = ""
                self._synthesize = Synthesize(text="", voice=stream_start.voice)
                _LOGGER.debug("Text stream started: voice=%s", stream_start.voice)
                return True

            if SynthesizeChunk.is_type(event.type):
                assert self._synthesize is not None
                stream_chunk = SynthesizeChunk.from_event(event)
                self._stream_text += stream_chunk.text
                _LOGGER.debug("Received stream chunk: %s", stream_chunk.text[:50])
                return True

            if SynthesizeStop.is_type(event.type):
                assert self._synthesize is not None
                if self._stream_text.strip():
                    self._synthesize.text = self._stream_text.strip()
                    await self._handle_synthesize(
                        self._synthesize, send_start=True, send_stop=True
                    )

                await self.write_event(SynthesizeStopped().event())
                self.is_streaming = False
                self._stream_text = ""
                _LOGGER.debug("Text stream stopped")
                return True

            return True
        except Exception as err:
            await self.write_event(
                Error(text=str(err), code=err.__class__.__name__).event()
            )
            raise err

    async def _handle_synthesize(
        self, synthesize: Synthesize, send_start: bool = True, send_stop: bool = True
    ) -> bool:
        """Handle synthesis request."""
        _LOGGER.debug(synthesize)

        raw_text = synthesize.text
        text = " ".join(raw_text.strip().splitlines())

        if not text:
            _LOGGER.warning("Empty text received")
            if send_stop:
                await self.write_event(AudioStop().event())
            return True

        _LOGGER.debug("synthesize: raw_text=%s, text='%s'", raw_text, text)
        
        # Add a sacrificial prefix to prevent the first word from being swallowed
        # by the voice prompt "blend region". This prefix audio will be trimmed later.
        text = "... " + text
        
        voice_name: Optional[str] = None

        if synthesize.voice is not None:
            voice_name = synthesize.voice.name

        if voice_name is None:
            voice_name = self.cli_args.voice

        # Extract voice name from model name if it's in format "pocket-tts-{voice}"
        if voice_name and voice_name.startswith("pocket-tts-"):
            voice_name = voice_name.replace("pocket-tts-", "", 1)

        if voice_name not in PREDEFINED_VOICES:
            _LOGGER.warning(
                "Voice '%s' not found, using default '%s'", voice_name, self.cli_args.voice
            )
            voice_name = self.cli_args.voice

        assert voice_name is not None

        async with _VOICE_LOCK:
            global _VOICE_STATES
            if voice_name not in _VOICE_STATES:
                _LOGGER.info("Loading voice state for: %s", voice_name)
                try:
                    _VOICE_STATES[voice_name] = self.tts_model.get_state_for_audio_prompt(
                        voice_name
                    )
                except Exception as e:
                    _LOGGER.error("Failed to load voice state for %s: %s", voice_name, e)
                    await self.write_event(
                        Error(
                            text=f"Failed to load voice: {voice_name}",
                            code="VoiceLoadError",
                        ).event()
                    )
                    return True

            voice_state = _VOICE_STATES[voice_name]

            try:
                _LOGGER.info(
                    "Synthesizing text (voice: %s, length: %d chars)", voice_name, len(text)
                )

                sample_rate = self.tts_model.sample_rate
                width = 2
                channels = 1
                bytes_per_sample = width * channels
                samples_per_chunk = 1024
                bytes_per_chunk = bytes_per_sample * samples_per_chunk

                if send_start:
                    await self.write_event(
                        AudioStart(
                            rate=sample_rate,
                            width=width,
                            channels=channels,
                        ).event(),
                    )

                audio_chunks = self.tts_model.generate_audio_stream(
                    model_state=voice_state, text_to_generate=text, copy_state=True
                )

                all_audio_arrays = []
                for audio_chunk in audio_chunks:
                    audio_array = audio_chunk.detach().cpu().numpy()
                    all_audio_arrays.append(audio_array)

                if not all_audio_arrays:
                    if send_stop:
                        await self.write_event(AudioStop().event())
                    return True

                full_audio = numpy.concatenate(all_audio_arrays)

                # Find and remove the sacrificial prefix ("...") by detecting the pause after it
                # This adapts to different voice speeds rather than using a fixed duration
                silence_threshold = 0.01
                max_amplitude = numpy.abs(full_audio).max()
                threshold = max_amplitude * silence_threshold
                
                # Minimum time before we start looking for the pause (avoid false early detection)
                min_prefix_samples = int(sample_rate * PREFIX_MIN_DURATION)
                # Maximum time to search for the prefix end
                max_prefix_samples = int(sample_rate * PREFIX_MAX_DURATION)
                # Minimum silence duration to consider it the gap after "..."
                min_silence_samples = int(sample_rate * PREFIX_SILENCE_GAP)
                
                # Find where the prefix ends by looking for a silence gap
                prefix_end = 0
                if len(full_audio) > min_prefix_samples:
                    search_end = min(len(full_audio), max_prefix_samples)
                    is_silent = numpy.abs(full_audio[:search_end]) < threshold
                    
                    # Look for a silence gap after the minimum prefix duration
                    i = min_prefix_samples
                    while i < search_end:
                        if is_silent[i]:
                            # Found start of silence, check if it's long enough
                            silence_start = i
                            while i < search_end and is_silent[i]:
                                i += 1
                            silence_length = i - silence_start
                            if silence_length >= min_silence_samples:
                                # Found the gap after the prefix - start after this silence
                                prefix_end = i
                                break
                        else:
                            i += 1
                
                if prefix_end > 0:
                    _LOGGER.debug("Trimming prefix: %d samples (%.3fs)", 
                                  prefix_end, prefix_end / sample_rate)
                    full_audio = full_audio[prefix_end:]

                # Trim any remaining leading silence
                non_silent_indices = numpy.where(numpy.abs(full_audio) > threshold)[0]
                if len(non_silent_indices) > 0:
                    padding_samples = int(sample_rate * 0.05)  # 50ms padding
                    first_non_silent = max(0, non_silent_indices[0] - padding_samples)
                    full_audio = full_audio[first_non_silent:]
                    
                    # Trim trailing silence at the end
                    non_silent_indices = numpy.where(numpy.abs(full_audio) > threshold)[0]
                    if len(non_silent_indices) > 0:
                        last_non_silent = non_silent_indices[-1] + padding_samples
                        full_audio = full_audio[:last_non_silent + 1]

                full_audio = (full_audio.clip(-1.0, 1.0) * 32767).astype("int16")
                audio_bytes = full_audio.tobytes()

                # Write debug WAV file if enabled
                if self.cli_args.debug_wav:
                    try:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        wav_filename = f"/output/debug_{voice_name}_{timestamp}.wav"
                        with wave.open(wav_filename, "wb") as wav_file:
                            wav_file.setnchannels(channels)
                            wav_file.setsampwidth(width)
                            wav_file.setframerate(sample_rate)
                            wav_file.writeframes(audio_bytes)
                        _LOGGER.info("Debug WAV file written: %s", wav_filename)
                    except Exception as e:
                        _LOGGER.warning("Failed to write debug WAV file: %s", e)

                num_chunks = int(numpy.ceil(len(audio_bytes) / bytes_per_chunk))
                for i in range(num_chunks):
                    offset = i * bytes_per_chunk
                    chunk = audio_bytes[offset : offset + bytes_per_chunk]
                    await self.write_event(
                        AudioChunk(
                            audio=chunk,
                            rate=sample_rate,
                            width=width,
                            channels=channels,
                        ).event(),
                    )

                if send_stop:
                    await self.write_event(AudioStop().event())

                _LOGGER.info("Synthesis complete")
            except Exception as e:
                _LOGGER.error("Error during synthesis: %s", e, exc_info=True)
                await self.write_event(
                    Error(text=str(e), code=e.__class__.__name__).event()
                )
                return True

        return True


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Wyoming Protocol TTS Server for Pocket-TTS"
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("WYOMING_HOST", "0.0.0.0"),
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to listen on (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        help=f"Default voice to use (default: {DEFAULT_VOICE})",
    )
    parser.add_argument(
        "--variant",
        default=MODEL_VARIANT,
        help=f"Model variant (default: {MODEL_VARIANT})",
    )
    parser.add_argument(
        "--uri",
        default=None,
        help="Server URI (e.g., tcp://0.0.0.0:10201). If not provided, constructed from --host and --port",
    )
    parser.add_argument(
        "--zeroconf",
        nargs="?",
        const="pocket-tts",
        help="Enable discovery over zeroconf with optional name (default: pocket-tts)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging output",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Log DEBUG messages",
    )
    parser.add_argument(
        "--debug-wav",
        action="store_true",
        help="Write complete WAV file to /output/ on every response (default: from DEBUG_WAV env var)",
    )
    parser.add_argument(
        "--log-format",
        default=logging.BASIC_FORMAT,
        help="Format for log messages",
    )

    args = parser.parse_args()
    
    # Override debug_wav from environment if not explicitly set via command line
    # Check environment variable at runtime (not just at module load)
    debug_wav_env = os.environ.get("DEBUG_WAV", "").lower() in ("true", "1", "yes")
    if not args.debug_wav:
        args.debug_wav = debug_wav_env

    log_level = logging.DEBUG if args.debug else (logging.ERROR if args.quiet else logging.INFO)
    logging.basicConfig(level=log_level, format=args.log_format)
    if args.debug_wav:
        _LOGGER.info("Debug WAV mode enabled - WAV files will be written to /output/ on every response")
    _LOGGER.debug(args)

    os.environ["MODEL_VARIANT"] = args.variant
    variant = os.environ.get("MODEL_VARIANT", MODEL_VARIANT)
    _LOGGER.info("Loading Pocket-TTS model (variant: %s)...", variant)
    tts_model = TTSModel.load_model(config=variant)
    _LOGGER.info("Model loaded successfully")
    _LOGGER.info("Sample rate: %d Hz", tts_model.sample_rate)

    _LOGGER.info("Pre-loading voice states for %d voices...", len(PREDEFINED_VOICES))
    for voice_name in PREDEFINED_VOICES:
        try:
            voice_state = tts_model.get_state_for_audio_prompt(voice_name)
            global _VOICE_STATES
            _VOICE_STATES[voice_name] = voice_state
            _LOGGER.info("Loaded voice state for: %s", voice_name)
        except Exception as e:
            _LOGGER.warning("Failed to load voice state for %s: %s", voice_name, e)
    _LOGGER.info("Voice states pre-loaded")

    voices = [
        TtsVoice(
            name=voice_name,
            description=f"Pocket-TTS voice: {voice_name}",
            attribution=Attribution(
                name="Kyutai Pocket-TTS",
                url="https://github.com/kyutai-labs/pocket-tts",
            ),
            installed=True,
            version=None,
            languages=["en"],
            speakers=None,
        )
        for voice_name in PREDEFINED_VOICES
    ]

    wyoming_info = Info(
        tts=[
            TtsProgram(
                name="pocket-tts",
                description="A fast, local, neural text to speech engine",
                attribution=Attribution(
                    name="Kyutai Pocket-TTS",
                    url="https://github.com/kyutai-labs/pocket-tts",
                ),
                installed=True,
                voices=sorted(voices, key=lambda v: v.name),
                version="1.0.1",
                supports_synthesize_streaming=True,
            )
        ],
    )

    if args.uri is None:
        args.uri = f"tcp://{args.host}:{args.port}"

    server = AsyncServer.from_uri(args.uri)

    zeroconf_name = args.zeroconf
    if not zeroconf_name:
        zeroconf_env = os.environ.get("ZEROCONF")
        if zeroconf_env:
            zeroconf_name = zeroconf_env if zeroconf_env != "true" else "pocket-tts"

    if zeroconf_name:
        if not isinstance(server, AsyncTcpServer):
            raise ValueError("Zeroconf requires tcp:// uri")

        from wyoming.zeroconf import HomeAssistantZeroconf
        import socket

        tcp_server: AsyncTcpServer = server
        zeroconf_host = tcp_server.host
        if zeroconf_host == "0.0.0.0" or not zeroconf_host:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                zeroconf_host = s.getsockname()[0]
                s.close()
            except Exception:
                zeroconf_host = "127.0.0.1"
        
        hass_zeroconf = HomeAssistantZeroconf(
            name=zeroconf_name, port=tcp_server.port, host=zeroconf_host
        )
        await hass_zeroconf.register_server()
        _LOGGER.debug("Zeroconf discovery enabled: name=%s, port=%d, host=%s", 
                     zeroconf_name, tcp_server.port, zeroconf_host)

    _LOGGER.info("Ready")
    _LOGGER.info("Available voices: %s", ", ".join(PREDEFINED_VOICES.keys()))
    await server.run(
        partial(
            PocketTTSEventHandler,
            wyoming_info,
            args,
            tts_model,
        )
    )


def run():
    """Run the server."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        _LOGGER.info("Server stopped")


if __name__ == "__main__":
    run()
