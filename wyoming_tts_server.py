#!/usr/bin/env python3
"""
Wyoming Protocol TTS Server for Pocket-TTS

Implements Wyoming protocol TTS server that wraps pocket-tts,
exposing available voices to Home Assistant for selection.
Modified for Low-Latency Streaming.
"""

import argparse
import asyncio
import logging
import os
import wave
from datetime import datetime
from functools import partial
from typing import Optional, List

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
# Reduced default MIN duration for faster response (0.15 -> 0.1)
PREFIX_MIN_DURATION = float(os.environ.get("PREFIX_MIN_DURATION", "0.1"))
PREFIX_MAX_DURATION = float(os.environ.get("PREFIX_MAX_DURATION", "1.0"))
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
        """Handle synthesis request with streaming."""
        raw_text = synthesize.text
        text = " ".join(raw_text.strip().splitlines())

        if not text:
            _LOGGER.warning("Empty text received")
            if send_stop:
                await self.write_event(AudioStop().event())
            return True

        _LOGGER.info("Synthesizing: '%s'", text)
        
        # Add a sacrificial prefix
        text = "... " + text
        
        voice_name: Optional[str] = None
        if synthesize.voice is not None:
            voice_name = synthesize.voice.name

        if voice_name is None:
            voice_name = self.cli_args.voice

        if voice_name and voice_name.startswith("pocket-tts-"):
            voice_name = voice_name.replace("pocket-tts-", "", 1)

        if voice_name not in PREDEFINED_VOICES:
            _LOGGER.warning("Voice '%s' not found, using default", voice_name)
            voice_name = self.cli_args.voice

        assert voice_name is not None

        async with _VOICE_LOCK:
            global _VOICE_STATES
            if voice_name not in _VOICE_STATES:
                _LOGGER.info("Loading voice state for: %s", voice_name)
                try:
                    _VOICE_STATES[voice_name] = self.tts_model.get_state_for_audio_prompt(voice_name)
                except Exception as e:
                    _LOGGER.error("Failed to load voice state: %s", e)
                    await self.write_event(Error(text=f"Failed to load voice: {voice_name}", code="VoiceLoadError").event())
                    return True

            voice_state = _VOICE_STATES[voice_name]

            try:
                sample_rate = self.tts_model.sample_rate
                
                if send_start:
                    await self.write_event(
                        AudioStart(rate=sample_rate, width=2, channels=1).event(),
                    )

                # Start the stream generator
                audio_stream = self.tts_model.generate_audio_stream(
                    model_state=voice_state, text_to_generate=text, copy_state=True
                )

                # Streaming Logic Variables
                prefix_buffer = numpy.array([], dtype=numpy.float32)
                prefix_processed = False
                max_prefix_samples = int(sample_rate * PREFIX_MAX_DURATION)
                
                # Container for full audio if debug wav is enabled
                debug_full_audio = [] if self.cli_args.debug_wav else None

                for chunk_tensor in audio_stream:
                    # Convert Tensor to Numpy
                    chunk_data = chunk_tensor.detach().cpu().numpy().flatten()
                    
                    if not prefix_processed:
                        # Buffer initial chunks to find and remove prefix
                        prefix_buffer = numpy.concatenate([prefix_buffer, chunk_data])
                        
                        if len(prefix_buffer) >= max_prefix_samples:
                            # Trim prefix
                            valid_audio = self._trim_prefix(prefix_buffer, sample_rate)
                            
                            # Send valid audio
                            await self._send_audio_data(valid_audio)
                            
                            # Store for debug
                            if debug_full_audio is not None:
                                debug_full_audio.append(valid_audio)
                            
                            prefix_processed = True
                            prefix_buffer = None # Clear memory
                    else:
                        # Prefix already handled, send immediately
                        await self._send_audio_data(chunk_data)
                        
                        # Store for debug
                        if debug_full_audio is not None:
                            debug_full_audio.append(chunk_data)
                            
                        # Yield control to event loop to allow sending
                        await asyncio.sleep(0)

                # Process remaining buffer if stream ended early
                if not prefix_processed and len(prefix_buffer) > 0:
                     valid_audio = self._trim_prefix(prefix_buffer, sample_rate)
                     await self._send_audio_data(valid_audio)
                     if debug_full_audio is not None:
                        debug_full_audio.append(valid_audio)

                if send_stop:
                    await self.write_event(AudioStop().event())

                _LOGGER.info("Synthesis complete")

                # Write Debug WAV if enabled
                if debug_full_audio and len(debug_full_audio) > 0:
                    self._save_debug_wav(numpy.concatenate(debug_full_audio), voice_name, sample_rate)

            except Exception as e:
                _LOGGER.error("Error during synthesis: %s", e, exc_info=True)
                await self.write_event(
                    Error(text=str(e), code=e.__class__.__name__).event()
                )
                return True

        return True

    def _trim_prefix(self, audio_data: numpy.ndarray, sample_rate: int) -> numpy.ndarray:
        """Helper to find silence after prefix and trim the array."""
        silence_threshold = 0.01
        max_amplitude = numpy.abs(audio_data).max() if len(audio_data) > 0 else 0
        threshold = max_amplitude * silence_threshold
        
        min_prefix_samples = int(sample_rate * PREFIX_MIN_DURATION)
        max_prefix_samples = int(sample_rate * PREFIX_MAX_DURATION)
        min_silence_samples = int(sample_rate * PREFIX_SILENCE_GAP)
        
        prefix_end = 0
        
        if len(audio_data) > min_prefix_samples:
            search_end = min(len(audio_data), max_prefix_samples)
            is_silent = numpy.abs(audio_data[:search_end]) < threshold
            
            i = min_prefix_samples
            while i < search_end:
                if is_silent[i]:
                    silence_start = i
                    while i < search_end and is_silent[i]:
                        i += 1
                    if (i - silence_start) >= min_silence_samples:
                        prefix_end = i
                        break
                else:
                    i += 1
        
        if prefix_end > 0:
            return audio_data[prefix_end:]
        
        # Fallback: Hard trim if no silence found
        return audio_data[min_prefix_samples:]

    async def _send_audio_data(self, float_audio: numpy.ndarray) -> None:
        """Convert float32 audio to int16 and send chunks."""
        if len(float_audio) == 0:
            return

        # Float32 -> Int16
        audio_int16 = (float_audio.clip(-1.0, 1.0) * 32767).astype("int16")
        audio_bytes = audio_int16.tobytes()
        
        # Chunking for HA (2048 bytes = 1024 samples)
        chunk_size = 2048 
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i : i + chunk_size]
            await self.write_event(
                AudioChunk(rate=24000, width=2, channels=1, audio=chunk).event()
            )

    def _save_debug_wav(self, audio_data: numpy.ndarray, voice_name: str, sample_rate: int):
        """Saves the full audio to a WAV file."""
        try:
            audio_int16 = (audio_data.clip(-1.0, 1.0) * 32767).astype("int16")
            audio_bytes = audio_int16.tobytes()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            wav_filename = f"/output/debug_{voice_name}_{timestamp}.wav"
            
            with wave.open(wav_filename, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_bytes)
            _LOGGER.info("Debug WAV file written: %s", wav_filename)
        except Exception as e:
            _LOGGER.warning("Failed to write debug WAV file: %s", e)


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
    
    debug_wav_env = os.environ.get("DEBUG_WAV", "").lower() in ("true", "1", "yes")
    if not args.debug_wav:
        args.debug_wav = debug_wav_env

    log_level = logging.DEBUG if args.debug else (logging.ERROR if args.quiet else logging.INFO)
    logging.basicConfig(level=log_level, format=args.log_format)
    
    if args.debug_wav:
        _LOGGER.info("Debug WAV mode enabled")
        
    _LOGGER.debug(args)

    os.environ["MODEL_VARIANT"] = args.variant
    variant = os.environ.get("MODEL_VARIANT", MODEL_VARIANT)
    _LOGGER.info("Loading Pocket-TTS model (variant: %s)...", variant)
    tts_model = TTSModel.load_model(variant=variant)
    _LOGGER.info("Model loaded successfully")

    # Load voices
    _LOGGER.info("Pre-loading voice states...")
    for voice_name in PREDEFINED_VOICES:
        try:
            voice_state = tts_model.get_state_for_audio_prompt(voice_name)
            global _VOICE_STATES
            _VOICE_STATES[voice_name] = voice_state
        except Exception as e:
            _LOGGER.warning("Failed to load voice %s: %s", voice_name, e)

    # Voices Setup (Fixed TtsVoice initialization)
    voices = [
        TtsVoice(
            name=voice_name,
            description=f"Pocket-TTS voice: {voice_name}",
            attribution=Attribution(
                name="Kyutai Pocket-TTS",
                url="https://github.com/kyutai-labs/pocket-tts",
            ),
            installed=True,
            version="1.0.0",  # Added version
            languages=["en"], # Added language
            speakers=None,
        )
        for voice_name in PREDEFINED_VOICES
    ]

    wyoming_info = Info(
        tts=[
            TtsProgram(
                name="pocket-tts",
                description="Fast Streaming Pocket-TTS",
                attribution=Attribution(
                    name="Kyutai Pocket-TTS",
                    url="https://github.com/kyutai-labs/pocket-tts",
                ),
                installed=True,
                voices=sorted(voices, key=lambda v: v.name),
                version="1.0.1",
                supports_synthesize_streaming=True, # Enabled Streaming Flag
            )
        ],
    )

    if args.uri is None:
        args.uri = f"tcp://{args.host}:{args.port}"

    server = AsyncServer.from_uri(args.uri)
    
    # Zeroconf logic setup
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
        _LOGGER.debug("Zeroconf discovery enabled: %s", zeroconf_name)

    _LOGGER.info("Ready on %s", args.uri)
    await server.run(
        partial(
            PocketTTSEventHandler,
            wyoming_info,
            args,
            tts_model,
        )
    )

def run():
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        _LOGGER.info("Server stopped")

if __name__ == "__main__":
    run()
