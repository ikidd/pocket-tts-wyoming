#!/usr/bin/env python3
"""
Wyoming Protocol TTS Server for Pocket-TTS (Official Repo Version)
Optimized for Streaming and Home Assistant Debug Timings.
"""

import argparse
import asyncio
import logging
import os
from functools import partial
from typing import Optional

import numpy

# Importe aus dem offiziellen pocket_tts Repository
from pocket_tts import TTSModel
from pocket_tts.default_parameters import DEFAULT_VARIANT
from pocket_tts.utils.utils import PREDEFINED_VOICES
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncServer
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

PREFIX_MIN_DURATION = float(os.environ.get("PREFIX_MIN_DURATION", "0.1"))
PREFIX_MAX_DURATION = float(os.environ.get("PREFIX_MAX_DURATION", "1.0"))
PREFIX_SILENCE_GAP = float(os.environ.get("PREFIX_SILENCE_GAP", "0.08"))

_VOICE_STATES: dict[str, dict] = {}
_VOICE_LOCK = asyncio.Lock()


class PocketTTSEventHandler(AsyncEventHandler):
    def __init__(self, wyoming_info: Info, cli_args: argparse.Namespace, tts_model: TTSModel, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.tts_model = tts_model
        self.is_streaming: Optional[bool] = None
        self._synthesize: Optional[Synthesize] = None
        self._stream_text: str = ""

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            return True

        try:
            # Normales TTS (z.B. über das Medien-Tab)
            if Synthesize.is_type(event.type):
                synthesize = Synthesize.from_event(event)
                # Bestätige den Start, damit HA die Zeitmessung beginnt
                await self.write_event(SynthesizeStart(text=synthesize.text).event())
                await self._handle_synthesize(synthesize)
                # Sende das Ende erst ganz zum Schluss
                await self.write_event(SynthesizeStopped().event())
                return True

            # Streaming TTS (z.B. über Voice Assistant / LLM)
            if SynthesizeStart.is_type(event.type):
                stream_start = SynthesizeStart.from_event(event)
                # Wir schicken das Event zurück als Acknowledge
                await self.write_event(stream_start.event())
                self.is_streaming = True
                self._stream_text = ""
                self._synthesize = Synthesize(text="", voice=stream_start.voice)
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
                    await self._handle_synthesize(self._synthesize)
                
                await self.write_event(SynthesizeStopped().event())
                self.is_streaming = False
                self._stream_text = ""
                return True

            return True
        except Exception as err:
            await self.write_event(Error(text=str(err), code=err.__class__.__name__).event())
            raise err

    async def _handle_synthesize(self, synthesize: Synthesize) -> bool:
        """Kern-Logik für die Generierung (ohne eigene Stop-Events)"""
        raw_text = synthesize.text
        text = " ".join(raw_text.strip().splitlines())
        if not text:
            return True

        _LOGGER.info("Synthesizing: '%s'", text)
        text = "... " + text 
        
        voice_name = synthesize.voice.name if synthesize.voice else self.cli_args.voice
        if voice_name and voice_name.startswith("pocket-tts-"):
            voice_name = voice_name.replace("pocket-tts-", "", 1)
        if voice_name not in PREDEFINED_VOICES:
            voice_name = self.cli_args.voice

        async with _VOICE_LOCK:
            global _VOICE_STATES
            if voice_name not in _VOICE_STATES:
                if hasattr(self.tts_model, 'get_state_for_audio_prompt'):
                    _VOICE_STATES[voice_name] = self.tts_model.get_state_for_audio_prompt(voice_name)
                else:
                    _VOICE_STATES[voice_name] = self.tts_model.get_state_for_voice(voice_name)
            
            voice_state = _VOICE_STATES[voice_name]

            try:
                sample_rate = self.tts_model.sample_rate
                # Starte den Audiostrom
                await self.write_event(AudioStart(rate=sample_rate, width=2, channels=1).event())

                audio_stream = self.tts_model.generate_audio_stream(
                    model_state=voice_state, text_to_generate=text, copy_state=True
                )

                prefix_buffer = numpy.array([], dtype=numpy.float32)
                prefix_processed = False
                max_prefix_samples = int(sample_rate * PREFIX_MAX_DURATION)
                
                for chunk_tensor in audio_stream:
                    chunk_data = chunk_tensor.detach().cpu().numpy().flatten()
                    
                    if not prefix_processed:
                        prefix_buffer = numpy.concatenate([prefix_buffer, chunk_data])
                        if len(prefix_buffer) >= max_prefix_samples:
                            valid_audio = self._trim_prefix(prefix_buffer, sample_rate)
                            await self._send_audio_data(valid_audio)
                            prefix_processed = True
                            prefix_buffer = None
                    else:
                        await self._send_audio_data(chunk_data)
                        await asyncio.sleep(0)

                if not prefix_processed and len(prefix_buffer) > 0:
                     valid_audio = self._trim_prefix(prefix_buffer, sample_rate)
                     await self._send_audio_data(valid_audio)

                # Beende den Audiostrom (aber NICHT den gesamten Synthesize-Task)
                await self.write_event(AudioStop().event())
                _LOGGER.info("Audio transmission complete")
            except Exception as e:
                _LOGGER.error("Error synthesis: %s", e, exc_info=True)
                await self.write_event(Error(text=str(e), code="SynthesisError").event())
        return True

    def _trim_prefix(self, audio_data, sample_rate):
        threshold = (numpy.abs(audio_data).max() if len(audio_data) > 0 else 0) * 0.01
        min_prefix_samples = int(sample_rate * PREFIX_MIN_DURATION)
        max_prefix_samples = int(sample_rate * PREFIX_MAX_DURATION)
        prefix_end = 0
        if len(audio_data) > min_prefix_samples:
            search_end = min(len(audio_data), max_prefix_samples)
            is_silent = numpy.abs(audio_data[:search_end]) < threshold
            i = min_prefix_samples
            while i < search_end:
                if is_silent[i]:
                    silence_start = i
                    while i < search_end and is_silent[i]: i += 1
                    if (i - silence_start) >= int(sample_rate * PREFIX_SILENCE_GAP):
                        prefix_end = i
                        break
                else: i += 1
        return audio_data[prefix_end:] if prefix_end > 0 else audio_data[min_prefix_samples:]

    async def _send_audio_data(self, float_audio):
        if len(float_audio) == 0: return
        _LOGGER.info(f"DEBUG STREAM: Sende {len(float_audio)} Samples...")
        audio_int16 = (float_audio.clip(-1.0, 1.0) * 32767).astype("int16")
        audio_bytes = audio_int16.tobytes()
        chunk_size = 2048
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i : i + chunk_size]
            await self.write_event(AudioChunk(rate=24000, width=2, channels=1, audio=chunk).event())


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument("--variant", default=MODEL_VARIANT)
    parser.add_argument("--uri", default=None)
    parser.add_argument("--zeroconf", nargs="?", const="pocket-tts")
    args, _ = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO)
    
    _LOGGER.info("Loading Pocket-TTS model (%s)...", args.variant)
    try:
        tts_model = TTSModel.load_model(args.variant)
    except TypeError:
        try:
            tts_model = TTSModel.load_model(model_variant_id=args.variant)
        except TypeError:
            tts_model = TTSModel.load_model(variant_id=args.variant)
    
    _LOGGER.info("Pre-loading voice states...")
    for v in PREDEFINED_VOICES:
        try:
             if hasattr(tts_model, 'get_state_for_audio_prompt'):
                 _VOICE_STATES[v] = tts_model.get_state_for_audio_prompt(v)
             else:
                 _VOICE_STATES[v] = tts_model.get_state_for_voice(v)
        except: pass

    voices = [
        TtsVoice(
            name=v, description=f"Pocket-TTS {v}",
            attribution=Attribution(name="Kyutai", url="https://kyutai.org"),
            installed=True, languages=["en"], version="1.0.1", speakers=None
        ) for v in PREDEFINED_VOICES
    ]

    wyoming_info = Info(
        tts=[TtsProgram(
            name="pocket-tts", description="Streaming Pocket-TTS",
            attribution=Attribution(name="Kyutai", url="https://kyutai.org"),
            installed=True, voices=voices, version="1.0.1",
            supports_synthesize_streaming=True
        )]
    )

    if args.uri is None: args.uri = f"tcp://{args.host}:{args.port}"
    _LOGGER.info("Server Ready on %s", args.uri)
    server = AsyncServer.from_uri(args.uri)
    await server.run(partial(PocketTTSEventHandler, wyoming_info, args, tts_model))

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: pass
