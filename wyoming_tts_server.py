#!/usr/bin/env python3
import argparse
import asyncio
import logging
import os
import re
import time
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

# Tuning für Trimming
PREFIX_MIN_DURATION = 0.08
PREFIX_MAX_DURATION = 0.8
PREFIX_SILENCE_GAP = 0.06

_VOICE_STATES: dict[str, dict] = {}
_VOICE_LOCK = asyncio.Lock()

class PocketTTSEventHandler(AsyncEventHandler):
    def __init__(self, wyoming_info: Info, cli_args: argparse.Namespace, tts_model: TTSModel, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.tts_model = tts_model
        
        self.is_streaming: bool = False
        self._text_buffer: str = ""
        self._voice_info: Optional[any] = None
        self._audio_started: bool = False
        self._last_stream_end_time: float = 0
        self._current_task: Optional[asyncio.Task] = None
        self._chunk_count: int = 0 

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            return True

        try:
            if Synthesize.is_type(event.type):
                if self.is_streaming or (time.time() - self._last_stream_end_time < 1.5):
                    return True
                self._cancel_current()
                synthesize = Synthesize.from_event(event)
                self._current_task = asyncio.create_task(self._run_single(synthesize))
                return True

            if SynthesizeStart.is_type(event.type):
                self._cancel_current()
                start_event = SynthesizeStart.from_event(event)
                await self.write_event(start_event.event())
                self.is_streaming = True
                self._text_buffer = ""
                self._voice_info = start_event.voice
                self._audio_started = False
                self._chunk_count = 0
                return True

            if SynthesizeChunk.is_type(event.type):
                if not self.is_streaming: return True
                chunk = SynthesizeChunk.from_event(event)
                self._text_buffer += chunk.text
                
                # PROGRESSIVER THRESHOLD:
                # 0. Chunk (Erster Satz): 1 Satz
                # 1. Chunk (Satz 2-4): 3 Sätze
                # 2. Chunk (Ab Satz 5): 50 Sätze (praktisch "der Rest")
                if self._chunk_count == 0:
                    threshold = 1
                elif self._chunk_count == 1:
                    threshold = 3
                else:
                    threshold = 50 

                if self._count_sentences(self._text_buffer) >= threshold:
                    # Wir nutzen take_all_complete=True ab dem 2. Chunk (index 1)
                    await self._process_buffer(
                        threshold=threshold, 
                        take_all_complete=(self._chunk_count >= 1)
                    )
                return True

            if SynthesizeStop.is_type(event.type):
                if self.is_streaming:
                    # Beim Stop vom LLM nehmen wir den gesamten Rest (auch ohne Punkt am Ende)
                    await self._process_buffer(force_all_buffer=True)
                    if self._audio_started:
                        await self.write_event(AudioStop().event())
                    await self.write_event(SynthesizeStopped().event())
                    self.is_streaming = False
                    self._last_stream_end_time = time.time()
                return True

            return True
        except Exception as err:
            _LOGGER.error("Fehler: %s", err)
            return False

    def _count_sentences(self, text: str) -> int:
        # Zählt Sätze (Punkt, Ausrufezeichen, Fragezeichen gefolgt von Leerzeichen oder Ende)
        # Verhindert Splitting bei "2.0" oder "Dr. Peter"
        return len(re.findall(r'[.!?](\s|$)', text))

    def _cancel_current(self):
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()

    async def _process_buffer(self, threshold=1, take_all_complete=False, force_all_buffer=False):
        if not self._text_buffer.strip(): return

        to_speak = ""
        # Wir splitten nur bei echten Satzzeichen
        split_pattern = r'([.!?](?:\s+|$))'
        
        if force_all_buffer:
            # LLM ist fertig: Alles raus, was noch da ist
            to_speak = self._text_buffer.strip()
            self._text_buffer = ""
        else:
            parts = re.split(split_pattern, self._text_buffer)
            # parts: ["Satz 1", ". ", "Satz 2", "! ", "Unvollständiger Rest..."]
            
            if take_all_complete:
                # Nimm alle fertigen Sätze, lass den unvollständigen Rest im Puffer
                last_punctuation_idx = (len(parts) // 2) * 2
                if last_punctuation_idx > 0:
                    to_speak = "".join(parts[:last_punctuation_idx]).strip()
                    self._text_buffer = "".join(parts[last_punctuation_idx:])
            else:
                # Nimm exakt die Anzahl 'threshold' an Sätzen
                if len(parts) >= (threshold * 2):
                    end_idx = threshold * 2
                    to_speak = "".join(parts[:end_idx]).strip()
                    self._text_buffer = "".join(parts[end_idx:])

        if to_speak:
            self._chunk_count += 1
            _LOGGER.info("Generiere Chunk %d: '%s'", self._chunk_count, to_speak)
            await self._handle_synthesize(to_speak, self._voice_info, is_last=False)

    async def _run_single(self, synthesize):
        await self._handle_synthesize(synthesize.text, synthesize.voice, is_last=True)

    async def _handle_synthesize(self, text: str, voice_obj: Optional[any], is_last: bool) -> bool:
        if not text.strip(): return True
        processed_text = "... " + text
        voice_name = voice_obj.name if voice_obj else self.cli_args.voice
        if voice_name and voice_name.startswith("pocket-tts-"):
            voice_name = voice_name.replace("pocket-tts-", "", 1)
        if voice_name not in PREDEFINED_VOICES: voice_name = self.cli_args.voice

        async with _VOICE_LOCK:
            if voice_name not in _VOICE_STATES:
                method = getattr(self.tts_model, 'get_state_for_audio_prompt', 
                                getattr(self.tts_model, 'get_state_for_voice', None))
                _VOICE_STATES[voice_name] = method(voice_name)
            
            voice_state = _VOICE_STATES[voice_name]

            try:
                sample_rate = self.tts_model.sample_rate
                if not self._audio_started:
                    await self.write_event(AudioStart(rate=sample_rate, width=2, channels=1).event())
                    self._audio_started = True

                audio_stream = self.tts_model.generate_audio_stream(
                    model_state=voice_state, text_to_generate=processed_text, copy_state=True
                )

                prefix_buffer = numpy.array([], dtype=numpy.float32)
                prefix_processed = False
                max_prefix_samples = int(sample_rate * PREFIX_MAX_DURATION)
                
                for chunk_tensor in audio_stream:
                    if asyncio.current_task().cancelled(): raise asyncio.CancelledError()

                    chunk_data = chunk_tensor.detach().cpu().numpy().flatten()
                    if not prefix_processed:
                        prefix_buffer = numpy.concatenate([prefix_buffer, chunk_data])
                        if len(prefix_buffer) >= max_prefix_samples:
                            valid_audio = self._trim_prefix(prefix_buffer, sample_rate)
                            await self._send_audio_chunks(valid_audio, sample_rate)
                            prefix_processed = True
                            prefix_buffer = None
                    else:
                        await self._send_audio_chunks(chunk_data, sample_rate)
                        await asyncio.sleep(0) 

                if not prefix_processed and len(prefix_buffer) > 0:
                     valid_audio = self._trim_prefix(prefix_buffer, sample_rate)
                     await self._send_audio_chunks(valid_audio, sample_rate)

                if is_last:
                    await self.write_event(AudioStop().event())
                
            except asyncio.CancelledError:
                await self.write_event(AudioStop().event())
                raise 
            except Exception as e:
                _LOGGER.error("Fehler: %s", e)
                return False
        return True

    def _trim_prefix(self, audio_data, sample_rate):
        if len(audio_data) == 0: return audio_data
        if len(audio_data) < int(sample_rate * 0.6):
            return audio_data[int(sample_rate * 0.04):]
        threshold = numpy.abs(audio_data).max() * 0.01
        min_prefix = int(sample_rate * PREFIX_MIN_DURATION)
        max_prefix = int(sample_rate * PREFIX_MAX_DURATION)
        silence_gap = int(sample_rate * PREFIX_SILENCE_GAP)
        prefix_end = 0
        search_end = min(len(audio_data), max_prefix)
        is_silent = numpy.abs(audio_data[:search_end]) < threshold
        i = min_prefix
        while i < search_end:
            if is_silent[i]:
                start = i
                while i < search_end and is_silent[i]: i += 1
                if (i - start) >= silence_gap:
                    prefix_end = i
                    break
            i += 1
        return audio_data[prefix_end:] if prefix_end > 0 else audio_data[int(sample_rate * 0.04):]

    async def _send_audio_chunks(self, float_audio, rate):
        if len(float_audio) == 0: return
        audio_int16 = (float_audio.clip(-1.0, 1.0) * 32767).astype("int16")
        audio_bytes = audio_int16.tobytes()
        chunk_size = 2048 
        for i in range(0, len(audio_bytes), chunk_size):
            await self.write_event(AudioChunk(rate=rate, width=2, channels=1, audio=audio_bytes[i:i+chunk_size]).event())

async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10201)
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument("--variant", default=MODEL_VARIANT)
    parser.add_argument("--uri", default=None)
    args, _ = parser.parse_known_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    try:
        tts_model = TTSModel.load_model(args.variant)
    except:
        tts_model = TTSModel.load_model(model_variant_id=args.variant)
    voices = [TtsVoice(name=v, description=f"Pocket-TTS {v}", attribution=Attribution(name="Kyutai", url="https://kyutai.org"),
              installed=True, languages=["en"], version="1.0.1", speakers=None) for v in PREDEFINED_VOICES]
    wyoming_info = Info(tts=[TtsProgram(name="pocket-tts", description="Fast Streaming Pocket-TTS",
        attribution=Attribution(name="Kyutai", url="https://kyutai.org"), installed=True, voices=voices, 
        version="1.0.1", supports_synthesize_streaming=True)])
    if args.uri is None: args.uri = f"tcp://{args.host}:{args.port}"
    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Server bereit auf %s", args.uri)
    await server.run(partial(PocketTTSEventHandler, wyoming_info, args, tts_model))

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: pass
