#!/usr/bin/env python3
"""
Wyoming Protocol TTS Server for Pocket-TTS (Official Repo Version)
Optimized for:
- Sentence-based Text-to-Audio Streaming (LLM Support)
- Correct Home Assistant Debug Timings (0s Fix)
- Real-time Audio Streaming
"""

import argparse
import asyncio
import logging
import os
import re
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

# Tuning für das Prefix-Trimming
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
        
        # Streaming State
        self.is_streaming: bool = False
        self._text_buffer: str = ""
        self._voice_info: Optional[str] = None

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            return True

        try:
            # 1. KLASSISCHES TTS (Ein ganzer Textblock auf einmal)
            if Synthesize.is_type(event.type):
                synthesize = Synthesize.from_event(event)
                _LOGGER.info("Klassische Anfrage: '%s'", synthesize.text)
                
                # Startsignal an HA (Timer startet)
                await self.write_event(SynthesizeStart(voice=synthesize.voice).event())
                
                await self._handle_synthesize(synthesize.text, synthesize.voice)
                
                # Stoppsignal an HA (Timer stoppt)
                await self.write_event(SynthesizeStopped().event())
                return True

            # 2. START TEXT-STREAMING (LLM schickt Text stückweise)
            if SynthesizeStart.is_type(event.type):
                start_event = SynthesizeStart.from_event(event)
                _LOGGER.info("LLM Text-Stream gestartet")
                
                # Bestätige HA den Start (Timer startet)
                await self.write_event(start_event.event())
                
                self.is_streaming = True
                self._text_buffer = ""
                self._voice_info = start_event.voice
                return True

            # 3. TEXT-CHUNK EMPFANGEN (Ein Wort oder Zeichen vom LLM)
            if SynthesizeChunk.is_type(event.type):
                chunk = SynthesizeChunk.from_event(event)
                self._text_buffer += chunk.text
                
                # Wir suchen nach Satzzeichen (., !, ?, \n)
                # Wir splitten den Text, um ganze Sätze zu sprechen (bessere Betonung)
                if any(c in self._text_buffer for c in ".!?\n"):
                    # Regex split bei Satzzeichen, behält das Zeichen aber bei
                    parts = re.split(r'([.!?\n]+)', self._text_buffer)
                    if len(parts) > 2:
                        # Alles außer dem letzten Teil (der evtl. unvollständig ist)
                        to_speak = "".join(parts[:-1]).strip()
                        self._text_buffer = parts[-1]
                        
                        if to_speak:
                            _LOGGER.info("Satz erkannt: '%s'", to_speak)
                            await self._handle_synthesize(to_speak, self._voice_info)
                return True

            # 4. ENDE TEXT-STREAMING (LLM ist fertig)
            if SynthesizeStop.is_type(event.type):
                # Rest im Puffer sprechen
                if self._text_buffer.strip():
                    _LOGGER.info("Letzter Rest im Stream: '%s'", self._text_buffer.strip())
                    await self._handle_synthesize(self._text_buffer.strip(), self._voice_info)
                
                # Signal an HA: Alles erledigt (Timer stoppt)
                await self.write_event(SynthesizeStopped().event())
                _LOGGER.info("LLM Text-Stream beendet.")
                self.is_streaming = False
                self._text_buffer = ""
                return True

            return True
        except Exception as err:
            _LOGGER.error("Fehler im Handler: %s", err, exc_info=True)
            await self.write_event(Error(text=str(err), code="HandlerError").event())
            return False

    async def _handle_synthesize(self, text: str, voice_obj: Optional[any]) -> bool:
        """Kern-Logik zur Generierung und zum Audio-Streaming"""
        text_to_speak = text.strip()
        if not text_to_speak:
            return True

        # Sacrificial Prefix gegen Wortverschlucken
        processed_text = "... " + text_to_speak
        
        # Stimme ermitteln
        voice_name = voice_obj.name if voice_obj else self.cli_args.voice
        if voice_name and voice_name.startswith("pocket-tts-"):
            voice_name = voice_name.replace("pocket-tts-", "", 1)
        if voice_name not in PREDEFINED_VOICES:
            voice_name = self.cli_args.voice

        async with _VOICE_LOCK:
            # Voice State laden / cachen
            if voice_name not in _VOICE_STATES:
                if hasattr(self.tts_model, 'get_state_for_audio_prompt'):
                    _VOICE_STATES[voice_name] = self.tts_model.get_state_for_audio_prompt(voice_name)
                else:
                    _VOICE_STATES[voice_name] = self.tts_model.get_state_for_voice(voice_name)
            
            voice_state = _VOICE_STATES[voice_name]

            try:
                sample_rate = self.tts_model.sample_rate
                
                # Audio-Startsignal (Wichtig für das HA Medien-Player Modul)
                await self.write_event(AudioStart(rate=sample_rate, width=2, channels=1).event())

                # Pocket-TTS Stream Generator
                audio_stream = self.tts_model.generate_audio_stream(
                    model_state=voice_state, text_to_generate=processed_text, copy_state=True
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
                            await self._send_audio_chunks(valid_audio, sample_rate)
                            prefix_processed = True
                            prefix_buffer = None
                    else:
                        # ECHTES STREAMING: Sofort senden
                        await self._send_audio_chunks(chunk_data, sample_rate)
                        await asyncio.sleep(0) # Event Loop freigeben

                # Falls der Stream endet bevor der Buffer voll war
                if not prefix_processed and len(prefix_buffer) > 0:
                     valid_audio = self._trim_prefix(prefix_buffer, sample_rate)
                     await self._send_audio_chunks(valid_audio, sample_rate)

                # Audio-Stop (Ende dieses Satzes/Segments)
                await self.write_event(AudioStop().event())
                
            except Exception as e:
                _LOGGER.error("Fehler bei Generierung: %s", e)
                return False
        return True

    def _trim_prefix(self, audio_data, sample_rate):
        """Findet die Stille nach '...' und schneidet davor ab"""
        if len(audio_data) == 0: return audio_data
        threshold = numpy.abs(audio_data).max() * 0.01
        min_prefix = int(sample_rate * PREFIX_MIN_DURATION)
        max_prefix = int(sample_rate * PREFIX_MAX_DURATION)
        silence_gap = int(sample_rate * PREFIX_SILENCE_GAP)
        
        prefix_end = 0
        search_end = min(len(audio_data), max_prefix)
        if len(audio_data) > min_prefix:
            is_silent = numpy.abs(audio_data[:search_end]) < threshold
            i = min_prefix
            while i < search_end:
                if is_silent[i]:
                    start = i
                    while i < search_end and is_silent[i]: i += 1
                    if (i - start) >= silence_gap:
                        prefix_end = i
                        break
                else: i += 1
        return audio_data[prefix_end:] if prefix_end > 0 else audio_data[min_prefix:]

    async def _send_audio_chunks(self, float_audio, rate):
        """Wandelt Float in Int16 und sendet kleine Chunks"""
        if len(float_audio) == 0: return
        _LOGGER.info("DEBUG STREAM: Sende %d Samples", len(float_audio))
        
        audio_int16 = (float_audio.clip(-1.0, 1.0) * 32767).astype("int16")
        audio_bytes = audio_int16.tobytes()
        
        # Wyoming Standard: 1024 Samples pro Chunk
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
    parser.add_argument("--zeroconf", nargs="?", const="pocket-tts")
    args, _ = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    _LOGGER.info("Lade Pocket-TTS Modell (%s)...", args.variant)
    
    # Robuster Lade-Vorgang (Positional Arguments fix)
    try:
        tts_model = TTSModel.load_model(args.variant)
    except Exception:
        try:
            tts_model = TTSModel.load_model(model_variant_id=args.variant)
        except Exception:
            tts_model = TTSModel.load_model(variant_id=args.variant)

    _LOGGER.info("Pre-loading Voices...")
    for v in PREDEFINED_VOICES:
        try:
            if hasattr(tts_model, 'get_state_for_audio_prompt'):
                _VOICE_STATES[v] = tts_model.get_state_for_audio_prompt(v)
            else:
                _VOICE_STATES[v] = tts_model.get_state_for_voice(v)
        except: pass

    # Wyoming Info
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
