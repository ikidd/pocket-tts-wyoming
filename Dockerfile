# Wir nutzen das offizielle uv Image (sehr schnell & sauber)
FROM ghcr.io/astral-sh/uv:debian

WORKDIR /app

# 1. Systempakete installieren
# WICHTIG: libsndfile1 wird zwingend für Audio-Operationen benötigt!
# git brauchen wir zum Clonen.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 2. Pocket-TTS Repository clonen
# Wir holen uns den aktuellsten Code direkt von der Quelle
RUN git clone https://github.com/kyutai-labs/pocket-tts.git .

# 3. Python Abhängigkeiten installieren
# Schritt A: Wir installieren die Abhängigkeiten, die Pocket-TTS selbst braucht (definiert in deren pyproject.toml)
RUN uv sync

# Schritt B: Wir fügen unsere Wyoming-Zusätze hinzu
# beartype ist wichtig für Pocket-TTS internals, falls es nicht im sync dabei war
RUN uv add "wyoming>=1.8,<2" zeroconf beartype

# 4. Unser optimiertes Streaming-Skript reinkopieren
# (Stelle sicher, dass die Datei lokal im Ordner liegt!)
COPY wyoming_tts_server.py .

# 5. Umgebungsvariablen
ENV WYOMING_PORT=10201 \
    WYOMING_HOST=0.0.0.0 \
    DEFAULT_VOICE=alba \
    MODEL_VARIANT=b6369a24 \
    # WICHTIG: Unbuffered Output für Echtzeit-Logs (DEBUG STREAM)
    PYTHONUNBUFFERED=1 \
    # Cache Pfad für HuggingFace Modelle
    HF_HOME=/data/hf

# Port freigeben
EXPOSE 10201

# Startbefehl
# Wir nutzen "uv run", das sorgt automatisch dafür, dass das Virtual Environment genutzt wird
CMD ["uv", "run", "python3", "wyoming_tts_server.py"]
