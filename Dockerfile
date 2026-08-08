FROM ghcr.io/astral-sh/uv:debian

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Pin pocket-tts to the commit the wyoming server was written against.
# Newer upstream commits rename DEFAULT_VARIANT -> DEFAULT_LANGUAGE and
# change the model-loading API, breaking this server.
ARG POCKET_TTS_REF=119ca2e618dd38e8907c0bc9609c8d1773853062
RUN git clone https://github.com/kyutai-labs/pocket-tts.git . \
 && git checkout "${POCKET_TTS_REF}"

COPY wyoming_tts_server.py .

RUN uv add "wyoming>=1.8,<2" zeroconf

ENV WYOMING_PORT=10201
ENV WYOMING_HOST=0.0.0.0
ENV DEFAULT_VOICE=alba
ENV MODEL_VARIANT=b6369a24
ENV PYTHONUNBUFFERED=1

EXPOSE 10201

CMD ["uv", "run", "python", "wyoming_tts_server.py"]
