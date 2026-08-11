# reelforge as a hosted remote connector.
#
# Build target: a container that speaks MCP over HTTP so claude.ai — web,
# desktop and mobile alike — can reach it the same way it reaches Higgsfield or
# Canva. Those appear in the Connectors list because they are servers with
# public URLs; this makes reelforge one too.
#
#   docker build -t reelforge .
#   docker run -p 8080:8080 -e REELFORGE_AUTH_TOKEN=... reelforge

FROM python:3.12-slim

# ffmpeg is the whole engine — every cut, reframe and render shells out to it.
# The install is deliberately minimal: no recommended extras, and the apt lists
# are dropped in the same layer so they never reach the image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies resolve from the project metadata alone, so copying it first lets
# Docker cache the slow install layer across source-only changes.
COPY pyproject.toml README.md ./
COPY src ./src

# `remote` rather than `all`: the hosted server has no use for faster-whisper's
# ~1GB of PyTorch, and transcription on a small instance would be miserable
# anyway. Hosted transcription (Scribe) covers it with an API key.
RUN pip install --no-cache-dir -e ".[remote,vision,transcribe]"

# Everything the server can touch lives here. Mount a volume over it to keep
# footage and renders across deploys — without one, a restart is a clean slate.
ENV REELFORGE_WORKSPACE=/data/workspace
RUN mkdir -p /data/workspace

EXPOSE 8080

# Unbuffered so container logs appear in real time rather than in bursts when
# the buffer flushes, which makes a hung render impossible to diagnose.
ENV PYTHONUNBUFFERED=1

# /health needs no credential precisely so this works without baking the secret
# into the image.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD curl -fsS http://127.0.0.1:8080/health || exit 1

CMD ["sh", "-c", "reelforge-mcp --transport streamable-http --host 0.0.0.0 --port ${PORT:-8080}"]
