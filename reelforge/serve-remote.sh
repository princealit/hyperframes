#!/usr/bin/env bash
# Serve reelforge from THIS machine with a public HTTPS URL, so it can be added
# to Claude as a connector and reached from web and phone.
#
#   bash serve-remote.sh
#
# Free, and faster than a small cloud instance — your laptop has more CPU and
# RAM than an entry-level VPS, and your footage is already on it, so nothing has
# to be uploaded anywhere. The trade is that the machine has to stay awake and
# this window has to stay open.

set -euo pipefail

PORT="${PORT:-8080}"
WORKSPACE="${REELFORGE_WORKSPACE:-$HOME/reelforge-workspace}"
VENV="$HOME/.reelforge-venv"
BIN="$VENV/bin/reelforge-mcp"

bold() { printf '\033[1m%s\033[0m\n' "$1"; }
die()  { printf '  \033[31m✗\033[0m %s\n' "$1" >&2; exit 1; }

[ -x "$BIN" ] || die "reelforge is not installed — run: bash install.sh"
mkdir -p "$WORKSPACE"

# --- token ------------------------------------------------------------------
# Persisted rather than regenerated, because the token is pasted into Claude's
# connector settings. A fresh one every launch would silently break the
# connector each time this script restarts.
TOKEN_FILE="$HOME/.reelforge-token"
if [ ! -f "$TOKEN_FILE" ]; then
  "$VENV/bin/python" -c 'import secrets; print(secrets.token_urlsafe(32))' > "$TOKEN_FILE"
  chmod 600 "$TOKEN_FILE"
fi
TOKEN="$(cat "$TOKEN_FILE")"

# --- tunnel -----------------------------------------------------------------
# cloudflared over ngrok: its quick tunnels need no account and no card, and do
# not put an interstitial warning page in front of the URL — which an API client
# like Claude cannot click through.
if ! command -v cloudflared >/dev/null 2>&1; then
  bold "installing cloudflared (free, no account needed)"
  if command -v brew >/dev/null 2>&1; then
    brew install cloudflared
  else
    die "install cloudflared: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
  fi
fi

cleanup() { kill "${SERVER_PID:-}" "${TUNNEL_PID:-}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

bold "starting reelforge"
REELFORGE_AUTH_TOKEN="$TOKEN" \
  "$BIN" --transport streamable-http --host 127.0.0.1 --port "$PORT" \
  --workspace "$WORKSPACE" > /tmp/reelforge-server.log 2>&1 &
SERVER_PID=$!

# Wait for the port rather than sleeping a fixed guess — a cold start that takes
# longer than the guess would otherwise tunnel to nothing.
for _ in $(seq 1 30); do
  curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  sleep 1
done
curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 \
  || { tail -20 /tmp/reelforge-server.log; die "server did not start"; }

bold "opening tunnel"
cloudflared tunnel --url "http://127.0.0.1:$PORT" --no-autoupdate \
  > /tmp/reelforge-tunnel.log 2>&1 &
TUNNEL_PID=$!

URL=""
for _ in $(seq 1 45); do
  URL="$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' /tmp/reelforge-tunnel.log 2>/dev/null | head -1 || true)"
  [ -n "$URL" ] && break
  sleep 1
done
[ -n "$URL" ] || { tail -20 /tmp/reelforge-tunnel.log; die "tunnel did not open"; }

# Prove the public URL actually reaches this server before printing setup
# instructions for it — a tunnel that resolves but does not route is otherwise
# indistinguishable from success until Claude fails to connect.
curl -fsS "$URL/health" >/dev/null 2>&1 || die "tunnel opened but does not reach the server"

echo
bold "ready — add this to Claude"
echo
echo "  + → Connectors → Add custom connector"
echo
echo "    Name    alivideoedit"
echo "    URL     $URL/mcp"
echo "    Header  Authorization: Bearer $TOKEN"
echo
echo "  workspace  $WORKSPACE"
echo "             put footage here, or use import_media with a share link"
echo
printf '  \033[33m!\033[0m the URL changes each time this restarts — repaste it when it does\n'
printf '  \033[33m!\033[0m keep this window open and the Mac awake (caffeinate -i is your friend)\n'
echo
bold "ctrl-c to stop"
wait
