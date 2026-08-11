#!/usr/bin/env bash
# One-command install. Sets up reelforge and registers it with Claude as a
# connector named "alivideoedit".
#
#   bash install.sh
#
# Safe to re-run — every step checks before acting.

set -euo pipefail

NAME="${CONNECTOR_NAME:-alivideoedit}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bold() { printf '\033[1m%s\033[0m\n' "$1"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$1"; }
die()  { printf '  \033[31m✗\033[0m %s\n' "$1" >&2; exit 1; }

bold "reelforge installer"
echo

# --- 1. ffmpeg --------------------------------------------------------------
# Not optional: it does every cut, reframe and render. Checked first because
# everything downstream is pointless without it.
if command -v ffmpeg >/dev/null 2>&1; then
  ok "ffmpeg $(ffmpeg -version 2>/dev/null | head -1 | awk '{print $3}')"
else
  warn "ffmpeg is missing — installing"
  if command -v brew >/dev/null 2>&1; then
    brew install ffmpeg
  elif command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -qq && sudo apt-get install -y ffmpeg
  else
    die "install ffmpeg manually: https://ffmpeg.org/download.html"
  fi
  ok "ffmpeg installed"
fi

# --- 2. python --------------------------------------------------------------
PY=""
for c in python3.12 python3.11 python3; do
  if command -v "$c" >/dev/null 2>&1; then
    v=$("$c" -c 'import sys; print(sys.version_info >= (3, 10))' 2>/dev/null || echo False)
    [ "$v" = "True" ] && { PY="$c"; break; }
  fi
done
[ -n "$PY" ] || die "need Python 3.10 or newer"
ok "$($PY --version)"

# --- 3. install into a dedicated venv ---------------------------------------
# A venv rather than the system Python so this can never collide with anything
# else installed, and so uninstalling is deleting one directory.
VENV="$HOME/.reelforge-venv"
if [ ! -d "$VENV" ]; then
  "$PY" -m venv "$VENV"
fi
"$VENV/bin/pip" install --quiet --upgrade pip
echo "  installing (this pulls PyTorch for local transcription — a few minutes)"
"$VENV/bin/pip" install --quiet -e "$HERE[all]"
ok "reelforge $("$VENV/bin/reelforge" --version 2>/dev/null || echo installed)"

BIN="$VENV/bin/reelforge-mcp"
[ -x "$BIN" ] || die "reelforge-mcp missing after install"

# --- 4. register with Claude Code -------------------------------------------
if command -v claude >/dev/null 2>&1; then
  # --scope user, emphatically. `claude mcp add` defaults to PROJECT scope,
  # which registers the server only for the directory it was run from — so the
  # connector exists in the reelforge checkout and is invisible in the folder
  # where the user actually keeps their footage. That failure is silent: the
  # tools simply are not there, with nothing to explain why.
  claude mcp remove --scope user "$NAME" >/dev/null 2>&1 || true
  claude mcp remove "$NAME" >/dev/null 2>&1 || true
  claude mcp add --scope user "$NAME" -- "$BIN"
  ok "Claude Code connector '$NAME' (available in every folder)"
else
  warn "the 'claude' CLI is not installed — skipping Claude Code"
fi

# --- 5. register with Claude Desktop ----------------------------------------
# Written with Python rather than sed because the file is JSON that may already
# contain other servers, and clobbering someone's existing connectors would be
# a rude way to install software.
case "$(uname -s)" in
  Darwin) CFG="$HOME/Library/Application Support/Claude/claude_desktop_config.json" ;;
  Linux)  CFG="$HOME/.config/Claude/claude_desktop_config.json" ;;
  *)      CFG="" ;;
esac

if [ -n "$CFG" ]; then
  mkdir -p "$(dirname "$CFG")"
  NAME="$NAME" BIN="$BIN" CFG="$CFG" "$VENV/bin/python" - <<'PY'
import json, os, pathlib, shutil

cfg = pathlib.Path(os.environ["CFG"])
name, binary = os.environ["NAME"], os.environ["BIN"]

data = {}
if cfg.exists() and cfg.stat().st_size:
    try:
        data = json.loads(cfg.read_text())
    except json.JSONDecodeError:
        # Never silently discard a file we cannot parse — it is the user's
        # config and may hold connectors that took effort to set up.
        backup = cfg.with_suffix(".json.broken")
        shutil.copy2(cfg, backup)
        print(f"  ! existing config was not valid JSON, backed up to {backup.name}")

servers = data.setdefault("mcpServers", {})
servers[name] = {"command": binary}
cfg.write_text(json.dumps(data, indent=2) + "\n")
print(f"  \033[32m✓\033[0m Claude Desktop connector '{name}'")
PY
fi

echo
bold "done"
echo
echo "  Claude Code    already live — just start talking"
echo "  Claude Desktop QUIT AND REOPEN IT (a reload is not enough)"
echo
echo "  Check it worked:  type /mcp in Claude and look for '$NAME'"
echo
echo "  Then try:"
echo "    cd ~/somewhere/with/videos"
echo "    claude"
echo "    > cut these clips into a 30-second Reel with captions"
echo
