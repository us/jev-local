#!/usr/bin/env bash
# One-line installer for jev-local:
#   curl -fsSL https://raw.githubusercontent.com/us/jev-local/main/install.sh | bash
#
# Clones (or reuses) the repo, starts the API with Docker Compose,
# waits for /health, runs a smoke test, prints usage.
set -euo pipefail

REPO_URL="https://github.com/us/jev-local.git"
DIR="${JEVLOCAL_DIR:-$HOME/jev-local}"
PORT="${PORT:-8000}"

need() { command -v "$1" >/dev/null 2>&1 || { echo "missing: $1 (please install it first)" >&2; exit 1; }; }
need git
need docker
docker compose version >/dev/null 2>&1 || { echo "missing: docker compose plugin" >&2; exit 1; }

if [ -d "$DIR/.git" ]; then
  echo "-> using existing checkout at $DIR"
  git -C "$DIR" pull --ff-only 2>/dev/null || true
else
  echo "-> cloning into $DIR"
  git clone --depth 1 "$REPO_URL" "$DIR"
fi
cd "$DIR"

echo "-> starting jev-local on port $PORT"
PORT="$PORT" docker compose up -d --build

echo "-> waiting for http://127.0.0.1:$PORT/health"
for _ in $(seq 1 30); do
  if curl -fs -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "-> API is up"
    break
  fi
  sleep 4
  if [ "$_ " = "30 " ]; then echo "server did not start; run: docker compose logs" >&2; exit 1; fi
done

echo "-> smoke test"
curl -fs -m 15 -X POST "http://127.0.0.1:$PORT/v1/systemone" \
  -H 'Content-Type: application/json' \
  -d '{"state":"Payouts failing for 3 days, help ASAP.","model":"jev-latest","questions":{"u":{"type":"noul","instructions":"Does this convey urgency?"}}}}'
echo
echo "done. API ready at http://127.0.0.1:$PORT"
echo "Point typesafe-sdk at it with base_url=\"http://127.0.0.1:$PORT\"."
