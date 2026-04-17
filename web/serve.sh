#!/bin/bash
# Serve the COT Folding Map static site
# Usage: ./serve.sh [port]

PORT="${1:-8080}"
DIR="$(cd "$(dirname "$0")" && pwd)/dist"

echo "============================================"
echo "  COT Folding Map - AIME24 (Static)"
echo "  Serving: $DIR"
echo "  URL:     http://0.0.0.0:$PORT"
echo "============================================"

cd "$DIR"
python3 -m http.server "$PORT" --bind 0.0.0.0
