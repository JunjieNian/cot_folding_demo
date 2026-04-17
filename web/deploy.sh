#!/bin/bash
# Build and deploy the COT Folding Map static site.
# Usage: ./deploy.sh [port]
#
# Steps:
#   1. npm run build (produces dist/ without data)
#   2. Symlink dist/data -> public/data (avoids 4.5GB copy)
#   3. Start Node.js static server on the given port

set -e
PORT="${1:-8080}"
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "[1/3] Building frontend..."
npx vite build

echo "[2/3] Linking static data..."
rm -rf dist/data
ln -s "$DIR/public/data" dist/data
echo "  dist/data -> public/data"

echo "[3/3] Starting server on port $PORT..."
node serve_node.mjs "$PORT"
