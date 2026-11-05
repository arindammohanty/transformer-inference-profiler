#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR"

echo "Building 8-slide technical deck via MCP Impress engine..."
export PYTHONPATH="$DIR:$PYTHONPATH"
python3 -m mcp_impress.server --build-deck

echo "✅ Generated slide deck:"
ls -lh "$DIR/slides/presentation.odp" "$DIR/slides/presentation.pdf"
