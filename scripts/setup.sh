#!/usr/bin/env bash
# prompt-distill: Initial setup (run once)
set -e

echo "=== prompt-distill Setup ==="

# Create venv if not exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate and install
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate
echo "Installing dependencies..."
pip install tiktoken deepeval openai python-dotenv pytest pytest-html -q

# Check .env
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ""
    echo "[!] Created .env from .env.example"
    echo "[!] Edit .env and add your OPENAI_API_KEY"
    echo ""
fi

echo ""
echo "=== Setup Complete ==="
echo "Run scripts/test.sh to execute tests"
