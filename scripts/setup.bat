@echo off
REM prompt-distill: Initial setup (run once)
echo === prompt-distill Setup ===

REM Create venv if not exists
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate and install
call .venv\Scripts\activate.bat
echo Installing dependencies...
pip install tiktoken deepeval openai python-dotenv pytest pytest-html -q

REM Check .env
if not exist ".env" (
    copy .env.example .env
    echo.
    echo [!] Created .env from .env.example
    echo [!] Edit .env and add your OPENAI_API_KEY
    echo.
)

echo.
echo === Setup Complete ===
echo Run scripts\test.bat to execute tests
