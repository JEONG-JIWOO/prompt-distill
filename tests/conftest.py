"""DeepEval configuration for prompt-distill compression quality tests."""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Project paths
EXAMPLES_DIR = PROJECT_ROOT / "examples"
BEFORE_FILE = EXAMPLES_DIR / "before-claude.md"
AFTER_FILE = EXAMPLES_DIR / "after-claude.md"

# Judge model — used by DeepEval GEval
JUDGE_MODEL = os.getenv("DEEPEVAL_JUDGE_MODEL", "gpt-4o")

# A/B comparison: compressed score must be >= QUALITY_RATIO × original score
QUALITY_RATIO = 0.9


def generate_response(scenario: str, prompt_file: Path) -> str:
    """Generate an LLM response using the given file as system prompt."""
    from openai import OpenAI

    client = OpenAI()
    system_prompt = prompt_file.read_text(encoding="utf-8")

    try:
        response = client.chat.completions.create(
            model=os.getenv("TEST_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": scenario},
            ],
            max_completion_tokens=2000,
        )
        result = response.choices[0].message.content or ""
        if not result:
            pytest.fail(f"LLM returned empty response for: {scenario[:50]}")
        return result
    except Exception as e:
        pytest.fail(f"LLM API call failed: {e}")
