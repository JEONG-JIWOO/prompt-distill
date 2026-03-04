#!/usr/bin/env bash
# prompt-distill: Run DeepEval compression quality tests
set -e

source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate

echo "=== prompt-distill Compression Quality Test ==="
echo "Judge model: ${DEEPEVAL_JUDGE_MODEL:-gpt-4o}"
echo "Test model:  ${TEST_MODEL:-gpt-4o-mini}"
echo ""

# Run token count first
echo "--- Token Count ---"
python scripts/token_count.py examples/before-claude.md examples/after-claude.md
echo ""

# Run DeepEval tests with verbose output + HTML report
echo "--- DeepEval Tests ---"
mkdir -p test-results
deepeval test run tests/test_compression_quality.py -v -d all -- --html=test-results/report.html --self-contained-html

echo ""
echo "Report saved to: test-results/report.html"
