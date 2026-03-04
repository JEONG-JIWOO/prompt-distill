@echo off
REM prompt-distill: Run DeepEval compression quality tests
setlocal

call .venv\Scripts\activate.bat

echo === prompt-distill Compression Quality Test ===
echo Judge model: %DEEPEVAL_JUDGE_MODEL%
echo Test model:  %TEST_MODEL%
echo.

REM Run token count first
echo --- Token Count ---
python scripts\token_count.py examples\before-claude.md examples\after-claude.md
echo.

REM Run DeepEval tests with verbose output + HTML report
echo --- DeepEval Tests ---
deepeval test run tests\test_compression_quality.py -v -d all -- --html=test-results\report.html --self-contained-html

echo.
echo Report saved to: test-results\report.html

endlocal
