#!/usr/bin/env python3
"""Token counter for prompt-distill. Compares before/after token counts."""

import sys
from pathlib import Path

try:
    import tiktoken
except ImportError:
    print("Install tiktoken: pip install tiktoken")
    sys.exit(1)


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))


def main():
    if len(sys.argv) < 2:
        print("Usage: python token_count.py <file> [compare_file]")
        sys.exit(1)

    file1 = Path(sys.argv[1])
    if not file1.exists():
        print(f"File not found: {file1}")
        sys.exit(1)

    text1 = file1.read_text(encoding="utf-8")
    tokens1 = count_tokens(text1)
    lines1 = len(text1.splitlines())

    if len(sys.argv) >= 3:
        file2 = Path(sys.argv[2])
        if not file2.exists():
            print(f"File not found: {file2}")
            sys.exit(1)

        text2 = file2.read_text(encoding="utf-8")
        tokens2 = count_tokens(text2)
        lines2 = len(text2.splitlines())
        saved = tokens1 - tokens2
        pct = (saved / tokens1 * 100) if tokens1 > 0 else 0

        print(f"Before: {tokens1:,} tokens ({lines1} lines)  {file1.name}")
        print(f"After:  {tokens2:,} tokens ({lines2} lines)  {file2.name}")
        print(f"Saved:  {saved:,} tokens ({pct:.1f}%)")
    else:
        print(f"{tokens1:,} tokens ({lines1} lines)  {file1.name}")


if __name__ == "__main__":
    main()
