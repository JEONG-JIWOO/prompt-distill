# prompt-distill

A Claude Code skill that analyzes markdown instruction files (CLAUDE.md, SKILL.md, agent prompts) and compresses them using a knowledge-tier system — removing what LLMs already know, anchoring what they likely know, and preserving what's truly novel.

## What it does

Most AI instruction files are 60%+ redundant. Telling Claude to "write clean code" or "use proper indentation" wastes tokens on things it already does by default.

prompt-distill classifies each instruction into 4 tiers:

| Tier | Action | Example |
|---|---|---|
| **T1: Remove** | Delete — LLM default | "Write clean, readable code" |
| **T2: Anchor** | Compress to keyword | "Use ESLint + Prettier for..." → `Lint: ESLint+Prettier` |
| **T3: Condense** | Rewrite concisely | Project-specific rules, shortened |
| **T4: Preserve** | Keep as-is | Internal APIs, business logic |

It presents an interactive plan for your approval before making any changes.

## Usage

```
/skill-optimizer          # or mention "optimize my CLAUDE.md"
```

## Install

```bash
git clone https://github.com/user/prompt-distill ~/.claude/skills/prompt-distill
```

## Project structure

```
prompt-distill/
├── SKILL.md                 # Skill entry point
├── references/              # Knowledge bases the skill references
├── scripts/                 # Helper scripts (token counting, parsing)
├── examples/                # Before/after samples
└── research/                # Research & development (not part of skill)
    ├── design/              # Design documents
    ├── papers/              # Paper summaries & citations
    ├── experiments/         # Benchmarks & test results
    └── notes/               # Ideas & working notes
```

## Research foundation

Based on findings from:
- "What Prompts Don't Say" (Yang et al., 2025) — 65.2% of prompt requirements are LLM defaults
- Anthropic prompting best practices — "Only add context Claude doesn't already have"
- LLMLingua (Microsoft, 2023) — 20x compression with <1.5% quality loss

See [research/](research/) for detailed notes and references.

## Status

Early development — research & design phase.

## License

MIT
