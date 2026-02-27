# skill-optimizer

> A meta-skill that analyzes markdown instruction files (CLAUDE.md, SKILL.md, agent.md), classifies each element by knowledge tier, and produces an interactive optimization plan for user approval before compression.

## Problem

AI-generated instruction markdown files are chronically verbose:
- 65.2% of requirements in typical prompts are already satisfied by LLM defaults (Yang et al., 2025, "What Prompts Don't Say")
- But fully removing them causes 2x regression risk across model/prompt changes
- The optimal strategy is **not deletion, but tier-based compression**: brief anchoring for known concepts, detailed specification only for novel knowledge

**Additional insight**: Claude 4.x's prompting best practices confirm that prompt formatting style influences output style — verbose CLAUDE.md produces verbose agent output, wasting both input AND output tokens.

## Core Innovation: Plan-First, Then Compress

Unlike blind compression tools, skill-optimizer follows a **plan → review → execute** workflow:

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  1. Analyze  │ ──→ │  2. Classify  │ ──→ │  3. Present  │ ──→ │  4. Execute   │
│  Parse MD    │     │  Tier each    │     │  Plan to     │     │  Compress per │
│  structure   │     │  element      │     │  user        │     │  approved plan│
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
                                               ↕
                                         User confirms,
                                         adjusts tiers,
                                         overrides
```

## Knowledge Tier System

Based on "What Prompts Don't Say" (Yang et al., 2025) findings and Anthropic's official guidance ("Only add context Claude doesn't already have"):

### Tier 1: REMOVE — Model Default Knowledge
Things all LLMs reliably know. Specifying them wastes tokens with zero benefit.

| Category | Examples |
|---|---|
| Language basics | "Use proper indentation", "Follow PEP 8" |
| Universal practices | "Be helpful", "Write clean code", "Handle errors" |
| Format knowledge | "Use markdown headers", "JSON should be valid" |
| Obvious behaviors | "Read the file before editing", "Don't delete important code" |

**Action**: Remove entirely. 0 tokens.
**Risk**: Minimal — 70.7% format-related defaults are reliable (Yang et al.)

### Tier 2: ANCHOR — Brief Keyword Reminder  
Model likely knows this, but a short anchor activates the right behavior and ensures consistency across model versions.

| Category | Before (verbose) | After (anchored) |
|---|---|---|
| Known framework | "When writing TypeScript, always enable strict mode for better type safety..." | `TypeScript: strict mode` |
| Standard tool | "Use ESLint for linting and Prettier for code formatting to ensure consistent style..." | `Lint: ESLint + Prettier` |
| Common pattern | "Follow conventional commits specification for all commit messages. This means using prefixes like feat:, fix:, chore:..." | `Commits: conventional commits` |

**Action**: Compress to keyword/phrase. 2-5 tokens per item.  
**Risk**: Low — anchor prevents regression while saving 80-90% tokens per item.

### Tier 3: CONDENSE — Keep but Make Concise
Project-specific rules or deviations from defaults that Claude cannot infer. These need explicit statement but not verbose explanation.

| Category | Before | After |
|---|---|---|
| Override default | "For error responses, please use the RFC 7807 Problem Details format instead of our older error format which used {error, message} structure..." | `Errors: RFC 7807 Problem Details (not legacy {error, message})` |
| Conditional | "When deploying to staging, always run the migration check script first, but skip it for local development..." | `Deploy staging → run migration check first. Skip for local.` |
| Team convention | "We use a monorepo structure with packages in the /packages directory. Each package has its own tsconfig..." | `Monorepo: /packages/*, each with own tsconfig` |

**Action**: Rewrite in imperative shorthand. 30-60% reduction.
**Risk**: Medium — must preserve all conditional logic and specifics.

### Tier 4: PRESERVE — Keep As-Is or Expand
Truly novel knowledge that the model cannot possibly know. Full specification required.

| Category | Examples |
|---|---|
| Internal API schemas | Custom REST endpoints, field names, auth flows |
| Business logic | Domain-specific rules, calculation formulas |
| Non-obvious workflows | "After X, wait for webhook Y before proceeding to Z" |
| Custom tool usage | Project-specific CLI tools, internal MCP servers |

**Action**: Keep full detail. May even suggest expansion if too terse.
**Risk**: High if compressed — model will hallucinate missing details.

## Workflow Detail

### Step 1: Analyze

```
User: "Optimize my CLAUDE.md" (or /skill-optimizer)
```

Skill reads the target file, parses it into semantic blocks:
- Each heading section
- Each bullet point / instruction
- Each code example
- Frontmatter metadata

Run `scripts/token_count.py` for baseline measurement.

### Step 2: Classify & Score

For each block, assess:

```
{
  "block": "Always use TypeScript strict mode for better type safety...",
  "current_tokens": 12,
  "tier": 2,
  "tier_reason": "TypeScript strict mode is well-known best practice",
  "proposed": "TypeScript: strict mode",
  "proposed_tokens": 4,
  "savings": 8,
  "risk": "low"
}
```

Aggregate into optimization plan.

### Step 3: Present Plan to User

```markdown
## 📊 Optimization Plan for CLAUDE.md

**Current**: 2,847 tokens (187 lines)
**Projected**: 1,050 tokens (68 lines)  
**Savings**: 63.1% (1,797 tokens)

### Tier 1: REMOVE (23 items, saves 847 tokens)

These are LLM default behaviors — removing them has no effect:

| # | Current instruction | Reason | Tokens saved |
|---|---|---|---|
| 1 | "Always write clean, readable code" | Universal default | 8 |
| 2 | "Make sure to handle errors properly" | Universal default | 9 |
| 3 | "Use proper indentation" | Language default | 6 |
| ... | ... | ... | ... |

> ⚠️ Want to keep any of these? Type the numbers to preserve.

### Tier 2: ANCHOR (15 items, saves 534 tokens)

These will be compressed to brief keyword reminders:

| # | Current (verbose) | Proposed (anchor) | Saved |
|---|---|---|---|
| 24 | "When writing TypeScript, always enable strict mode for..." (28 tok) | `TypeScript: strict mode` (4 tok) | 24 |
| 25 | "Use ESLint for linting and Prettier for formatting..." (22 tok) | `Lint: ESLint+Prettier` (5 tok) | 17 |
| ... | ... | ... | ... |

> ✏️ Want to adjust any anchored version? Type the number.

### Tier 3: CONDENSE (12 items, saves 416 tokens)

Project-specific rules, rewritten concisely:

| # | Current | Proposed | Saved |
|---|---|---|---|
| 39 | "For error responses, use RFC 7807..." (45 tok) | "Errors: RFC 7807, not legacy format" (9 tok) | 36 |
| ... | ... | ... | ... |

> 🔍 Review these carefully. Any meaning lost? Type the number.

### Tier 4: PRESERVE (8 items, 0 token change)

These contain project-unique knowledge — kept as-is:

| # | Instruction | Reason |
|---|---|---|
| 51 | Internal API auth flow description | Novel knowledge |
| 52 | Custom deployment pipeline steps | Novel knowledge |
| ... | ... | ... |

> ➕ Any of these need MORE detail?

### Summary

| Tier | Items | Action | Tokens saved |
|---|---|---|---|
| T1: Remove | 23 | Delete | 847 |
| T2: Anchor | 15 | Keyword compress | 534 |
| T3: Condense | 12 | Rewrite concise | 416 |
| T4: Preserve | 8 | Keep as-is | 0 |
| **Total** | **58** | | **1,797 (63.1%)** |

Per-session impact (estimated 20 messages/session):
- Input tokens saved: ~35,940/session
- At $3/MTok: ~$0.11/session
- 50 sessions/day: ~$5.39/day, ~$161.70/month
```

### Step 4: User Confirms / Adjusts

User can:
- **Approve all**: "Looks good, proceed"
- **Override tiers**: "Keep #3 and #7 as-is" (promotes to Tier 4)
- **Adjust anchors**: "#25 should also mention the config file path"
- **Request expansion**: "Add more detail to #52"
- **Batch approve**: "Remove all Tier 1, but let me review Tier 2 one by one"

### Step 5: Execute & Validate

1. Generate optimized file based on approved plan
2. Run fact extraction on both versions
3. Show diff + validation report:

```
✅ 34/34 actionable rules preserved
✅ All conditional logic maintained
✅ No project-specific details lost
⚠️ 23 generic instructions removed (Tier 1)
⚠️ 15 instructions compressed to anchors (Tier 2)

Output: CLAUDE.optimized.md (review before replacing)
```

Original is never overwritten — outputs as `.optimized.md` for user review.

## Architecture

```
skill-optimizer/
├── SKILL.md                      # Core instructions
├── references/
│   ├── knowledge-tiers.md        # Tier classification guide + examples
│   ├── compression-patterns.md   # Before/after patterns for each tier
│   ├── validation-guide.md       # Fact extraction methodology
│   └── model-defaults.md         # Known LLM default behaviors catalog
├── scripts/
│   ├── token_count.py            # Token counting (tiktoken)
│   ├── block_parser.py           # Parse MD into semantic blocks
│   └── diff_report.py            # Generate before/after diff
└── examples/
    ├── plan-example.md           # Sample optimization plan output
    ├── before-claude.md          # Verbose CLAUDE.md sample
    └── after-claude.md           # Optimized version
```

## SKILL.md Frontmatter

```yaml
---
name: skill-optimizer
description: >
  Analyzes and optimizes markdown instruction files (CLAUDE.md, SKILL.md, 
  agent.md) for token efficiency. Creates an interactive optimization plan 
  that classifies each instruction by knowledge tier, then compresses with 
  user approval. Use when user mentions "optimize", "compress", "reduce 
  tokens", "too long", "slim down", or "clean up" for markdown files, 
  CLAUDE.md, skills, or agent instructions. Also use when reviewing any 
  SKILL.md or CLAUDE.md for efficiency.
allowed-tools:
  - Read
  - Write  
  - Bash(python *)
  - Grep
  - Glob
---
```

## Key Design Decisions

### Why plan-first instead of auto-compress?

1. **Trust**: User sees exactly what will change before it happens
2. **Domain knowledge**: Only the user knows which "obvious" rules are actually critical for their project
3. **Education**: The plan teaches users what wastes tokens and why, improving future md authoring
4. **Safety**: The "What Prompts Don't Say" paper shows 2x regression risk from removing unspecified requirements — user oversight mitigates this
5. **Iterative**: User can run optimizer multiple times with different tier thresholds

### Why tiers instead of binary keep/remove?

The research clearly shows:
- Full deletion → regression risk (Yang et al., 2025)
- Full preservation → 65% redundancy (Yang et al., 2025)
- **Anchoring (Tier 2) is the sweet spot**: near-zero tokens, near-zero risk

This is the key innovation. Existing prompt compressors (LLMLingua etc.) treat tokens as binary keep/remove. Our tier system recognizes that **a 2-token anchor can replace a 30-token explanation** with minimal risk.

### Model-awareness

Per the Anthropic skill best practices: "What works perfectly for Opus might need more detail for Haiku."

The plan should note model sensitivity:

```
⚠️ Tier 2 items: Reliable on Opus/Sonnet. If using Haiku, 
consider promoting these to Tier 3 for more explicit instructions.
```

### Cache-friendliness

Reorder the optimized output for prompt caching efficiency:
- **Top**: Static, rarely-changing rules (cache-friendly)
- **Bottom**: Dynamic, frequently-updated rules
- Cached tokens cost 75% less — structural optimization beyond content compression

## Distribution

### Primary: GitHub + one-line install
```bash
git clone https://github.com/user/skill-optimizer ~/.claude/skills/skill-optimizer
```

### Secondary: Submit to anthropics/skills repo

### Packaging: .skill file for sharing

## Success Metrics

| Metric | Target |
|---|---|
| Token reduction | 40-70% on AI-generated md files |
| Information preservation | 100% of Tier 3-4 rules maintained |
| Tier classification accuracy | >90% agreement with expert review |
| User override rate | <15% (indicates good default classification) |
| Plan acceptance rate | >80% approved without major changes |

## Research Foundation

| Finding | Source | Application |
|---|---|---|
| 65.2% of prompt requirements are LLM defaults | Yang et al., 2025 | Tier 1-2 classification basis |
| Unspecified prompts have 2x regression risk | Yang et al., 2025 | Why we anchor (Tier 2) instead of delete |
| Format requirements 70.7% reliable as defaults | Yang et al., 2025 | Tier 1 confidence for format-related items |
| Conditional requirements only 22.9% guessable | Yang et al., 2025 | Tier 3-4 for all conditional logic |
| Prompt style influences output style | Anthropic, 2025 | Verbose md → verbose agent output |
| Opus overtriggers on aggressive phrasing | Anthropic, 2025 | Compress MUST/ALWAYS/CRITICAL |
| "Only add context Claude doesn't already have" | Anthropic Skill Best Practices | Core philosophy |
| SKILL.md under 500 lines recommended | Anthropic Skill Best Practices | Concrete target |
| 20x compression possible with <1.5% loss | Microsoft LLMLingua, 2023 | Upper bound reference |
| Instruction referencing reduces repeated rules | MLMastery, 2025 | Cross-file dedup strategy |

## Future Extensions

- **Watch mode**: Auto-analyze when skills are created/modified (via hooks)
- **Cross-file dedup**: Detect rules duplicated across CLAUDE.md and multiple skills
- **Token budget planner**: "Your total startup context is X tokens, here's a breakdown and how to reach Y"
- **Model-adaptive tiers**: Different compression levels for Haiku vs Sonnet vs Opus
- **Team mode**: Shared tier overrides — "our team always keeps X explicit"
- **Benchmark suite**: Standard corpus of verbose md files + optimized ground truth for evaluation
- **Learning loop**: Track which Tier 1/2 removals users override → improve default classification
