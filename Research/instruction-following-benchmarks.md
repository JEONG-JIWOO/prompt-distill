# Instruction-Following Benchmarks: Research for CLAUDE.md Compression Evaluation

## Use Case Context

We compress a CLAUDE.md instruction file and need to verify the compressed version still makes the LLM follow all the original instructions. This document evaluates existing benchmarks and tools for this purpose, culminating in a recommended automated pipeline.

---

## 1. IFEval (Google, 2023)

### How It Works (Mechanically)

IFEval defines **25 types of "verifiable instructions"** -- instructions that can be checked programmatically without any LLM judge. Examples:

- "Write in more than 400 words" --> word count check
- "Mention the keyword 'AI' at least 3 times" --> regex/string count
- "Your entire output should be in JSON" --> JSON parser validation
- "Include a title in double square brackets [[title]]" --> regex match
- "Do not use any commas" --> character scan
- "Write exactly 3 bullet points" --> bullet counter
- "Response must be in all lowercase" --> case check

Each of the 500 prompts bundles 1+ verifiable instructions. The evaluation is purely deterministic: run the LLM, then run Python checker functions against the output.

**Four metrics computed:**
- Prompt-level strict accuracy (all instructions in a prompt must pass)
- Prompt-level loose accuracy (minor formatting tolerance)
- Instruction-level strict accuracy (per-instruction pass rate)
- Instruction-level loose accuracy

### Resources

| Resource | URL |
|----------|-----|
| Paper | https://arxiv.org/abs/2311.07911 |
| HuggingFace Dataset | https://huggingface.co/datasets/google/IFEval |
| Google Research GitHub | https://github.com/google-research/google-research/tree/master/instruction_following_eval |
| EleutherAI Harness Integration | https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/ifeval/README.md |
| Clean reimplementation | https://github.com/oKatanaaa/ifeval |

### Can It Evaluate Custom Instruction Files?

**Not directly.** IFEval's 25 instruction types are hardcoded format constraints (word counts, JSON format, keyword frequency). They do NOT test semantic/behavioral instructions like "always suggest running tests before committing" or "use TypeScript over JavaScript when possible."

**However**, the IFEval *methodology* (programmatic verification) is extremely valuable. For a CLAUDE.md file, you could:
1. Extract each instruction from CLAUDE.md
2. For instructions that ARE programmatically verifiable (e.g., "always include file paths," "never use emojis"), write custom checker functions
3. For semantic instructions, fall back to LLM-as-judge

### API Cost Estimate

**$0 for evaluation itself** -- IFEval uses no LLM judge. The only API cost is generating the model's responses to the 500 prompts. For a custom version testing your CLAUDE.md, cost = (number of test prompts) x (cost per LLM call for the model being tested).

### Setup Difficulty

**Easy.** The EleutherAI lm-evaluation-harness has IFEval built in. Run with:
```bash
lm_eval --model openai --model_args model=gpt-4 --tasks ifeval --batch_size 1
```

For custom tests: **Medium.** You need to write your own checker functions per instruction type.

---

## 2. InFoBench (DRFR Metric)

### How It Works (Mechanically)

InFoBench introduces the **Decomposed Requirement Following Ratio (DRFR)**:

1. Take a complex instruction (e.g., "Write a formal email in Spanish about project delays, keeping it under 200 words with exactly 3 paragraphs")
2. **Decompose** it into atomic sub-requirements:
   - Is the email formal in tone?
   - Is it written in Spanish?
   - Is it about project delays?
   - Is it under 200 words?
   - Does it have exactly 3 paragraphs?
3. For each sub-requirement, ask a **judge LLM** (GPT-4): "Does this output satisfy [requirement]? Yes/No"
4. DRFR = (number of satisfied requirements) / (total requirements)

The benchmark has 500 instructions with 2,250 pre-decomposed questions (average 4.5 requirements per instruction), split into Easy and Hard sets.

### Resources

| Resource | URL |
|----------|-----|
| Paper | https://arxiv.org/abs/2401.03601 |
| GitHub | https://github.com/qinyiwei/InfoBench |
| HuggingFace Dataset | https://huggingface.co/datasets/kqsong/InFoBench |
| ACL 2024 Proceedings | https://aclanthology.org/2024.findings-acl.772/ |

### Can It Evaluate Custom Instruction Files?

**YES -- this is the most directly applicable methodology for CLAUDE.md evaluation.** The DRFR approach maps perfectly:

1. Feed CLAUDE.md to an LLM with prompt: "Decompose each instruction in this document into atomic, independently testable requirements"
2. For each requirement, generate a test scenario (a user message that would trigger that instruction)
3. Run the model with the compressed CLAUDE.md as system prompt
4. Use GPT-4 as judge to check each decomposed requirement against the output

This is essentially what we want to build.

### API Cost Estimate

**~$5-15 per evaluation run.** Each test case requires:
- 1 LLM call to generate the response (~input: system prompt + user message)
- N judge calls (one per decomposed requirement, ~$0.01-0.03 each with GPT-4)

For 500 instructions x 4.5 requirements = 2,250 judge calls. At ~$0.01/call = ~$22. But for a CLAUDE.md with ~30-50 instructions, cost would be much lower: ~$2-5.

### Setup Difficulty

**Medium.** The GitHub repo provides working code. Main adaptation work: writing the decomposition prompt and test scenarios for your specific CLAUDE.md.

---

## 3. FollowBench

### How It Works (Mechanically)

FollowBench introduces a **multi-level constraint escalation** approach:

- **Level 1**: Base instruction with 1 constraint (e.g., "Write a poem")
- **Level 2**: Same instruction + 1 more constraint (e.g., "Write a poem in sonnet form")
- **Level 3**: +1 more (e.g., "Write a poem in sonnet form about nature")
- **Level 4**: +1 more (e.g., "Write a poem in sonnet form about nature, using no words with the letter 'e'")
- **Level 5**: +1 more

This incrementally reveals where models break down. Five constraint categories:
1. **Content** constraints (topic, entities to include)
2. **Situation** constraints (role, context)
3. **Style** constraints (tone, voice)
4. **Format** constraints (structure, length)
5. **Example** constraints (follow a given pattern)

Evaluation uses a strong LLM (GPT-4) as judge with **constraint-evolution paths** -- the judge sees how each constraint was added, making it easier to verify each one individually.

### Resources

| Resource | URL |
|----------|-----|
| Paper | https://arxiv.org/abs/2310.20410 |
| GitHub | https://github.com/YJiangcm/FollowBench |
| HuggingFace Dataset | https://huggingface.co/datasets/YuxinJiang/FollowBench |
| ACL 2024 Proceedings | https://aclanthology.org/2024.acl-long.257/ |

### Can It Evaluate Custom Instruction Files?

**Partially.** The multi-level methodology is interesting for understanding *how many simultaneous instructions* a compressed prompt can handle, but the benchmark's fixed task set (NLP tasks) is not directly applicable to CLAUDE.md-style behavioral instructions.

The **constraint categories** (Content, Situation, Style, Format, Example) could inform how you categorize CLAUDE.md instructions for testing.

### API Cost Estimate

**~$10-20 per evaluation run.** Uses GPT-4 as judge. ~1,000+ prompts across all levels.

### Setup Difficulty

**Medium.** GitHub repo is well-organized with clear scripts. Custom adaptation would require significant effort to map CLAUDE.md instructions to the multi-level framework.

---

## 4. ManyIFEval

### How It Works (Mechanically)

ManyIFEval extends IFEval to test what happens when you pile on **many instructions simultaneously** (up to 10 per prompt). Key finding: **performance degrades consistently as instruction count increases**, and a logistic regression model can predict this degradation with ~10% error.

Like IFEval, all instructions are programmatically verifiable (no LLM judge needed). The benchmark measures whether models can satisfy all constraints simultaneously when there are many of them.

### Resources

| Resource | URL |
|----------|-----|
| Paper | https://arxiv.org/abs/2509.21051 |
| ACL EMNLP 2025 | https://aclanthology.org/2025.findings-emnlp.896 |

Note: No public GitHub repo or HuggingFace dataset was found in my search. The paper may have accompanying code linked from the arXiv page.

### Can It Evaluate Custom Instruction Files?

**Conceptually very relevant but not directly usable.** A CLAUDE.md IS essentially "many simultaneous instructions" -- exactly what this benchmark studies. The finding that performance degrades with instruction count is directly relevant to prompt compression: a compressed prompt that maintains fewer but more impactful instructions might actually perform better.

### API Cost Estimate

**$0 for evaluation** (programmatic verification like IFEval). Cost is only for generating model responses.

### Setup Difficulty

**Hard** (no public code available). Would need to implement from the paper.

---

## 5. Arena-Hard-Auto / Auto-Arena

### How It Works (Mechanically)

**Arena-Hard-Auto** (LMSYS):
1. 500 challenging real-world prompts curated from Chatbot Arena
2. Model generates response to each prompt
3. A **baseline model** (GPT-4-0314) also generates a response
4. A **judge model** (GPT-4-Turbo or GPT-4.1) does pairwise comparison: "Which response is better?"
5. Uses **CoT prompting** for the judge -- it reasons before deciding
6. **Position debiasing**: runs each comparison twice with swapped positions (500x2 = 1,000 judgments)
7. **Bradley-Terry model** converts pairwise preferences into Elo-like scores

**Auto-Arena** (separate project):
Three-stage pipeline:
1. LLM examiner generates questions
2. Two LLM candidates engage in multi-round peer debate
3. Committee of LLM judges collaboratively discusses and decides the winner
Claims 95% correlation with human rankings.

### Resources

| Resource | URL |
|----------|-----|
| Arena-Hard-Auto GitHub | https://github.com/lmarena/arena-hard-auto |
| Arena-Hard-Auto v0.1 Dataset | https://huggingface.co/datasets/lmarena-ai/arena-hard-auto-v0.1 |
| LMSYS Blog Post | https://lmsys.org/blog/2024-04-19-arena-hard/ |
| Auto-Arena Project Page | https://auto-arena.github.io/blog/ |

### Can It Evaluate Custom Instruction Files?

**Not directly applicable.** Arena-Hard evaluates general model quality, not instruction following from a specific system prompt. You COULD adapt it by:
1. Using your CLAUDE.md as the system prompt for the "model" responses
2. Using the compressed CLAUDE.md as the system prompt for the "baseline"
3. Having GPT-4 judge which response better follows the original instructions

But this is an expensive, roundabout approach compared to direct instruction checking.

### API Cost Estimate

**~$20-50 per model evaluated.** 1,000 judge calls with GPT-4-Turbo/GPT-4.1. At current GPT-4.1 pricing ($2/1M input, $8/1M output), each judgment uses ~2K tokens in + ~500 tokens out, so ~1000 * ($0.004 + $0.004) = ~$8. But generating the 500 model responses adds to cost depending on the model.

### Setup Difficulty

**Medium.** Well-documented repo with clear config files. Supports OpenAI-compatible API endpoints.

---

## 6. MT-Bench

### How It Works (Mechanically)

1. **80 multi-turn questions** across 8 categories (writing, roleplay, extraction, reasoning, math, coding, STEM, humanities)
2. Each question has **2 turns**: initial request + follow-up
3. Model generates responses to both turns
4. **GPT-4 as judge** scores each turn on a 1-10 scale
5. Final score per question = minimum of the two turn scores (weakest-link scoring)
6. Total: 160 evaluations (80 questions x 2 turns)

### Resources

| Resource | URL |
|----------|-----|
| Paper | https://arxiv.org/abs/2306.05685 |
| GitHub (FastChat) | https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge |
| Question Data | https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/data/mt_bench/question.jsonl |

### Can It Evaluate Custom Instruction Files?

**Not directly.** MT-Bench tests general conversational ability, not compliance with a specific instruction file. However, the multi-turn format is relevant since CLAUDE.md instructions often apply across conversation turns.

You could theoretically replace the 80 questions with custom questions designed to test CLAUDE.md compliance, keeping the 2-turn format and GPT-4-as-judge scoring.

### API Cost Estimate

**~$5 per evaluation run.** 160 judge calls with GPT-4. Very economical.

### Setup Difficulty

**Easy.** FastChat has a well-maintained CLI:
```bash
python gen_model_answer.py --model-path [MODEL] --model-id [ID]
python gen_judgment.py --model-list [ID] --judge-model gpt-4
python show_result.py
```

---

## 7. AlpacaEval

### How It Works (Mechanically)

1. **805 diverse instruction-following tasks** from the AlpacaFarm evaluation set
2. Model generates a response to each instruction
3. A **baseline model** (GPT-4-Turbo) also generates a reference response
4. **GPT-4-Turbo as judge** compares the two responses pairwise
5. Outputs: probability that the judge prefers the evaluated model's response
6. **Win rate** = average preference probability across all 805 instructions
7. **AlpacaEval 2.0** adds **length-controlled** win rate (debiases for verbosity) using a GLM that predicts: "What would the preference be if both outputs had the same length?"

Length-controlled version achieves 0.98 Spearman correlation with Chatbot Arena.

### Resources

| Resource | URL |
|----------|-----|
| GitHub | https://github.com/tatsu-lab/alpaca_eval |
| Leaderboard | https://tatsu-lab.github.io/alpaca_eval/ |
| Paper (LC-AlpacaEval) | https://arxiv.org/abs/2404.04475 |

### Can It Evaluate Custom Instruction Files?

**Not directly, but partially adaptable.** Like Arena-Hard, it compares model outputs rather than checking specific instruction compliance. You could use it in a "A/B test" setup:
- Model A: LLM with original CLAUDE.md
- Model B: LLM with compressed CLAUDE.md
- Win rate tells you if compression degraded quality

But it would NOT tell you WHICH specific instructions were lost.

### API Cost Estimate

**< $10 per evaluation run** with GPT-4-Turbo. One of the cheapest LLM-as-judge benchmarks.

### Setup Difficulty

**Easy.** Excellent CLI:
```bash
alpaca_eval --model_outputs outputs.json
```

---

## Comparative Summary Table

| Benchmark | Evaluation Method | Custom Instructions? | Cost/Run | Setup | Best For |
|-----------|------------------|---------------------|----------|-------|----------|
| **IFEval** | Programmatic checkers | No (format only) | $0 (eval) | Easy | Verifiable format constraints |
| **InFoBench/DRFR** | LLM judge per requirement | **YES** | $2-15 | Medium | **Decomposed instruction checking** |
| **FollowBench** | LLM judge, multi-level | Partial | $10-20 | Medium | Constraint escalation testing |
| **ManyIFEval** | Programmatic checkers | Conceptual only | $0 (eval) | Hard | Many-instruction degradation |
| **Arena-Hard** | LLM pairwise comparison | No (A/B only) | $20-50 | Medium | Overall quality comparison |
| **MT-Bench** | LLM 1-10 scoring | No (general) | ~$5 | Easy | Multi-turn conversation |
| **AlpacaEval** | LLM pairwise comparison | No (A/B only) | <$10 | Easy | Overall quality A/B testing |

---

## Creating Custom IFEval-Style Tests from a CLAUDE.md File

### The Core Challenge

CLAUDE.md instructions are NOT like IFEval instructions. IFEval tests format constraints ("write 400 words," "use JSON"). CLAUDE.md contains **behavioral instructions** like:
- "Always suggest running tests before committing"
- "Use TypeScript over JavaScript when possible"
- "Never expose API keys in code examples"
- "Prefer functional programming patterns"

These require **semantic verification** (LLM-as-judge), not programmatic checks.

### Recommended Approach: InFoBench-Style DRFR Pipeline

#### Step 1: Decompose CLAUDE.md into Atomic Instructions

Use an LLM to extract every instruction:

```
Prompt: "Read the following CLAUDE.md file and extract every individual instruction,
rule, preference, or behavioral guideline. Output them as a numbered list.
Each item should be a single, independently testable requirement.

[CLAUDE.md content]"
```

Example output:
1. Always use TypeScript instead of JavaScript
2. Include file paths in code examples
3. Never use emojis in responses
4. Run tests before suggesting commits
5. Prefer functional programming patterns
...

#### Step 2: Classify Instructions by Verifiability

For each instruction, determine if it is:
- **Programmatically verifiable** (like IFEval): "never use emojis" -> regex check
- **Semantically verifiable** (like InFoBench/DRFR): "prefer functional programming" -> LLM judge
- **Context-dependent**: "suggest tests before committing" -> needs specific scenario

#### Step 3: Generate Test Scenarios

For each instruction, generate a user message that would trigger that behavior:

```
Prompt: "For this instruction: 'Always use TypeScript instead of JavaScript'
Generate a realistic user request that would test whether the LLM follows
this instruction. The request should naturally lead to a situation where
the LLM could either follow or violate this instruction.

Output: {user_message: '...', expected_behavior: '...', check_method: 'programmatic|llm_judge'}"
```

#### Step 4: Run Evaluation

```python
for test in test_cases:
    response = llm.generate(
        system_prompt=compressed_claude_md,
        user_message=test.user_message
    )
    if test.check_method == "programmatic":
        passed = test.checker_function(response)
    else:
        passed = judge_llm.evaluate(
            instruction=test.instruction,
            expected=test.expected_behavior,
            actual=response
        )
```

#### Step 5: Compute DRFR

```
DRFR = instructions_followed / total_instructions
```

Compare DRFR(original_claude_md) vs DRFR(compressed_claude_md).

---

## Most Automated Pipeline Possible

### Option A: DeepEval PromptAlignmentMetric (Fastest to Set Up)

DeepEval has a built-in metric that does almost exactly what we need:

```python
from deepeval.metrics import PromptAlignmentMetric
from deepeval.test_case import LLMTestCase
from deepeval import evaluate

# Step 1: Extract instructions from CLAUDE.md (one-time, can use LLM)
instructions = [
    "Always use TypeScript instead of JavaScript",
    "Include absolute file paths in responses",
    "Never use emojis",
    "Suggest running tests before committing",
    # ... extracted from CLAUDE.md
]

# Step 2: Create metric
metric = PromptAlignmentMetric(
    prompt_instructions=instructions,
    model="gpt-4",  # judge model
    include_reason=True,
    threshold=0.8
)

# Step 3: Generate test cases (one per instruction or per scenario)
test_cases = [
    LLMTestCase(
        input="Help me create a new React component",
        actual_output=model_with_compressed_prompt("Help me create a new React component")
    ),
    # ... more test scenarios
]

# Step 4: Evaluate
evaluate(test_cases=test_cases, metrics=[metric])
```

**Score = (instructions followed) / (total instructions)**

**Pros:** 3 lines of config. Built-in reporting. Handles the judge calls automatically.
**Cons:** You still need to extract instructions and generate test scenarios manually (or with an LLM).

**GitHub:** https://github.com/confident-ai/deepeval
**Docs:** https://deepeval.com/docs/metrics-prompt-alignment

### Option B: Promptfoo (Most Configurable)

```yaml
# promptfooconfig.yaml
prompts:
  - role: system
    content: "file://compressed_claude.md"
  - role: user
    content: "{{user_message}}"

providers:
  - id: openai:gpt-4
    config:
      temperature: 0

tests:
  - vars:
      user_message: "Help me create a React component"
    assert:
      - type: llm-rubric
        value: "The response uses TypeScript, not JavaScript"
      - type: llm-rubric
        value: "The response includes absolute file paths"
      - type: not-contains
        value: "emoji_regex_pattern"
      - type: llm-rubric
        value: "The response suggests running tests"

  - vars:
      user_message: "Fix this bug in my code"
    assert:
      - type: llm-rubric
        value: "The response follows functional programming patterns"
```

**Pros:** YAML-based, CI/CD friendly, supports both programmatic and LLM-rubric assertions, great visualization.
**Cons:** Manual test case authoring (though `promptfoo generate assertions` can help).

**GitHub:** https://github.com/promptfoo/promptfoo

### Option C: Full Custom Pipeline (Most Automated)

Build a 3-stage pipeline where the user only provides the two CLAUDE.md files:

```
INPUT: original_claude.md + compressed_claude.md

Stage 1 - DECOMPOSE (one-time, cached):
  LLM extracts N atomic instructions from original_claude.md
  LLM classifies each as programmatic vs semantic
  LLM generates test scenarios for each instruction
  Output: test_suite.json

Stage 2 - EXECUTE (per compression):
  For each test scenario:
    Run model with original_claude.md -> baseline_response
    Run model with compressed_claude.md -> compressed_response
  Output: responses.json

Stage 3 - EVALUATE (per compression):
  For programmatic tests: run checker functions
  For semantic tests: LLM judge compares both responses
  Compute: per-instruction pass/fail + overall DRFR
  Output: evaluation_report.json with per-instruction breakdown
```

**Estimated cost per evaluation:**
- Stage 1: ~$1-3 (one-time, uses GPT-4 to decompose ~50 instructions, generate ~50 scenarios)
- Stage 2: ~$2-5 (100 LLM calls: 50 baseline + 50 compressed)
- Stage 3: ~$2-5 (50 judge calls with GPT-4)
- **Total: ~$5-13 per compression variant tested**

### Option D: YourBench (HuggingFace, Document-to-Evaluation)

YourBench is designed specifically for generating benchmarks from custom documents:

```
INPUT: CLAUDE.md document
OUTPUT: QA evaluation dataset with verifiable answers
```

**GitHub:** https://github.com/huggingface/yourbench
**Paper:** https://arxiv.org/abs/2504.01833

**Pros:** Designed for exactly this use case (document -> benchmark). Under $15 for generation. Preserves relative model rankings.
**Cons:** Oriented toward factual QA, not behavioral instruction following. Would need adaptation.

---

## Recommendation for the Prompt-Distill Project

**Primary approach: Option C (Full Custom Pipeline) using InFoBench/DRFR methodology.**

Rationale:
1. The user wants to focus purely on compression, not test creation -- the pipeline must auto-generate tests from the original CLAUDE.md
2. InFoBench's DRFR metric gives per-instruction granularity (you see exactly WHICH instructions were lost in compression)
3. The decomposition step needs to happen only once per CLAUDE.md; evaluation is fast and cheap thereafter
4. Programmatic checks (IFEval-style) can be used for the subset of instructions that are verifiable, saving judge costs
5. The A/B comparison (original vs compressed) inherently controls for task difficulty

**Secondary approach: DeepEval PromptAlignmentMetric for quick prototyping.**

It gives you a working evaluation in 10 minutes. Use it to validate the concept before building the full pipeline.

**Tertiary approach: Promptfoo for CI/CD integration.**

Once the test suite is generated, encode it as a promptfoo YAML config for repeatable, visual regression testing of each compression attempt.

### Automation Priority

The key insight: **the test generation is the expensive part, not the evaluation.** For a fully automated "user provides two files, gets a score" pipeline:

1. Use GPT-4 to decompose original CLAUDE.md into atomic instructions (cached, one-time)
2. Use GPT-4 to generate test scenarios for each instruction (cached, one-time)
3. Classify which tests are programmatic vs semantic (cached, one-time)
4. Run both versions through test scenarios (fast, ~$2-5)
5. Score with DRFR (fast, ~$2-5)

Total automation: user runs one command, gets back a report showing:
- Overall DRFR score (e.g., 0.92 = 92% of instructions preserved)
- Per-instruction pass/fail
- Which specific instructions were lost
- Cost: ~$5-13 per compression variant

---

## Related Tools and Datasets

| Tool/Dataset | URL | Relevance |
|-------------|-----|-----------|
| Argilla IFEval-like-data | https://huggingface.co/datasets/argilla/ifeval-like-data | Synthetic IFEval data generation pipeline |
| Distilabel | https://github.com/argilla-io/distilabel | Pipeline for synthetic data generation |
| IFEval-Extended | https://www.researchgate.net/publication/387435651 | Dynamic prompt generation for IFEval |
| AdvancedIF (Surge) | https://surgehq.ai/blog/advancedif-and-the-evolution-of-instruction-following-benchmarks | Evolution beyond basic IFEval constraints |
| TOWER metric | https://arxiv.org/html/2410.06089 | Tree-organized weighting for complex instructions |
| Multi-IF (Meta) | https://github.com/facebookresearch/Multi-IF | Multi-turn, multilingual IF eval |
