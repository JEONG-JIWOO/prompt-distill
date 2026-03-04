# LLM Model Defaults Catalog

> T1 제거 대상: LLM이 프롬프트 없이도 기본으로 수행하는 행동 목록.
> 근거: Yang et al. (2025) — 65.2%의 프롬프트 요구사항이 LLM 기본 행동과 일치.

---

## Code Quality & Style

These are universal programming practices all modern LLMs follow by default:

| Default Behavior | Example Prompt Text (removable) |
|---|---|
| Write readable code | "Write clean, readable code" |
| Use proper indentation | "Always indent properly" |
| Follow naming conventions | "Use meaningful variable names" |
| Add appropriate whitespace | "Use blank lines between functions" |
| Use consistent casing | "Use camelCase for variables" (when language default) |
| Avoid magic numbers | "Don't use magic numbers" |
| Keep functions focused | "Each function should do one thing" |
| DRY principle | "Don't repeat yourself" |
| Avoid deep nesting | "Keep nesting to 3 levels max" |

## Error Handling

| Default Behavior | Example Prompt Text (removable) |
|---|---|
| Handle errors | "Always handle errors properly" |
| Use try-catch | "Wrap risky code in try-catch" |
| Return meaningful errors | "Return helpful error messages" |
| Validate inputs | "Check inputs before processing" |
| Handle edge cases | "Consider edge cases" |
| Don't swallow exceptions | "Never silently catch errors" |

## Documentation & Comments

| Default Behavior | Example Prompt Text (removable) |
|---|---|
| Add comments for complex logic | "Comment your code" |
| Use JSDoc/docstrings | "Document public functions" |
| Write self-documenting code | "Code should be self-explanatory" |

## Security Basics

| Default Behavior | Example Prompt Text (removable) |
|---|---|
| Sanitize user input | "Always sanitize inputs" |
| Don't expose secrets | "Never hardcode passwords" |
| Use parameterized queries | "Prevent SQL injection" |
| Escape output | "Prevent XSS attacks" |
| Use HTTPS | "Always use secure connections" |

## Testing

| Default Behavior | Example Prompt Text (removable) |
|---|---|
| Write tests | "Write unit tests for your code" |
| Test happy path + edge cases | "Test both normal and edge cases" |
| Use descriptive test names | "Test names should describe behavior" |
| Arrange-Act-Assert pattern | "Structure tests with AAA pattern" |

## General Assistant Behavior

| Default Behavior | Example Prompt Text (removable) |
|---|---|
| Be helpful | "Be helpful and informative" |
| Be accurate | "Provide accurate information" |
| Explain reasoning | "Explain your thinking" |
| Ask for clarification | "Ask if something is unclear" |
| Follow instructions | "Follow the user's instructions" |
| Be concise | "Keep responses concise" |
| Use markdown formatting | "Format responses with markdown" |

## File Operations (Claude Code specific)

| Default Behavior | Example Prompt Text (removable) |
|---|---|
| Read before editing | "Always read the file before making changes" |
| Don't delete important code | "Be careful not to delete existing functionality" |
| Preserve existing style | "Match the existing code style" |
| Make minimal changes | "Only change what's necessary" |

## Language-Specific Defaults

### JavaScript/TypeScript
| Default | Removable Prompt |
|---|---|
| Use `const`/`let` over `var` | "Never use var" |
| Use arrow functions where appropriate | "Prefer arrow functions" |
| Use template literals | "Use template literals instead of concatenation" |
| Use async/await over callbacks | "Use async/await" |
| Use destructuring | "Use destructuring assignment" |

### Python
| Default | Removable Prompt |
|---|---|
| Follow PEP 8 | "Follow PEP 8 style guide" |
| Use f-strings | "Use f-strings for formatting" |
| Use list comprehensions | "Use list comprehensions when appropriate" |
| Use type hints | "Add type hints" |
| Use context managers | "Use 'with' for file operations" |

### General
| Default | Removable Prompt |
|---|---|
| Use modern language features | "Use latest language features" |
| Follow community conventions | "Follow standard conventions" |
| Import only what's needed | "Don't import unnecessary modules" |

---

## Boundary Cases: When Defaults Become Non-Default

The items above are T1 **only when stated generically**. They become T2+ when:

| Condition | Tier Escalation | Example |
|---|---|---|
| Specifies a particular tool/config | → T2 (Anchor) | "Use ESLint with airbnb config" |
| Overrides the default convention | → T3 (Condense) | "Use tabs instead of spaces (project standard)" |
| Has conditional logic | → T3 (Condense) | "Use snake_case in Python, camelCase in JS" |
| References internal systems | → T4 (Preserve) | "Use our custom error handler at /lib/errors.ts" |

---

## Source

- Yang et al. (2025) "What Prompts Don't Say" §3.2 — 65.2% of requirements guessed correctly by LLMs
- Format-related defaults: 70.7% reliable (§3.2)
- Conditional requirements: only 22.9% guessable (§3.2) — NOT defaults
- Anthropic: "Only add context Claude doesn't already have"
