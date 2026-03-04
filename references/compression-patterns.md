# Compression Patterns Dictionary

> Tier별 before/after 변환 패턴. 스킬이 압축 시 참조하는 패턴 사전.

---

## T1: REMOVE Patterns

원문을 완전히 삭제. 출력에 흔적 없음.

| # | Before | Tokens | Action |
|---|---|---|---|
| 1 | "Always write clean, readable, and well-structured code" | 10 | Delete |
| 2 | "Make sure to handle errors properly and gracefully" | 10 | Delete |
| 3 | "Use proper indentation and formatting" | 7 | Delete |
| 4 | "Be helpful and provide accurate information" | 8 | Delete |
| 5 | "Follow best practices for the language you're using" | 10 | Delete |
| 6 | "Write unit tests for your code" | 7 | Delete |
| 7 | "Use meaningful variable and function names" | 8 | Delete |
| 8 | "Don't hardcode sensitive information like passwords or API keys" | 11 | Delete |
| 9 | "Always read the file before editing it" | 8 | Delete |
| 10 | "Consider edge cases in your implementation" | 7 | Delete |

---

## T2: ANCHOR Patterns

장황한 설명을 2-5 토큰 키워드로 압축.

| # | Before (verbose) | Tok | After (anchored) | Tok | Saved |
|---|---|---|---|---|---|
| 1 | "When writing TypeScript, always enable strict mode for better type safety and to catch potential errors at compile time" | 22 | `TypeScript: strict mode` | 4 | 18 |
| 2 | "Use ESLint for linting and Prettier for code formatting to ensure consistent style across the codebase" | 18 | `Lint: ESLint + Prettier` | 5 | 13 |
| 3 | "Follow the conventional commits specification for all commit messages, using prefixes like feat:, fix:, chore:, docs:" | 19 | `Commits: conventional commits` | 4 | 15 |
| 4 | "Use Jest as the testing framework for all unit and integration tests in this project" | 16 | `Test: Jest` | 3 | 13 |
| 5 | "Structure the project following the MVC (Model-View-Controller) architecture pattern for clean separation of concerns" | 18 | `Architecture: MVC` | 3 | 15 |
| 6 | "Use Docker for containerization and ensure all services can be run via docker-compose" | 15 | `Docker + docker-compose` | 4 | 11 |
| 7 | "Follow the Airbnb JavaScript Style Guide for all JavaScript and TypeScript code" | 14 | `Style: Airbnb` | 3 | 11 |
| 8 | "Use React Query (TanStack Query) for all server state management and data fetching" | 14 | `Data fetching: React Query` | 5 | 9 |
| 9 | "Implement authentication using JSON Web Tokens (JWT) with refresh token rotation" | 13 | `Auth: JWT + refresh rotation` | 5 | 8 |
| 10 | "Use Zod for runtime schema validation of all API inputs and outputs" | 12 | `Validation: Zod` | 3 | 9 |
| 11 | "Write all database queries using Prisma ORM instead of raw SQL" | 11 | `DB: Prisma ORM` | 4 | 7 |
| 12 | "Use GitHub Actions for CI/CD pipeline with automatic testing and deployment" | 12 | `CI/CD: GitHub Actions` | 4 | 8 |
| 13 | "Follow the BEM (Block Element Modifier) naming convention for all CSS class names" | 14 | `CSS: BEM naming` | 4 | 10 |
| 14 | "Use pnpm as the package manager for faster installations and disk space savings" | 14 | `Package manager: pnpm` | 4 | 10 |
| 15 | "Implement logging using Winston with structured JSON output for production" | 11 | `Logging: Winston (JSON)` | 5 | 6 |

---

## T3: CONDENSE Patterns

프로젝트 특화 규칙을 명령형 단문으로 재작성. 의미와 조건을 100% 보존.

| # | Before (verbose) | Tok | After (condensed) | Tok | Saved% |
|---|---|---|---|---|---|
| 1 | "For error responses, please use the RFC 7807 Problem Details format instead of our older error format which used {error, message} structure. This ensures consistency across all our microservices." | 38 | `Errors: RFC 7807 Problem Details (not legacy {error, message})` | 12 | 68% |
| 2 | "When deploying to the staging environment, always run the database migration check script first before proceeding with the deployment. However, for local development you can skip this step." | 35 | `Deploy staging → run migration check first. Skip for local.` | 11 | 69% |
| 3 | "We use a monorepo structure with all packages located in the /packages directory. Each package maintains its own tsconfig.json and package.json files, and they reference each other via workspace protocol." | 38 | `Monorepo: /packages/*, own tsconfig + package.json, workspace protocol` | 11 | 71% |
| 4 | "All API endpoints should be versioned using URL path versioning (e.g., /api/v1/users, /api/v2/users). When introducing breaking changes, create a new version rather than modifying the existing one." | 37 | `API: URL path versioning (/api/v{n}/). New version for breaking changes, keep old.` | 15 | 59% |
| 5 | "Environment variables should follow the naming convention MODULE_SETTING (e.g., DB_HOST, AUTH_SECRET_KEY). Never use .env files in production; use the vault service instead." | 32 | `Env vars: MODULE_SETTING format. Production: vault, not .env.` | 11 | 66% |
| 6 | "Feature branches should be named as feature/JIRA-123-short-description. Always rebase on main before creating a PR, and squash commits when merging." | 27 | `Branches: feature/JIRA-{id}-{desc}. Rebase on main before PR. Squash merge.` | 13 | 52% |
| 7 | "The frontend uses a custom design system located at /packages/ui. Always import components from @acme/ui rather than creating new ones. Check the Storybook at localhost:6006 for available components." | 36 | `UI components: import from @acme/ui (/packages/ui). Check Storybook :6006 before creating new.` | 16 | 56% |
| 8 | "Database column names use snake_case, but the API response fields should use camelCase. The ORM handles this transformation automatically, so never manually convert between the two in application code." | 36 | `DB: snake_case columns. API: camelCase responses. ORM auto-converts — don't manual transform.` | 14 | 61% |
| 9 | "All public-facing API responses must include pagination metadata following our standard format: { data: [...], meta: { page, perPage, total, totalPages } }. Default perPage is 20, maximum is 100." | 37 | `Pagination: { data, meta: {page, perPage, total, totalPages} }. Default 20, max 100.` | 16 | 57% |
| 10 | "When writing React components, always use functional components with hooks. Class components are not allowed. State management uses Zustand stores located in /stores directory." | 31 | `React: functional + hooks only. State: Zustand stores in /stores/.` | 12 | 61% |

---

## T4: PRESERVE Patterns

원문 그대로 유지. 아래는 보존이 필요한 유형 예시 (압축하지 않음).

### Internal API Schemas
```
POST /api/internal/incidents
Body: {
  severity: "P1" | "P2" | "P3" | "P4",
  team_id: string (from /api/teams),
  runbook_url: string (must link to Confluence),
  affected_services: string[] (service registry names),
  customer_impact: boolean
}
Response: { incident_id: string, slack_channel: string }
```

### Custom Business Logic
```
Commission calculation:
- Base rate: 12% of net revenue
- Tiers: >$50K/month → 15%, >$100K → 18%
- Clawback: If customer churns within 90 days, deduct full commission
- Split deals: Primary rep 60%, secondary 40%
- Quarterly bonus: If team hits 110% quota, additional 3% retroactive
```

### Non-obvious Workflows
```
Release process:
1. Create release branch from develop
2. Run /scripts/bump-version.sh (updates package.json + CHANGELOG)
3. Push and wait for CI (includes e2e against staging-mirror)
4. Get sign-off in #releases Slack channel (min 2 thumbs-up from senior eng)
5. Merge to main via fast-forward only (no squash!)
6. Tag triggers auto-deploy to production
7. Monitor Datadog dashboard "Release Health" for 30min
8. If error rate > 0.5%, run /scripts/rollback.sh immediately
```

---

## Anti-Patterns: Never Compress These

| Pattern | Why |
|---|---|
| Internal URLs/hostnames | Model will hallucinate different URLs |
| Specific port numbers (non-standard) | Model will guess wrong ports |
| Custom CLI tool flags | Model will invent flags |
| Authentication header names | Model will use generic names |
| Database table/column names | Model will use conventional names that don't match |
| Exact threshold values | Model will use "reasonable" defaults that differ |
| Multi-step workflows with ordering | Model will reorder or skip steps |

---

## Compression Rules

1. **Preserve all conditional logic**: "if/when/unless/except" → keep the condition, shorten the explanation
2. **Preserve all specific values**: port numbers, paths, limits, names
3. **Use imperative mood**: "Use X" not "You should use X" or "Please make sure to use X"
4. **Use symbols for brevity**: → (then/leads to), + (and), / (or), ! (not)
5. **Omit motivation**: "for better performance" or "to ensure consistency" can be dropped — the LLM doesn't need to know *why*
6. **Keep project-specific nouns**: @acme/ui, /packages/, JIRA-123 — these are the payload
7. **Drop universal qualifiers**: "always", "never", "make sure to", "please" — add no information
