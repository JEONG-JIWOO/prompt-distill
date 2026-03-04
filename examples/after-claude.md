# CLAUDE.md — Acme Corp Backend

<!-- Optimized by prompt-distill: 63% token reduction -->
<!-- T1 removed: 18 items | T2 anchored: 11 items | T3 condensed: 12 items | T4 preserved: 3 items -->

## Stack
TypeScript strict, no `any` | Express | Prisma (`/packages/api/prisma/schema.prisma`) | Jest | ESLint + Prettier | Zod (all API I/O) | pnpm | Redis cache (TTL 300s, key: `{service}:{entity}:{id}`)

## Structure
Monorepo: `/packages/{api, shared, workers, db}`, own tsconfig + package.json, workspace protocol

## API
- Versioning: URL path `/api/v{n}/`. New version for breaking changes, keep old.
- Errors: RFC 7807 Problem Details (not legacy `{error, message}`)
- Pagination: `{ data, meta: {page, perPage, total, totalPages} }`. Default 20, max 100.
- Paths: kebab-case. JSON: camelCase.

## Database
- DB: snake_case columns → API: camelCase. Prisma auto-converts — don't manual transform.
- Soft delete (`deleted_at`) for user entities. Hard delete only for internal system records.
- Connection: `DATABASE_URL` env var

## Auth
JWT + refresh token rotation:
1. Access: 15min expiry. Refresh: 7d, single-use, both rotate.
2. Storage: HttpOnly cookies (not localStorage)
3. Secret: `AUTH_JWT_SECRET` env var
4. Payload: `{ sub: userId, role: UserRole, org: orgId }`

Roles: ADMIN (full) | MANAGER (team + reports) | MEMBER (own resources) | VIEWER (read-only)

Middleware: `authenticate → authorize(roles) → rateLimiter → handler`

## Env Vars
Format: `MODULE_SETTING`. Production: vault, not .env. Docs: `/packages/api/.env.example`

## Git
Conventional commits. Branches: `feature/JIRA-{id}-{desc}`. Rebase on main before PR. Squash merge.

## Testing
- Integration: `__tests__/integration/`
- Factories: `/packages/shared/test/factories/` (no raw DB inserts in tests)
- Coverage: ≥80% new code
- E2E: Playwright in `/packages/e2e/`

## Logging
Logger: `/packages/shared/src/logger.ts` (Winston JSON wrapper)
Levels: error (failure) | warn (degraded) | info (business) | debug (dev)
Include `x-request-id` correlation in all entries. Dashboards: Datadog "acme-backend"

## Deploy
Staging → run migration check first. Skip for local.

Production:
1. PR → main → CI (tests + lint) → auto-deploy staging
2. Smoke: `pnpm run test:smoke --env=staging`
3. Manual approval (GitHub Actions) → production
4. Monitor Datadog "Release Health" 30min
5. Error rate > 0.5% → `/scripts/rollback.sh`
