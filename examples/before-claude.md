# CLAUDE.md — Acme Corp Backend

## General Principles

- Always write clean, readable, and well-structured code
- Follow best practices for the language you're using
- Make sure to handle errors properly and gracefully
- Use meaningful variable and function names
- Consider edge cases in your implementation
- Be helpful and provide accurate information when answering questions
- Read the file before editing it to understand the existing code

## Tech Stack

- We use TypeScript for all backend code. Always enable strict mode in TypeScript for better type safety and to catch potential errors at compile time. Never use the `any` type.
- Use Node.js with Express.js for the API server
- Use Prisma ORM for all database operations instead of writing raw SQL queries. The Prisma schema is located at `/packages/api/prisma/schema.prisma`.
- Use Jest as the testing framework for all unit and integration tests in this project
- Use ESLint for linting and Prettier for code formatting to ensure consistent style across the codebase
- Use Zod for runtime schema validation of all API inputs and outputs
- Use pnpm as the package manager for faster installations and disk space savings
- Use Redis for caching with a default TTL of 300 seconds. Cache keys must follow the format `{service}:{entity}:{id}`.

## Project Structure

We use a monorepo structure with all packages located in the /packages directory. Each package maintains its own tsconfig.json and package.json files, and they reference each other via workspace protocol. The main packages are:

- `/packages/api` — Express API server
- `/packages/shared` — Shared types and utilities
- `/packages/workers` — Background job processors
- `/packages/db` — Prisma client and migrations

## API Conventions

- All API endpoints should be versioned using URL path versioning (e.g., /api/v1/users, /api/v2/users). When introducing breaking changes, create a new version rather than modifying the existing one.
- For error responses, please use the RFC 7807 Problem Details format instead of our older error format which used {error, message} structure. This ensures consistency across all our microservices.
- All public-facing API responses must include pagination metadata following our standard format: `{ data: [...], meta: { page, perPage, total, totalPages } }`. Default perPage is 20, maximum is 100.
- Always use proper HTTP status codes. Don't just return 200 for everything.
- Use kebab-case for URL paths and camelCase for JSON fields

## Database

- Database column names use snake_case, but the API response fields should use camelCase. The ORM handles this transformation automatically, so never manually convert between the two in application code.
- Always create database migrations for schema changes. Never modify the database manually.
- Use soft deletes (deleted_at timestamp) for user-facing entities. Hard delete is only allowed for internal system records.
- Connection string is stored in `DATABASE_URL` environment variable

## Authentication & Authorization

Our authentication system uses JWT tokens with refresh token rotation:

1. Access tokens expire after 15 minutes
2. Refresh tokens expire after 7 days and are single-use
3. On refresh, both tokens are rotated
4. Tokens are stored in HttpOnly cookies, not localStorage
5. The JWT secret is in the `AUTH_JWT_SECRET` environment variable
6. Token payload: `{ sub: userId, role: UserRole, org: orgId }`

Role-based access control:
- `ADMIN` — Full access to all resources
- `MANAGER` — Can manage team members and view reports
- `MEMBER` — Can only access own resources
- `VIEWER` — Read-only access, no mutations

Middleware chain: `authenticate → authorize(roles) → rateLimiter → handler`

## Environment Variables

Environment variables should follow the naming convention MODULE_SETTING (e.g., DB_HOST, AUTH_SECRET_KEY). Never use .env files in production; use the vault service instead. Required variables for each environment are documented in `/packages/api/.env.example`.

## Git Workflow

- Follow the conventional commits specification for all commit messages, using prefixes like feat:, fix:, chore:, docs:
- Feature branches should be named as feature/JIRA-123-short-description. Always rebase on main before creating a PR, and squash commits when merging.
- Don't push directly to main. Always create a pull request.
- Write descriptive PR descriptions

## Testing

- Write unit tests for your code
- Test both normal and edge cases
- Integration tests go in `__tests__/integration/` directory
- Use factories from `/packages/shared/test/factories/` to create test data. Never use raw database inserts in tests.
- Minimum test coverage: 80% for new code
- E2E tests use Playwright and are in `/packages/e2e/`

## Logging & Monitoring

- Use our custom logger at `/packages/shared/src/logger.ts` (wraps Winston with structured JSON output)
- Log levels: `error` for failures, `warn` for degraded state, `info` for business events, `debug` for development
- All API requests are automatically logged by the request logger middleware
- Include correlation ID (`x-request-id` header) in all log entries
- Monitoring dashboards are in Datadog under the "acme-backend" service

## Deployment

When deploying to the staging environment, always run the database migration check script first before proceeding with the deployment. However, for local development you can skip this step.

Production deployment process:
1. Merge PR to main
2. CI runs all tests + lint
3. If CI passes, auto-deploys to staging
4. Run smoke tests: `pnpm run test:smoke --env=staging`
5. Manual approval in GitHub Actions
6. Deploy to production
7. Monitor Datadog "Release Health" dashboard for 30 minutes
8. If error rate exceeds 0.5%, run `/scripts/rollback.sh`

## Code Review

- Always review for security vulnerabilities
- Check for proper error handling
- Ensure tests are included
- Verify API backwards compatibility
- Make sure documentation is updated
