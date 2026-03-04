"""
Compression quality tests using DeepEval GEval with A/B comparison.

For each topic, generates responses using BOTH the original and compressed
CLAUDE.md as system prompts, then scores each with GEval. The compressed
version must score >= 90% of the original to pass.

Usage:
    pip install -r tests/requirements.txt
    OPENAI_API_KEY=sk-... deepeval test run tests/test_compression_quality.py
"""

import pytest
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from conftest import (
    BEFORE_FILE, AFTER_FILE, JUDGE_MODEL, QUALITY_RATIO,
    generate_response,
)


# ---------------------------------------------------------------------------
# Topic-based GEval metrics with evaluation steps
# ---------------------------------------------------------------------------

TOPIC_TESTS = [
    {
        "name": "api_conventions",
        "scenario": (
            "Create a new REST API endpoint POST /api/v1/users for user "
            "registration. It should accept email and password in the request "
            "body, validate the input, hash the password, save to the database, "
            "and return the created user. Include error handling for duplicate "
            "emails."
        ),
        "criteria": (
            "Does the response follow the project's API conventions for "
            "endpoint structure, error handling, validation, and naming?"
        ),
        "evaluation_steps": [
            "Check if the endpoint uses versioned URL path format (/api/v1/...)",
            "Check if input validation uses Zod schemas",
            "Check if error responses follow RFC 7807 Problem Details format or similar structured error format, not plain {error, message}",
            "Check if URL paths use kebab-case and JSON response fields use camelCase",
            "Check if the code is written in TypeScript (not plain JavaScript)",
            "Check if database operations use Prisma ORM (not raw SQL)",
        ],
    },
    {
        "name": "auth_system",
        "scenario": (
            "Implement a POST /api/v1/auth/login endpoint that authenticates "
            "a user with email and password. It should verify credentials, "
            "generate tokens, and set up the session. Also show how to protect "
            "a route so only MANAGER and ADMIN roles can access it."
        ),
        "criteria": (
            "Does the response correctly implement the project's authentication "
            "and authorization system?"
        ),
        "evaluation_steps": [
            "Check if JWT tokens are used for authentication",
            "Check if the token payload includes user ID, role, and org fields",
            "Check if tokens are set in HttpOnly cookies (not localStorage)",
            "Check if role-based access control is implemented with the correct roles (ADMIN, MANAGER, MEMBER, VIEWER)",
            "Check if the middleware chain follows authenticate → authorize → handler pattern",
        ],
    },
    {
        "name": "database_patterns",
        "scenario": (
            "Implement a soft-delete feature for the User model. Create a "
            "DELETE /api/v1/users/:id endpoint that soft-deletes the user, "
            "and update the GET /api/v1/users endpoint to exclude soft-deleted "
            "users by default. Show the Prisma query patterns."
        ),
        "criteria": (
            "Does the response follow the project's database conventions for "
            "Prisma usage, naming, and soft deletes?"
        ),
        "evaluation_steps": [
            "Check if Prisma ORM is used for all database operations",
            "Check if soft delete uses a deleted_at timestamp field",
            "Check if database columns use snake_case naming",
            "Check if API response fields use camelCase naming",
            "Check if the code avoids manual case conversion (relies on ORM auto-conversion)",
        ],
    },
    {
        "name": "project_structure",
        "scenario": (
            "Set up a new package called 'notifications' in the monorepo. "
            "Show the directory structure, package.json, tsconfig.json, and "
            "how it references the shared package. Include the commands to "
            "install dependencies."
        ),
        "criteria": (
            "Does the response follow the project's monorepo structure and "
            "package management conventions?"
        ),
        "evaluation_steps": [
            "Check if the new package is placed under /packages/ directory",
            "Check if it has its own tsconfig.json and package.json",
            "Check if workspace protocol is used for internal package references",
            "Check if pnpm is used as the package manager (not npm or yarn)",
        ],
    },
    {
        "name": "testing_practices",
        "scenario": (
            "Write integration tests for a payment webhook handler at "
            "POST /api/v1/webhooks/payments. The handler should verify the "
            "webhook signature, process the payment event, and update the "
            "order status. Show test setup, test data creation, and assertions."
        ),
        "criteria": (
            "Does the response follow the project's testing conventions for "
            "framework, test data, and structure?"
        ),
        "evaluation_steps": [
            "Check if Jest is used as the testing framework",
            "Check if test data is created using factories (not raw database inserts)",
            "Check if the test file is placed in an appropriate test directory",
            "Check if the tests cover both success and error scenarios",
        ],
    },
    {
        "name": "infra_conventions",
        "scenario": (
            "Add Redis caching to the GET /api/v1/products/:id endpoint. "
            "Show the cache lookup, cache miss handling, cache invalidation "
            "on product update, and structured logging for cache hits/misses."
        ),
        "criteria": (
            "Does the response follow the project's infrastructure conventions "
            "for caching, logging, and configuration?"
        ),
        "evaluation_steps": [
            "Check if Redis is used for caching",
            "Check if cache keys follow the {service}:{entity}:{id} format",
            "Check if the project's custom logger is used (Winston-based structured JSON)",
            "Check if log entries include appropriate log levels (info, debug, error)",
        ],
    },
]


# ---------------------------------------------------------------------------
# Tests — A/B comparison per topic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "topic",
    TOPIC_TESTS,
    ids=[t["name"] for t in TOPIC_TESTS],
)
def test_compression_preserves_quality(topic: dict):
    """
    For each topic:
    1. Generate response with original CLAUDE.md (baseline)
    2. Generate response with compressed CLAUDE.md
    3. Score both with GEval
    4. Assert compressed >= original * QUALITY_RATIO
    """
    scenario = topic["scenario"]

    # Generate A/B responses
    response_original = generate_response(scenario, BEFORE_FILE)
    response_compressed = generate_response(scenario, AFTER_FILE)

    # Create GEval metric
    metric = GEval(
        name=topic["name"],
        criteria=topic["criteria"],
        evaluation_steps=topic["evaluation_steps"],
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        model=JUDGE_MODEL,
        threshold=0.5,
    )

    # Score original
    case_original = LLMTestCase(
        input=scenario,
        actual_output=response_original,
    )
    metric.measure(case_original)
    score_original = metric.score
    reason_original = metric.reason

    # Score compressed
    case_compressed = LLMTestCase(
        input=scenario,
        actual_output=response_compressed,
    )
    metric.measure(case_compressed)
    score_compressed = metric.score
    reason_compressed = metric.reason

    # Report
    print(f"\n{'='*60}")
    print(f"Topic: {topic['name']}")
    print(f"Original score:   {score_original:.2f}  {reason_original}")
    print(f"Compressed score: {score_compressed:.2f}  {reason_compressed}")
    print(f"Ratio: {score_compressed/max(score_original, 0.01):.2f} (need >= {QUALITY_RATIO})")
    print(f"{'='*60}")

    # Assert: compressed must be within QUALITY_RATIO of original
    min_acceptable = score_original * QUALITY_RATIO
    assert score_compressed >= min_acceptable, (
        f"[{topic['name']}] Compression degraded quality: "
        f"compressed={score_compressed:.2f} < "
        f"original={score_original:.2f} × {QUALITY_RATIO} = {min_acceptable:.2f}"
    )
