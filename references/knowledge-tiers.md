# Knowledge Tier Classification Guide

> 각 지시를 4개 tier로 분류하는 기준과 결정 트리.

---

## Tier Definitions

### T1: REMOVE — Model Default Knowledge
LLM이 프롬프트 없이도 기본으로 수행하는 행동. 명시해도 이득 없음.

- **Action**: 완전 삭제 (0 tokens)
- **Risk**: Minimal
- **Catalog**: `model-defaults.md` 참조

**판별 기준**:
- 모든 현대 LLM이 기본으로 하는가? → T1
- 특정 도구/설정 없이 일반적 원칙만 언급하는가? → T1
- 제거해도 출력이 변하지 않을 것인가? → T1

### T2: ANCHOR — Brief Keyword Reminder
LLM이 알지만, 짧은 키워드로 올바른 행동을 확실히 활성화. 모델 버전 간 일관성 보장.

- **Action**: 2-5 토큰 키워드로 압축
- **Risk**: Low
- **Savings**: 항목당 80-90%

**판별 기준**:
- 잘 알려진 도구/프레임워크/패턴 이름인가? → T2
- 이름만 언급하면 LLM이 올바르게 사용법을 아는가? → T2
- 설명 없이 키워드만으로 의도가 전달되는가? → T2

### T3: CONDENSE — Keep but Make Concise
프로젝트 특화 규칙이나 기본값을 벗어나는 설정. 명시 필요하지만 장황할 필요 없음.

- **Action**: 명령형 단문으로 재작성 (30-60% 감소)
- **Risk**: Medium — 조건 로직과 구체적 값을 반드시 보존
- **Optimal length**: 8-20 단어

**판별 기준**:
- 프로젝트 고유 관례인가? → T3
- 기본값을 재정의하는가? → T3
- 조건 로직이 포함되어 있는가? → T3
- LLM이 추측할 수 없는 구체적 값이 있는가? → T3

### T4: PRESERVE — Keep As-Is or Expand
LLM이 절대 알 수 없는 정보. 압축하면 환각 위험.

- **Action**: 원문 유지 (필요시 확장)
- **Risk**: 압축 시 High — 모델이 누락 정보를 환각

**판별 기준**:
- 내부 API/엔드포인트인가? → T4
- 커스텀 비즈니스 로직인가? → T4
- 사내 도구/서비스인가? → T4
- 비표준 워크플로우인가? → T4
- 이 정보 없이 모델이 올바르게 동작할 수 있는가? 아니면 → T4

---

## Decision Tree

```
지시문 하나를 읽는다
│
├─ 모든 LLM이 기본으로 하는가?
│  ├─ YES → T1 REMOVE
│  └─ NO ↓
│
├─ 잘 알려진 도구/패턴/규약의 이름인가?
│  ├─ YES → 이름만으로 의도가 충분한가?
│  │  ├─ YES → T2 ANCHOR
│  │  └─ NO (커스텀 설정 포함) → T3 CONDENSE
│  └─ NO ↓
│
├─ 프로젝트 특화 규칙인가? (기본값 재정의, 조건 로직, 팀 관례)
│  ├─ YES → T3 CONDENSE
│  └─ NO ↓
│
└─ 내부 시스템/API/비즈니스 로직인가?
   ├─ YES → T4 PRESERVE
   └─ 판단 불가 → T3 CONDENSE (보수적 선택)
```

---

## Tier Boundary Examples

### T1 vs T2 경계

| Instruction | Tier | Reason |
|---|---|---|
| "Write clean code" | T1 | 범용 원칙, LLM 기본 |
| "Use ESLint" | T2 | 특정 도구 → 앵커 필요 |
| "Use ESLint with airbnb config" | T3 | 특정 설정 → 축약 필요 |

### T2 vs T3 경계

| Instruction | Tier | Reason |
|---|---|---|
| "Use TypeScript" | T2 | 잘 알려진 언어, 키워드 충분 |
| "TypeScript strict mode, no any" | T3 | 구체적 설정 2개 → 축약 |
| "TypeScript: strict, paths alias @/ → src/" | T3 | 프로젝트 설정 포함 |

### T3 vs T4 경계

| Instruction | Tier | Reason |
|---|---|---|
| "Errors: RFC 7807 format" | T3 | 공개 표준, 축약 가능 |
| "Errors: POST /api/v2/incidents with fields {severity, team_id, runbook_url}" | T4 | 내부 API 스키마 |
| "Deploy staging → run migration check" | T3 | 일반적 DevOps 패턴, 축약 |
| "Deploy: SSH to bastion-prod.internal, run /opt/deploy/release.sh --env=prod --skip-cdn" | T4 | 내부 인프라 상세 |

---

## Classification Heuristics

### Signals for T1 (Remove)
- "always", "properly", "correctly" + generic verb
- No specific tool, file, or value mentioned
- Would apply to ANY project in ANY language
- Appears in `model-defaults.md` catalog

### Signals for T2 (Anchor)
- Named tool: ESLint, Prettier, Jest, Docker, etc.
- Named pattern: conventional commits, BEM, REST, etc.
- Named framework: React, Next.js, FastAPI, etc.
- Simple mention without custom configuration

### Signals for T3 (Condense)
- Contains conditional logic: "if X then Y", "except when", "unless"
- References specific file paths or directory structure
- Overrides a language/framework default
- Team-specific convention different from community standard
- Contains specific numeric values (port numbers, limits, thresholds)

### Signals for T4 (Preserve)
- Internal URLs, hostnames, IP addresses
- Custom API schemas with field names
- Business-specific calculation formulas
- Multi-step workflows unique to the organization
- References to internal tools not publicly documented
- Authentication flows with specific token/header names

---

## Model Sensitivity Note

Tier 2 items are reliably activated on Claude Opus/Sonnet. For Haiku or smaller models, consider promoting T2 → T3 with slightly more context to ensure correct behavior.
