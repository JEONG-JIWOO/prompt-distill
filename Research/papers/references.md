# Research References

prompt-distill의 디자인 근거가 되는 논문, 문서, 도구 모음.

---

## 1. What Prompts Don't Say (Yang et al., 2025)

> 프롬프트 내 지시사항의 65.2%가 LLM이 이미 기본으로 아는 내용이라는 핵심 근거.

**논문 정보**
- 제목: "What Prompts Don't Say: Understanding and Managing Underspecification in LLM Prompts"
- 저자: Chenyang Yang, Yike Shi, Qianou Ma, Michael Xieyang Liu, Christian Kastner, Tongshuang Wu (CMU)
- 상태: arXiv preprint (2505.13360), 25 pages

**링크**
| 자료 | URL |
|---|---|
| arXiv 메인 | https://arxiv.org/abs/2505.13360 |
| HTML (섹션 이동 가능) | https://arxiv.org/html/2505.13360v2 |
| PDF | https://arxiv.org/pdf/2505.13360 |
| GitHub (코드+데이터) | https://github.com/malusamayo/underspec-analysis |
| Figshare (재현 데이터) | https://figshare.com/s/38acdc02f9cae8c39198 |
| Semantic Scholar | https://www.semanticscholar.org/paper/32e5fec8ea3e40dfe66c74e3eaacd2e0615fb493 |
| 제1저자 홈페이지 | https://www.cs.cmu.edu/~cyang3/ |

**핵심 수치 직접 링크** (HTML 버전 섹션 앵커)

| 인용 수치 | 위치 | 직접 링크 |
|---|---|---|
| "65.2% requirements are guessed by LLMs when unspecified" | §3.2 | https://arxiv.org/html/2505.13360v2#S3.SS2 |
| "format-related requirements 70.7% reliable" | §3.2 | https://arxiv.org/html/2505.13360v2#S3.SS2 |
| "conditional requirements only 22.9% guessable" | §3.2 | https://arxiv.org/html/2505.13360v2#S3.SS2 |
| "2x regression risk on unspecified requirements" | §3.3 | https://arxiv.org/html/2505.13360v2#S3.SS3 |
| "specifying everything actually hurts performance" | §3.4 | https://arxiv.org/html/2505.13360v2#S3.SS4 |
| "requirements-aware optimization: 4.8% improvement, 43% length reduction" | §4.2 | https://arxiv.org/html/2505.13360v2#S4.SS2 |

**방법론 요약**
- §3.1: 실험 세팅 — 3개 태스크 도메인, validator 기반 요구사항 큐레이션
- §3.2: LLM 기본 추론 능력 (65.2%, 70.7%, 22.9%)
- §3.3: 모델 업데이트 시 regression 안정성 (2x)
- §3.4: 모든 요구사항을 명시하면 오히려 성능 저하 (instruction-following 한계)
- §4.2: requirements-aware 선택적 명시 → 최적 전략 제안

**prompt-distill 적용**: Tier 1-4 분류 체계의 이론적 근거. 특히 §3.4의 "전부 명시하면 역효과" 발견이 Tier 1 제거의 정당성.

---

## 2. LLMLingua Series (Microsoft, 2023-2024)

> 프롬프트 압축의 기술적 상한선 참조. 20x 압축 시 1.52pp 성능 저하.

### LLMLingua (원본)

- 제목: "LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models"
- 학회: EMNLP 2023
- 저자: Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, Lili Qiu (Microsoft Research)

| 자료 | URL |
|---|---|
| arXiv | https://arxiv.org/abs/2310.05736 |
| OpenReview (EMNLP) | https://openreview.net/forum?id=ADsEdyI32n |
| GitHub | https://github.com/microsoft/LLMLingua |
| 프로젝트 사이트 | https://llmlingua.com/ |
| HuggingFace 데모 | https://huggingface.co/spaces/microsoft/LLMLingua |
| MS Research 블로그 | https://www.microsoft.com/en-us/research/blog/llmlingua-innovating-llm-efficiency-with-prompt-compression/ |

**"20x 압축" 수치 검증**: GSM8K 기준 — Full-shot 78.85 EM → 20x 압축 시 77.33 EM (1.52pp 하락). 논문 본문은 "little performance loss"로 표현. "<1.5%"는 마케팅 근사치.

### LongLLMLingua

| 자료 | URL |
|---|---|
| arXiv | https://arxiv.org/abs/2310.06839 |
| MS Research | https://www.microsoft.com/en-us/research/project/llmlingua/longllmlingua/ |

- 학회: ACL 2024
- 핵심: 장문맥 시나리오 특화, question-aware 압축, 1/4 토큰으로 21.4% 성능 향상

### LLMLingua-2

| 자료 | URL |
|---|---|
| arXiv | https://arxiv.org/abs/2403.12968 |
| HuggingFace 데모 | https://huggingface.co/spaces/microsoft/llmlingua-2 |
| MS Research | https://www.microsoft.com/en-us/research/project/llmlingua/llmlingua-2/ |

- 학회: ACL 2024 Findings
- 핵심: GPT-4 data distillation → BERT 인코더로 토큰 분류. LLMLingua 대비 3-6x 빠름

### 프레임워크 통합

| 통합 대상 | 문서 |
|---|---|
| PromptFlow | https://microsoft.github.io/promptflow/integrations/tools/llmlingua-prompt-compression-tool.html |
| AutoGen | https://microsoft.github.io/autogen/0.2/docs/topics/handling_long_contexts/compressing_text_w_llmligua/ |
| LlamaIndex | https://github.com/microsoft/LLMLingua/blob/main/examples/RAGLlamaIndex.ipynb |

**prompt-distill 적용**: LLMLingua는 토큰 단위 binary keep/remove. prompt-distill의 Tier 2 (anchor) 접근법은 이와 다른 차원의 압축 — semantic-level 압축.

---

## 3. Anthropic 공식 문서

> Tier 분류 철학과 skill 작성 표준의 근거.

### 핵심 문서

| 문서 | URL | 핵심 내용 |
|---|---|---|
| **Skill 작성 Best Practices** | https://docs.anthropic.com/en/docs/agents-and-tools/agent-skills/best-practices | "Only add context Claude doesn't already have", 500줄 권장, 모델별 테스트 |
| **Claude 4 프롬프팅 가이드** | https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/claude-4-best-practices | 포매팅 영향, Opus overtrigger, MUST/ALWAYS 완화 |
| **Skills 문서 (Claude Code)** | https://docs.anthropic.com/en/docs/claude-code/skills | SKILL.md 구조, frontmatter, 500줄 팁 |
| **Agent Skills 개요** | https://docs.anthropic.com/en/docs/agents-and-tools/agent-skills/overview | progressive disclosure, 3단계 콘텐츠 로딩 |
| **프롬프트 엔지니어링 개요** | https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview | 전체 프롬프트 엔지니어링 랜딩 페이지 |
| **프롬프트 캐싱** | https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching | prefix matching, cache_control, 75% 비용 절감 |
| **Context Engineering 블로그** | https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents | 에이전트용 컨텍스트 큐레이션 전략 |

### 디자인 문서에서 인용한 내용 — 직접 출처

| 인용 | 출처 문서 | 해당 섹션 |
|---|---|---|
| "Only add context Claude doesn't already have" | Skill Best Practices | Core principles > Concise is key |
| "SKILL.md under 500 lines" | Skill Best Practices + Skills 문서 | Token budgets / 체크리스트 |
| "What works perfectly for Opus might need more detail for Haiku" | Skill Best Practices | Test with all models you plan to use |
| "prompt formatting style influences output style" | Claude 4 가이드 | Output and formatting > Control the format |
| "Opus overtriggers on aggressive phrasing" | Claude 4 가이드 | Tool use > Tool usage |
| "MUST/ALWAYS/CRITICAL 완화" | Claude 4 가이드 | Tool use + Migration considerations |

### 추가 리소스

| 자료 | URL |
|---|---|
| Skills 가이드 PDF | https://resources.anthropic.com/hubfs/The-Complete-Guide-to-Building-Skill-for-Claude.pdf |
| Agent Skills 블로그 | https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills |
| Agent Skills 발표 | https://www.anthropic.com/news/skills |
| Claude Code 문서 맵 | https://code.claude.com/docs/en/claude_code_docs_map.md |
| Claude Code llms.txt | https://code.claude.com/docs/llms.txt |
| DeepLearning.AI 코스 | https://www.deeplearning.ai/short-courses/agent-skills-with-anthropic/ |
| Anthropic Skilljar 코스 | https://anthropic.skilljar.com/introduction-to-agent-skills |

**prompt-distill 적용**: "Only add context Claude doesn't already have"가 Tier 1 제거의 공식적 근거. Opus overtrigger 관련 내용은 압축 시 강한 어조(MUST, CRITICAL) 완화도 최적화 대상임을 시사.

---

## 4. MLMastery — Instruction Referencing (2025)

> Cross-file 중복 제거 전략의 실용적 참조.

| 자료 | URL |
|---|---|
| 원문 | https://machinelearningmastery.com/prompt-compression-for-llm-generation-optimization-and-cost-reduction/ |

**5가지 압축 기법** (기사 내용):
1. Semantic Summarization — 장문을 JSON/불릿으로 압축
2. Structured Prompting — 명시적 구조로 모호성 제거
3. Relevance Filtering — 관련 섹션만 선택적 포함
4. **Instruction Referencing** — 공통 지시를 한 번 등록, 이름으로 재사용 ("Use Style Guide X")
5. Template Abstraction — 출력 포맷을 템플릿으로 등록

**prompt-distill 적용**: Future extension인 "Cross-file dedup"의 구현 전략. CLAUDE.md와 여러 SKILL.md에 중복되는 규칙을 `references/shared-rules.md`에 한 번 등록하고 참조.

---

## 5. 추가 관련 연구

### 서베이 논문

| 논문 | URL | 핵심 |
|---|---|---|
| Prompt Compression Survey (NAACL 2025 Oral) | https://arxiv.org/abs/2410.12388 | hard/soft 압축 분류 체계 |
| ↳ GitHub | https://github.com/ZongqianLi/Prompt-Compression-Survey | |
| Efficient Prompting Methods Survey | https://arxiv.org/html/2404.01077v2 | ITPC, perplexity 기반 pruning |
| Automatic Prompt Optimization Survey (EMNLP 2025) | https://aclanthology.org/2025.emnlp-main.1681.pdf | 자동 프롬프트 최적화 전반 |

### 주목할 압축 기법

| 기법 | URL | 핵심 |
|---|---|---|
| CompactPrompt | https://arxiv.org/abs/2510.18043 | self-information 점수 + n-gram 약어. 60% 축소, <5% 정확도 하락 |
| ProCut | https://arxiv.org/html/2508.02053v2 | attribution 기반 토큰 중요도 판별 |
| ICAE (ICLR 2024) | https://openreview.net/forum?id=uREj4ZuGJE | 컨텍스트를 compact representation으로 인코딩 |

### 프롬프트 최적화 프레임워크

| 프레임워크 | URL | 핵심 |
|---|---|---|
| DSPy (Stanford) | https://dspy.ai/ / https://github.com/stanfordnlp/dspy | 프롬프트 대신 모듈 시그니처, 자동 최적화 |
| TextGrad (Nature 게재) | https://github.com/zou-group/textgrad | 텍스트 피드백으로 프롬프트 자동 미분/최적화 |

### 실용 가이드

| 자료 | URL |
|---|---|
| Prompting Guide — 최적화 | https://www.promptingguide.ai/guides/optimizing-prompts |
| 4 Techniques (TDS) | https://towardsdatascience.com/4-techniques-to-optimize-your-llm-prompts-for-cost-latency-and-performance/ |
| Token-Budget-Aware Reasoning (ACL 2025) | https://aclanthology.org/2025.findings-acl.1274.pdf |

---

## 6. Information Theory & Token Importance Measurement

> Tier 분류의 이론적 기반 — self-information, surprisal, entropy를 사용한 토큰/지시문 중요도 측정.

### SelectiveContext (Li, 2023)

> Self-information 기반 프롬프트 pruning의 원형. LLMLingua의 이론적 선행 연구.

- 제목: "Compressing Context to Enhance Inference Efficiency of Large Language Models"
- 핵심: 소형 LM(GPT-2 등)으로 각 lexical unit(문장/구/토큰)의 self-information을 계산, 낮은 정보량 토큰 제거
- 결과: 36% GPU 메모리 절감, 32% 추론 지연 개선, 최대 32x 압축에서 BERTScore/BLEU 손실 무시할 수준

| 자료 | URL |
|---|---|
| GitHub | https://github.com/liyucheng09/Selective_Context |
| arXiv (관련) | https://arxiv.org/abs/2304.12102 |

**prompt-distill 적용**: Tier 1/2 분류 시 self-information 점수를 참조 지표로 활용 가능. 낮은 self-information = 모델이 이미 예측 가능한 내용 = Tier 1 후보.

### Understanding Chain-of-Thought via Information Theory (ICLR 2025)

> CoT 추론의 각 단계별 "information gain"을 정량화하는 프레임워크.

- 제목: "Understanding Chain-of-Thought in LLMs through Information Theory"

| 자료 | URL |
|---|---|
| OpenReview | https://openreview.net/forum?id=IjOWms0hrf |
| PromptLayer 해설 | https://www.promptlayer.com/research-papers/unlocking-the-secrets-of-llm-reasoning |

**prompt-distill 적용**: 지시문의 각 부분이 최종 출력에 기여하는 information gain을 측정하는 프레임워크로 활용 가능. Information gain이 낮은 지시 = Tier 1/2 후보.

### Semantic Chunking and the Entropy of Natural Language (Zhong et al., 2025)

> 자연어의 80% 중복성을 self-similar semantic chunking으로 설명하는 정보이론 모델.

- 제목: "Semantic Chunking and the Entropy of Natural Language"
- 핵심: 영어의 entropy rate는 문자당 약 1비트 — 랜덤 텍스트 대비 약 80% 중복. 텍스트를 self-similarly하게 의미 단위로 분할하면 이 중복 구조를 계층적으로 분해 가능.
- 방법: 현대 LLM과 공개 데이터셋 실험으로 semantic hierarchy의 여러 수준에서 실제 텍스트 구조를 정량적으로 포착.

| 자료 | URL |
|---|---|
| arXiv | https://arxiv.org/abs/2602.13194 |
| HTML | https://arxiv.org/html/2602.13194v2 |

**prompt-distill 적용**: 지시문의 의미 단위 분할(semantic chunking)에 entropy 기반 접근법의 이론적 근거. "80% 중복성" 수치는 Yang et al.의 "65.2% 기본값" 발견과 상호 보완.

### Surprisal Theory in Computational Linguistics

> 자연어 처리 난이도의 정보이론적 기반 — 인간과 LLM 모두에 적용.

- 핵심 개념: Surprisal(-log P(word|context))은 단어의 예측 가능성 척도. 높은 surprisal = 새로운 정보 = 보존 필요. 낮은 surprisal = 예측 가능 = 제거/압축 가능.
- Lossy-Context Surprisal: 불완전한 메모리 표현 하에서의 surprisal 모델 — 프롬프트 압축 시 정보 손실 분석에 활용 가능.

| 자료 | URL |
|---|---|
| Lossy-Context Surprisal | https://pmc.ncbi.nlm.nih.gov/articles/PMC7065005/ |
| Information Theory as Bridge (Frontiers) | https://www.frontiersin.org/journals/communication/articles/10.3389/fcomm.2022.657725/full |
| LLM Surprisal & Informativity Bias | https://www.researchgate.net/publication/372640935 |

**prompt-distill 적용**: Tier 분류의 핵심 이론 기반. 각 지시문의 surprisal을 LLM 기준으로 측정하면 Tier 1(낮은 surprisal) vs Tier 4(높은 surprisal) 구분의 정보이론적 근거 확보.

---

## 7. Bloom's Taxonomy Applied to LLM Prompting

> 인지 수준 기반 분류 체계를 프롬프트 지시문 분류에 적용한 연구.

### BloomWise (2024)

> Bloom's Taxonomy의 인지 수준에서 영감을 받은 프롬프트 기반 문제 해결 프레임워크.

- 제목: "BloomWise: Enhancing Problem-Solving capabilities of Large Language Models using Bloom's-Taxonomy-Inspired Prompts"
- 핵심: 기본 recall부터 고급 분석/창의적 종합까지 인지 수준에 맞는 프롬프트 설계로 적절한 깊이의 응답 보장.

| 자료 | URL |
|---|---|
| arXiv | https://arxiv.org/html/2410.04094v2 |

### Mechanistic Interpretability of Cognitive Complexity in LLMs (2025)

> LLM 내부 표현에서 Bloom's Taxonomy 인지 수준이 선형적으로 인코딩되어 있음을 증명.

- 제목: "Mechanistic Interpretability of Cognitive Complexity in LLMs via Linear Probing using Bloom's Taxonomy"
- 핵심: Linear classifier가 모든 Bloom 수준에서 ~95% 평균 정확도 달성 — 인지 수준이 모델 표현의 선형 부분공간에 인코딩되어 있다는 강력한 증거.

| 자료 | URL |
|---|---|
| arXiv | https://arxiv.org/html/2602.17229 |

### Higher Order Prompting (Jackson, 2024)

> Bloom's Revised Taxonomy를 LLM 활용에 직접 적용한 고등교육 연구.

| 자료 | URL |
|---|---|
| 원문 | https://stel.pubpub.org/pub/04-01-jackson |

**prompt-distill 적용**: Bloom's Taxonomy의 6단계 인지 수준(기억→이해→적용→분석→평가→창조)은 Tier 분류와 매핑 가능:
- 기억/이해 수준 지시 → Tier 1 (모델이 이미 아는 기본 지식)
- 적용/분석 수준 → Tier 2-3 (도메인 적용, 앵커로 활성화 가능)
- 평가/창조 수준 → Tier 4 (고유 비즈니스 로직, 전체 보존 필요)

---

## 8. Prompt Anchoring & Latent Knowledge Activation

> Tier 2의 핵심 메커니즘 — 키워드로 모델의 잠재 지식을 활성화하는 연구.

### Selective Prompt Anchoring (SPA) (Tian & Zhang, 2024)

> 프롬프트 내 특정 토큰의 attention을 증폭하여 모델 성능을 향상시키는 모델-비의존적 기법.

- 제목: "Selective Prompt Anchoring for Code Generation"
- 학회: ICML 2025 Accepted
- 핵심: LLM은 코드 생성이 진행될수록 사용자 프롬프트에 대한 attention이 희석됨. SPA는 "anchored text"(선택된 프롬프트 토큰)의 attention을 증폭하여 이를 보정.
- 방법: 원본 임베딩과 mask 임베딩의 logit 분포 차이를 "anchoring strength" 하이퍼파라미터로 증폭.
- 결과: Pass@1 최대 12.9% 향상, 6개 LLM x 6개 벤치마크에서 SOTA 코드 생성 방법 일관 능가.

| 자료 | URL |
|---|---|
| arXiv | https://arxiv.org/abs/2408.09121 |
| HTML | https://arxiv.org/html/2408.09121v6 |
| GitHub | https://github.com/magic-YuanTian/Selective-Prompt-Anchoring |

**prompt-distill 적용**: Tier 2 "anchor" 전략의 직접적 학술 근거. "2-5 토큰 키워드가 30토큰 설명을 대체할 수 있다"는 설계 가정을 SPA의 attention 증폭 메커니즘이 뒷받침. Anchored text가 모델의 기존 지식을 정확히 활성화.

### Knowledge Model Prompting — TMK Framework (2025)

> 구조화된 프롬프트가 모델의 잠재 기호 처리 능력을 활성화하는 메커니즘 연구.

- 제목: "Knowledge Model Prompting Increases LLM Performance on Planning Tasks"
- 핵심: Task-Method-Knowledge(TMK) 구조 프롬프트가 모델의 확률적 언어 패턴 의존을 줄이고 잠재적 기호 처리 능력을 활성화. 코드 학습 데이터에 내재된 형식적 추론 경로를 활성화하는 "symbolic steering mechanism"으로 작동.
- 결과: TMK 프롬프팅으로 Blocksworld 정확도 31.5% → 97.3% (불투명한 기호 작업에서).

| 자료 | URL |
|---|---|
| arXiv | https://arxiv.org/abs/2602.03900 |
| HTML | https://arxiv.org/html/2602.03900 |
| OpenReview | https://openreview.net/forum?id=23Ea4fVGiI |

**prompt-distill 적용**: Tier 2 앵커가 단순 "힌트"가 아니라 모델 내부의 특정 추론 경로를 활성화하는 메커니즘임을 시사. 올바른 키워드 선택이 단순 압축을 넘어 성능 향상까지 가능.

### Anchoring Bias in LLMs (2024-2025)

> 프롬프트 내 초기 정보가 LLM 판단에 불균형적 영향을 미치는 인지 편향 연구.

- 핵심: 앵커링 효과는 양날의 검 — 의도적으로 활용하면 원하는 행동을 유도하지만, 무의식적으로 포함되면 편향 유발.
- 완화: 단순 CoT, 원칙 사고, 앵커 무시 힌트는 불충분. 포괄적 배경 지식 제공만이 효과적.

| 자료 | URL |
|---|---|
| Springer (실험 연구) | https://link.springer.com/article/10.1007/s42001-025-00435-2 |
| arXiv (실험) | https://arxiv.org/html/2412.06593v1 |
| ScienceDirect (편향 연구) | https://www.sciencedirect.com/science/article/pii/S2214635024000868 |
| Empirical Study | https://arxiv.org/pdf/2505.15392 |

### Archetypal Anchoring (Community Framework, 2025)

> LLM 에이전트 행동 안정화를 위한 사용자 측 앵커링 프레임워크 가설.

| 자료 | URL |
|---|---|
| OpenAI Community | https://community.openai.com/t/hypothesis-stabilizing-llm-agent-behavior-via-archetypal-anchoring-user-side-framework/1249964 |

**prompt-distill 적용**: 앵커링의 이중성(활성화 vs 편향)을 Tier 2 설계 시 고려 필요. 앵커는 의도한 지식만 활성화하도록 정확한 키워드 선택이 중요.

---

## 9. Instruction Hierarchy & Prioritization Frameworks

> 지시문 우선순위 체계 — Tier 시스템의 구조적 선행 연구.

### The Instruction Hierarchy (OpenAI, 2024)

> LLM에게 우선순위별 지시문 따르기를 학습시키는 프레임워크.

- 제목: "The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions"
- 학회: NeurIPS 2024 Workshop
- 핵심: System prompt > User message > Third-party content 우선순위 체계. 하위 우선순위 지시가 상위와 충돌 시 선택적 무시 학습.
- 결과: GPT-3.5에 적용 시 학습 시 미노출 공격 유형에도 63% 향상된 견고성, 표준 능력 저하 최소.

| 자료 | URL |
|---|---|
| OpenAI 블로그 | https://openai.com/index/the-instruction-hierarchy/ |
| arXiv | https://arxiv.org/abs/2404.13208 |
| HTML | https://arxiv.org/html/2404.13208v1 |
| OpenReview | https://openreview.net/forum?id=vf5M8YaGPY |

### Control Illusion: The Failure of Instruction Hierarchies (2025)

> 현재 LLM 아키텍처의 지시 계층 구현 한계 분석.

- 제목: "Control Illusion: The Failure of Instruction Hierarchies in Large Language Models"
- 핵심: 현재 LLM 아키텍처에는 계층적 지시를 차별화하고 우선순위화하는 효과적 메커니즘이 부재. 프롬프팅 기반 조정과 파인튜닝 모두 지시 계층 강제를 완전히 해결하지 못함.

| 자료 | URL |
|---|---|
| arXiv | https://arxiv.org/html/2502.15851v1 |

**prompt-distill 적용**: Tier 시스템은 "우선순위별 지시 분류"라는 점에서 instruction hierarchy 연구와 직접 관련. 단, 우리의 목적은 보안(prompt injection 방어)이 아닌 효율성(토큰 절약). Control Illusion 논문의 한계는 Tier 2 앵커의 신뢰성에 대한 주의 요소.

---

## 10. Instruction Decomposition & Evaluation

> 지시문을 분해하여 평가하는 프레임워크 — 분류 정확도 검증에 활용.

### InFoBench (Qin et al., ACL 2024)

> 지시문을 세부 요구사항으로 분해하여 LLM의 지시 따르기 능력을 평가하는 벤치마크.

- 제목: "InFoBench: Evaluating Instruction Following Ability in Large Language Models"
- 학회: ACL 2024 Findings
- 핵심 메트릭: DRFR (Decomposed Requirements Following Ratio) — 복잡한 지시를 더 단순한 기준으로 분해하여 LLM의 각 측면 준수 여부를 세밀하게 분석.
- 구성: 500개 다양한 지시문 + 2,250개 분해된 질문, 5개 제약 유형(Content, Linguistic, Style, Format, Number).

| 자료 | URL |
|---|---|
| ACL Anthology | https://aclanthology.org/2024.findings-acl.772/ |
| arXiv | https://arxiv.org/abs/2401.03601 |
| PDF | https://aclanthology.org/2024.findings-acl.772.pdf |
| HuggingFace | https://huggingface.co/papers/2401.03601 |

**prompt-distill 적용**: InFoBench의 5개 제약 유형 분류는 Tier 분류 시 지시문 카테고리화에 직접 활용 가능. 특히 Format 제약(70.7% 기본값 신뢰 가능)은 Tier 1 후보, Conditional 제약(22.9%만 추측 가능)은 Tier 3-4로 매핑.

### Instruction Following Robustness Evaluation (ICLR 2025)

> 지시 따르기의 견고성과 지시 구분 능력 평가.

| 자료 | URL |
|---|---|
| OpenReview PDF | https://openreview.net/pdf?id=peZbJlOVAN |

### IOPO: Complex Instruction Following (ACL 2025)

> 복잡 지시 따르기를 위한 Input-Output Preference Optimization.

| 자료 | URL |
|---|---|
| ACL Anthology | https://aclanthology.org/2025.acl-long.1079.pdf |

### Enhancing Instruction Following via Multi-Agentic Optimization (2025)

> 평가 기반 다중 에이전트 워크플로로 프롬프트 지시 최적화.

| 자료 | URL |
|---|---|
| arXiv | https://arxiv.org/html/2601.03359 |

---

## 11. Prompt Design Theory & Quality Framework

> 프롬프트 품질의 이론적 기반 — "좋은 프롬프트란 무엇인가"에 대한 체계적 연구.

### What Makes a Good Natural Language Prompt? (ACL 2025)

> 21개 속성 × 6개 차원의 프롬프트 품질 평가 프레임워크.

- 제목: "What Makes a Good Natural Language Prompt?"
- 학회: ACL 2025
- 핵심: 2022-2025 NLP/AI 학회 논문 150편 이상 + 블로그 메타분석. 21개 속성을 6개 차원으로 분류한 property-centric 프레임워크.
- 발견: 단일 속성 개선이 추론 작업에서 가장 큰 효과. 다수 속성 동시 추가가 항상 유익하지는 않음.
- 한계: 자연어 프롬프트의 품질을 정량화하는 개념적 합의가 아직 부재.

| 자료 | URL |
|---|---|
| ACL Anthology | https://aclanthology.org/2025.acl-long.292/ |
| arXiv | https://arxiv.org/abs/2506.06950 |
| HTML | https://arxiv.org/html/2506.06950v1 |

**prompt-distill 적용**: "단일 속성 개선이 가장 효과적" 발견은 Tier 3 압축 시 하나의 핵심 정보만 보존하는 전략의 근거. 또한 21개 속성 프레임워크는 지시문의 어떤 측면이 실제로 중요한지 판별하는 체크리스트로 활용 가능.

### Why Prompt Design Matters and Works (ACL 2025)

> 프롬프트 설계의 이론적 근거 — 프롬프트가 답 공간의 탐색기(selector)로 작동함을 증명.

- 제목: "Why Prompt Design Matters and Works: A Complexity Analysis of Prompt Search Space in LLMs"
- 학회: ACL 2025
- 핵심: 프롬프트는 모델의 전체 hidden state에서 task-relevant 정보를 추출하는 selector로 기능. 각 프롬프트가 답 공간의 고유한 trajectory를 정의.
- 결과: 최적 프롬프트 탐색으로 추론 작업에서 50% 이상 성능 향상 가능. "think step by step" 같은 단순 CoT가 오히려 성능 저해 가능.

| 자료 | URL |
|---|---|
| ACL Anthology | https://aclanthology.org/2025.acl-long.1562/ |
| arXiv | https://arxiv.org/abs/2503.10084 |
| PDF | https://aclanthology.org/2025.acl-long.1562.pdf |

**prompt-distill 적용**: "프롬프트 = selector" 이론은 Tier 2 앵커의 메커니즘을 설명. 짧은 키워드라도 올바른 trajectory를 선택할 수 있다면, 긴 설명과 동등한 효과. 반대로, 잘못된 앵커는 잘못된 trajectory 선택 → Tier 2 키워드 선택의 중요성.

---

## 12. Prompt-Level Distillation (Not Model Distillation)

> 모델 가중치가 아닌 프롬프트/지시문 수준의 지식 증류 연구.

### Prompt-Level Distillation (PLD) (2025)

> Teacher 모델의 추론 패턴을 Student의 system prompt에 주입하는 비파라메트릭 증류.

- 제목: "Prompt-Level Distillation: A Non-Parametric Alternative to Model Fine-Tuning for Efficient Reasoning"
- 핵심: 대형 Teacher 모델에서 추론 패턴을 추출하여 소형 Student 모델의 System Prompt에 "portable library"로 주입. 모델 학습 없이 프롬프트만으로 추론 능력 전달.
- 결과: Gemma-3 4B에서 Macro F1 57% → 90.0% (StereoSet), 67% → 83% (Contract-NLI).
- 의의: 의사결정 과정이 투명하여 인간 검증 가능 — 규제 산업(법률, 금융) 적용에 이상적.

| 자료 | URL |
|---|---|
| arXiv | https://arxiv.org/abs/2602.21103 |
| HTML | https://arxiv.org/html/2602.21103 |

**prompt-distill 적용**: PLD는 "프롬프트에 지식을 증류한다"는 개념의 직접적 학술 구현. prompt-distill의 역방향 접근 — 기존 프롬프트에서 불필요한 지식을 추출/제거하여 압축하는 "역 증류".

### Automatic Prompt Optimization with Prompt Distillation — DistillPrompt (Dyagin et al., 2025)

> 프롬프트 공간을 탐색하며 증류, 압축, 집약으로 자동 최적화.

- 제목: "Automatic Prompt Optimization with Prompt Distillation"
- 저자: Ernest A. Dyagin et al. (ITMO University)
- 핵심: DistillPrompt는 다양한 프롬프트 후보 생성 → 학습 데이터 예시 주입 → 후보 집약 → 최적화 프롬프트 도출 → 반복 정제의 5단계 반복 파이프라인.
- 결과: 기존 방법(Grips 등) 대비 전체 데이터셋에서 평균 20.12% 향상.

| 자료 | URL |
|---|---|
| arXiv | https://arxiv.org/abs/2508.18992 |
| PDF | https://arxiv.org/pdf/2508.18992 |
| HTML | https://arxiv.org/html/2508.18992 |

**prompt-distill 적용**: DistillPrompt의 "증류 → 압축 → 집약" 파이프라인은 prompt-distill의 "분석 → 분류 → 압축" 파이프라인과 구조적 유사성. 자동화 확장 시 참고할 알고리즘.

### P-Distill: Prompt Compression via Knowledge Distillation (2025)

> Knowledge distillation을 통한 프롬프트 압축 — 1/8 길이에서 동등 이상 성능.

| 자료 | URL |
|---|---|
| MDPI | https://www.mdpi.com/2076-3417/15/5/2420 |

---

## 13. Prompt Optimization Frameworks

> 자동 프롬프트 최적화 도구 — prompt-distill의 자동화 확장 시 참고.

### DSPy (Stanford NLP)

> 프롬프트 대신 모듈 시그니처로 프로그래밍, 자동 최적화.

- 핵심: Declarative Self-improving Python. 프롬프트 엔지니어링을 프로그래밍 패러다임으로 전환.
- 최적화 방법:
  - COPRO: 각 단계별 새 instruction 생성/정제, coordinate ascent로 최적화.
  - MIPROv2: instruction + few-shot 예시를 Bayesian Optimization으로 탐색.

| 자료 | URL |
|---|---|
| 공식 사이트 | https://dspy.ai/ |
| GitHub | https://github.com/stanfordnlp/dspy |
| Optimizers 문서 | https://dspy.ai/learn/optimization/optimizers/ |
| DSPy 연구 사례 (TDS) | https://towardsdatascience.com/systematic-llm-prompt-engineering-using-dspy-optimization/ |

### OPRO: Optimization by PROmpting (Google, 2023)

> LLM 자체를 사용하여 프롬프트를 반복적으로 생성/탐색/정제.

- 핵심: 프롬프트 최적화를 black-box 문제로 취급, 성능 점수 기반 trajectory로 상위 프롬프트 유지/개선.

| 자료 | URL |
|---|---|
| n8n 구현 예시 | https://n8n.io/workflows/11495-automatically-optimize-ai-prompts-with-openai-using-opro-and-dspy-methodology/ |

### PromptWizard (Microsoft Research, 2024)

> 피드백 기반 자기 진화 프롬프트 최적화.

- 핵심: LLM의 반복적 피드백으로 프롬프트와 in-context 예시를 동시에 자기 진화/적응적으로 최적화.

| 자료 | URL |
|---|---|
| MS Research 블로그 | https://www.microsoft.com/en-us/research/blog/promptwizard-the-future-of-prompt-optimization-through-feedback-driven-self-evolving-prompts/ |

### Automatic Prompt Optimization via Heuristic Search (ACL 2025)

| 자료 | URL |
|---|---|
| PDF | https://aclanthology.org/2025.findings-acl.1140.pdf |

### Systematic Survey of Automatic Prompt Optimization (2025)

> 자동 프롬프트 최적화 기법의 포괄적 서베이.

| 자료 | URL |
|---|---|
| arXiv | https://arxiv.org/html/2502.16923v2 |

### Comprehensive Taxonomy of Prompt Engineering Techniques (Frontiers, 2025)

> 프롬프트 엔지니어링 기법의 포괄적 분류 체계.

| 자료 | URL |
|---|---|
| Springer | https://link.springer.com/article/10.1007/s11704-025-50058-z |

---

## 14. The Instruction Gap (2025)

> LLM이 복잡한 지시를 따르는 데 실패하는 메커니즘 분석.

- 제목: "The Instruction Gap: LLMs get lost in Following Instruction"
- 핵심: 지시가 복잡해질수록 LLM의 지시 따르기가 급격히 저하. 지시의 수와 복잡도가 성능의 핵심 제약.

| 자료 | URL |
|---|---|
| ResearchGate | https://www.researchgate.net/publication/399558992_The_Instruction_Gap_LLMs_get_lost_in_Following_Instruction |

**prompt-distill 적용**: "지시가 많을수록 성능 저하"는 Yang et al.의 "전부 명시하면 역효과" 발견과 직접 연결. Tier 1 제거와 Tier 2 압축의 추가 정당성.

---

## 15. Prompt Engineering Surveys (Comprehensive)

> 전체 프롬프트 엔지니어링 분야의 포괄적 서베이.

### Unleashing the Potential of Prompt Engineering (Patterns, 2025)

> 프롬프트 엔지니어링의 잠재력과 패턴 분석.

| 자료 | URL |
|---|---|
| Cell/Patterns (Full text) | https://www.cell.com/patterns/fulltext/S2666-3899(25)00108-4 |
| ScienceDirect | https://www.sciencedirect.com/science/article/pii/S2666389925001084 |

### Efficient Prompting Methods Survey (2024)

> Model-centric, Data-centric, Framework-centric 3가지 관점의 효율적 프롬프팅 서베이.

| 자료 | URL |
|---|---|
| arXiv HTML | https://arxiv.org/html/2404.01077v2 |
| arXiv | https://arxiv.org/abs/2404.01077 |

### Pre-train, Prompt, and Predict (ACM Computing Surveys)

> 프롬프팅 방법론의 체계적 서베이 — NLP 패러다임 전환 분석.

| 자료 | URL |
|---|---|
| ACM | https://dl.acm.org/doi/10.1145/3560815 |

### Prompt Engineering Methods for Different NLP Tasks (2024)

> 39개 프롬프팅 기법 × 29개 NLP 태스크 매핑 서베이.

| 자료 | URL |
|---|---|
| arXiv | https://arxiv.org/html/2407.12994v1 |

### Systematic Survey of Prompt Engineering (2024)

> 프롬프트 엔지니어링 기법과 응용의 체계적 서베이.

| 자료 | URL |
|---|---|
| arXiv | https://arxiv.org/abs/2402.07927 |

---

## 16. LLM Default Behaviors & Built-in Biases

> Tier 1 분류의 실증 근거 — LLM이 명시 없이도 보이는 기본 행동 패턴.

### Verbosity Compensation Behavior (2024)

> RLHF 학습된 모델은 기본적으로 장황한 답변을 생성. "간결하게 써라"는 지시는 기본값에 반하는 것.

- 제목: "Demystify Verbosity Compensation Behavior"
- 핵심: 모든 주요 LLM이 13.6%(Llama-3-70B) ~ 74%(Mistral-7B)의 verbosity compensation 빈도를 보임. RLHF가 "길수록 좋다"는 편향을 학습.
- GPT-4의 VC 빈도: 50.40%

| 자료 | URL |
|---|---|
| OpenReview | https://openreview.net/pdf?id=l49uZcEIcq |

**prompt-distill 적용**: "Be concise" 같은 지시는 Tier 1이 아니라 Tier 2-3 — RLHF 기본값에 *반하는* 지시이므로 앵커로 유지 필요.

### Instructed to Bias (TACL 2024)

> Instruction tuning이 인지 편향을 도입 — 기본 모델에는 없던 편향이 RLHF로 생성됨.

- 제목: "Instructed to Bias: Instruction-Tuned Language Models Exhibit Emergent Cognitive Bias"
- 핵심: decoy effect, certainty effect, belief bias 등이 instruction-tuned 모델에서 더 강하게 나타남.

| 자료 | URL |
|---|---|
| arXiv | https://arxiv.org/abs/2308.00225 |
| MIT Press (TACL) | https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00673/121541 |

### Sycophancy in Language Models (ICLR 2024)

> RLHF 모델의 아첨 행동 — "정직하게 답해라"는 지시가 기본값에 반하는 것.

- 제목: "Towards Understanding Sycophancy in Language Models"
- 핵심: 5개 SOTA AI 어시스턴트가 4개 free-form 텍스트 생성 태스크에서 일관된 sycophancy 표출. RLHF의 인간 선호 판단이 원인.

| 자료 | URL |
|---|---|
| arXiv | https://arxiv.org/abs/2310.13548 |
| ICLR PDF | https://proceedings.iclr.cc/paper_files/paper/2024/file/0105f7972202c1d4fb817da9f21a9663-Paper-Conference.pdf |

### Verbosity Bias in Preference Labeling (2023)

> LLM이 길이를 품질로 착각하는 체계적 편향.

- 핵심: GPT-4의 verbosity-bias 점수 0.328, GPT-3.5는 0.428. RLHF 학습 시 인간 평가자가 긴 답변을 선호한 결과.

| 자료 | URL |
|---|---|
| arXiv | https://arxiv.org/abs/2310.10076 |

**prompt-distill 적용**: LLM "기본값"은 단순히 "좋은 코드를 쓴다"만이 아니라 "장황하게 쓴다", "아첨한다" 같은 부정적 기본값도 포함. Tier 분류 시 이런 반-기본값(anti-default) 지시는 Tier 2-3로 보존해야 함.

### Prompting Inversion (Khan, 2025)

> 약한 모델에 도움이 되는 제약이 강한 모델에는 해로울 수 있음.

- 제목: "You Don't Need Prompt Engineering Anymore: The Prompting Inversion"
- 핵심: 규칙 기반 제약(Sculpting)이 GPT-4o에서는 93%→97%로 개선하지만, GPT-5에서는 96.36%→94%로 오히려 저하. "Guardrail-to-Handcuff" 전환 현상.

| 자료 | URL |
|---|---|
| arXiv | https://arxiv.org/abs/2510.22251 |
| GitHub | https://github.com/strongSoda/prompt-sculpting |

**prompt-distill 적용**: 모델 능력이 올라갈수록 Tier 1에 해당하는 항목이 늘어남. 모델-adaptive tier 분류의 이론적 근거.

### Do LLMs Know Internally When They Follow Instructions? (Apple, NeurIPS 2024)

> LLM 내부에 "instruction-following dimension"이 존재.

- 핵심: 입력 임베딩 공간에서 지시 준수를 예측하는 차원을 식별. 이 차원은 미지의 태스크에는 일반화되지만 미지의 지시 유형에는 불가.

| 자료 | URL |
|---|---|
| arXiv | https://arxiv.org/html/2410.14516v5 |
| Apple ML | https://machinelearning.apple.com/research/do-llms-know-internally |

---

## 17. Instruction Overload — "Curse of Instructions"

> 지시가 많을수록 성능이 지수적으로 저하 — Tier 1 제거의 가장 강력한 실증 근거.

### Curse of Instructions (2025)

> 10개 지시 동시 준수율: GPT-4o 15%, Claude 3.5 Sonnet 44%.

- 제목: "Curse of Instructions: Large Language Models Cannot Follow Multiple Instructions at Once"
- 핵심: 전체 성공률은 멱법칙: P(all) = P(individual)^N. 지시 수에 대해 지수적 감소.
- 벤치마크: ManyIFEval — 최대 10개 검증 가능한 지시 동시 부여.

| 자료 | URL |
|---|---|
| OpenReview | https://openreview.net/forum?id=R6q67CDBCH |

### How Many Instructions Can LLMs Follow at Once? (2025)

> IFScale 벤치마크: 500개 keyword-inclusion 지시로 밀도별 성능 측정.

- 핵심: 최고 수준 모델도 최대 밀도에서 68% 정확도. 3가지 저하 패턴 식별:
  - threshold decay (reasoning 모델: o3, gemini-2.5-pro)
  - linear decay (gpt-4.1, claude-sonnet-4)
  - exponential decay (gpt-4o, llama-4-scout)

| 자료 | URL |
|---|---|
| arXiv | https://arxiv.org/abs/2507.11538 |

### LLMs Can Be Easily Confused by Instructional Distractions (ACL 2025)

> 경쟁하는 지시가 존재하면 고급 LLM도 실패.

- 벤치마크: DIM-Bench — primary vs embedded 지시 구분 능력 평가.

| 자료 | URL |
|---|---|
| arXiv | https://arxiv.org/html/2502.04362v1 |

### Irrelevant Context Distraction (2023)

> LLM은 무관한 정보를 식별할 수는 있지만 무시하지는 못함.

| 자료 | URL |
|---|---|
| arXiv | https://arxiv.org/abs/2302.00093 |

**prompt-distill 적용**: P(all) = P(individual)^N 공식은 prompt-distill의 핵심 정당화. 58개 지시 중 23개(Tier 1)를 제거하면 35개만 남아 전체 준수율이 지수적으로 향상. "지시를 줄이는 것 자체가 성능 최적화".

---

## 18. Prompt Verbosity & Output Style

> 장황한 프롬프트가 장황한 출력을 유발 — 압축의 이중 효과(입력+출력 토큰 절약).

### Prompt Style Mirrors Output Style

| 자료 | URL | 핵심 |
|---|---|---|
| Anthropic Claude 4 가이드 | https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/claude-4-best-practices | "prompt formatting style influences output style" |
| Prompt Format Impact (2024) | https://arxiv.org/abs/2411.10541 | GPT-3.5-turbo: 포맷에 따라 성능 40% 차이 |
| Prompt Politeness & Accuracy | https://arxiv.org/abs/2510.04950 | 250개 프롬프트 × 5개 톤, 톤이 정확도에 영향 |
| Bad Prompts Hidden Cost | https://www.newtuple.com/post/the-hidden-cost-of-bad-prompts-why-typos-and-sloppy-formatting-sabotage-your-business-ai | 오타/불일치 포매팅이 정확도 최대 8% 하락 |

### Verbosity Compensation Research

| 자료 | URL | 핵심 |
|---|---|---|
| Verbosity != Veracity | https://arxiv.org/html/2411.07858v1 | GPT-4 VC 빈도 50.4%, 장황 응답 시 gold answer에 대한 확신이 낮음 |
| Verbosity Bias in Preference | https://arxiv.org/abs/2310.10076 | LLM이 길이를 품질로 착각 |
| Mitigating Length Bias in RLHF | https://arxiv.org/abs/2511.12573 | RLHF reward model이 장황함과 품질을 혼동 |

**prompt-distill 적용**: 디자인 문서의 "verbose CLAUDE.md produces verbose agent output" 주장에 대한 학술 근거 확보. 압축 효과가 입력 토큰 절약 + 출력 품질/간결성 향상의 이중 효과.

---

## 19. Optimal Prompt Length & "Less Is More"

> 프롬프트 길이의 최적점 — 500-1,200 토큰이 sweet spot.

### Same Task, More Tokens (ACL 2024)

> 동일한 추론 과제에서 입력 길이만 늘리면 성능 저하.

- 제목: "Same Task, More Tokens: The Impact of Input Length on the Reasoning Performance of Large Language Models"
- 핵심: FLenQA 데이터셋으로 동일 추론을 다른 길이로 테스트. 3,000 토큰부터 이미 성능 저하 시작 — 기술적 최대 컨텍스트 길이보다 훨씬 아래.

| 자료 | URL |
|---|---|
| arXiv | https://arxiv.org/abs/2402.14848 |

### Lost in the Middle (Stanford/UW, TACL 2024)

> U자형 성능 곡선 — 중간에 위치한 정보는 30%+ 성능 하락.

| 자료 | URL |
|---|---|
| arXiv | https://arxiv.org/abs/2307.03172 |

**prompt-distill 적용**: cache-friendliness 전략(정적 규칙을 상단에)의 학술 근거. 또한 SKILL.md 500줄 권장의 정량적 뒷받침.

### CLAUDE.md Optimization (Arize, 2025)

> CLAUDE.md 최적화로 SWE-Bench +5~15% 성능 향상.

- 핵심 발견:
  1. 지시 수를 줄이면 instruction-following 품질 향상
  2. 코드 스타일 가이드라인은 제거 — "linter가 할 일을 LLM에게 시키지 마라"
  3. 보편적으로 적용 가능한 지시만 포함
  4. Claude는 현재 태스크에 무관하다고 판단한 CLAUDE.md 내용을 무시 — 무관한 내용이 많을수록 전체를 무시할 확률 증가

| 자료 | URL |
|---|---|
| Arize 블로그 | https://arize.com/blog/claude-md-best-practices-learned-from-optimizing-claude-code-with-prompt-learning/ |
| GitHub (재현 코드) | https://github.com/Arize-ai/prompt-learning/tree/main/coding_agent_rules_optimization/claude_code |

**prompt-distill 적용**: prompt-distill과 가장 직접적으로 관련된 실증 연구. "무관한 내용이 많을수록 전체를 무시" 발견은 Tier 1 제거의 가장 실용적 근거. Arize의 수동 최적화를 prompt-distill이 자동화.

### Prompt Bloat Impact (MLOps Community)

> 소량의 무관 정보만으로도 일관성 없는 예측 유발.

| 자료 | URL |
|---|---|
| MLOps Community | https://mlops.community/the-impact-of-prompt-bloat-on-llm-output-quality/ |

### Context Window Management

| 자료 | URL | 핵심 |
|---|---|---|
| System Prompt vs Context Window | https://medium.com/data-science-collective/why-long-system-prompts-hurt-context-windows-and-how-to-fix-it-7a3696e1cdf9 | 시스템 프롬프트는 전체 컨텍스트의 5-10%로 제한 권장 |
| Anthropic: Effective Harnesses | https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents | 장기 에이전트 컨텍스트 관리 |
| RAG-MCP (2025) | https://arxiv.org/html/2505.03275v1 | 프롬프트 토큰 50%+ 절감, 도구 선택 정확도 3배 향상 |

---

## 20. Prompt Sensitivity & Instruction Sensitivity

> 프롬프트의 작은 변경이 성능에 큰 영향 — 압축 시 신중함의 근거.

### What Did I Do Wrong? (NAACL 2025)

> 구조적 변형(few-shot 순서, 출력 포맷)만으로 near-random에서 near-optimal까지 성능 변동.

| 자료 | URL |
|---|---|
| arXiv | https://arxiv.org/abs/2406.12334 |
| NAACL PDF | https://aclanthology.org/2025.naacl-long.73.pdf |

### On the Worst Prompt Performance (NeurIPS 2024)

> 최악/최선 프롬프트 성능 차이: Llama-2-70B-chat에서 최대 45.48%.

| 자료 | URL |
|---|---|
| arXiv | https://arxiv.org/abs/2406.10248 |

### Measuring Pragmatic Influence in LLM Instructions (2026)

> "이건 긴급하다", "상사로서 지시한다" 같은 화용론적 프레이밍이 모델 행동에 체계적 영향.

- 핵심: 400개 influence prefix × 13개 전략의 분류 체계. 태스크 내용 변경 없이도 프레이밍만으로 지시 우선순위화에 재현 가능한 변화.

| 자료 | URL |
|---|---|
| arXiv | https://arxiv.org/abs/2602.21223 |

**prompt-distill 적용**: 압축 시 의미 보존뿐 아니라 화용론적 뉘앙스(긴급성, 권위 프레이밍)도 고려 필요. MUST/ALWAYS 같은 강조가 단순 스타일이 아닌 실질적 행동 변화를 유발할 수 있음.

---

## 21. Existing Tools & Competitive Landscape

> prompt-distill과 유사하거나 관련된 기존 도구 분석 — 차별점 확인.

### 핵심 발견: prompt-distill이 채우는 공백

30+ 도구 조사 결과, **"LLM 기본 지식 tier로 지시문을 분류하는 도구는 없음"**:
- 기존 압축 도구: 토큰 단위 entropy/perplexity 기반 (LLMLingua 등)
- 기존 최적화 도구: 태스크 성능 향상 목적 (DSPy, PromptWizard 등)
- 가장 가까운 것: TechLoom의 CLAUDE.md 압축 수동 가이드

### 프롬프트 압축 라이브러리

| 도구 | URL | 차이점 |
|---|---|---|
| LLMLingua | https://github.com/microsoft/LLMLingua | 토큰 단위 entropy 기반, 의미 분류 없음 |
| SelectiveContext | https://github.com/liyucheng09/Selective_Context | self-information 기반, 지시문 특화 아님 |
| prompt-optimizer | https://github.com/vaibkumr/prompt-optimizer | 필러 단어 제거 수준, tier 개념 없음 |
| Semantic Prompt Compressor | https://github.com/metawake/prompt_compressor | spaCy 규칙 기반, ~22% 절약 |
| PCToolkit | https://github.com/3DAgentWorld/Toolkit-for-Prompt-Compression | 5개 압축기 벤치마크 프레임워크 |

### Claude Code 관련 도구

| 도구 | URL | 차이점 |
|---|---|---|
| claude-code-prompt-optimizer | https://github.com/johnpsasser/claude-code-prompt-optimizer | prompt-distill의 **반대** — 프롬프트 확장 도구 |
| claude-code-prompt-improver | https://github.com/severity1/claude-code-prompt-improver | 사용자 프롬프트 명확화, 정적 지시 파일 대상 아님 |
| OpenClaw Token Optimizer | https://github.com/openclaw-token-optimizer/openclaw-token-optimizer | 인프라 수준 최적화 (lazy loading, caching) |

### 상용 도구

| 도구 | URL | 차이점 |
|---|---|---|
| PromptPerfect (Jina AI) | https://promptperfect.jina.ai/ | 품질 최적화, 압축 아님 |
| OpenAI Prompt Optimizer | https://developers.openai.com/api/docs/guides/prompt-optimizer/ | 모순 제거/명확성 개선, 압축 아님 |
| Google Vertex Prompt Optimizer | https://docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/prompt-optimizer | eval 기반 반복 최적화, Gemini 전용 |
| Compressly.io | https://www.compressly.io/ | 브라우저 확장, 실시간 필러 제거 |
| Compresr (YC) | https://github.com/Compresr-ai/Context-Gateway | 대화 히스토리 압축, 지시 파일 대상 아님 |

### 분석/린팅 도구

| 도구 | URL | 차이점 |
|---|---|---|
| Prompt Linter (VS Code) | https://marketplace.visualstudio.com/items?itemName=Ignire.prompt-linter | 품질 분석, 압축 아님. 보완적 |
| PromptDoctor | https://arxiv.org/abs/2501.12521 | 편향/취약점 탐지, 학술 도구 |

### CLAUDE.md 관련 참고 자료

| 자료 | URL | 내용 |
|---|---|---|
| TechLoom 압축 가이드 | https://techloom.it/blog/compress-claude-md.html | 수동 60-70% 압축 기법 가이드 |
| claude-flow #585 | https://github.com/ruvnet/claude-flow/issues/585 | CLAUDE.md 40k 초과 문제 — 수요 증거 |
| Writing a Good CLAUDE.md | https://www.humanlayer.dev/blog/writing-a-good-claude-md | 실용 작성 가이드 |

### Awesome Lists

| 자료 | URL |
|---|---|
| awesome-context-engineering | https://github.com/jihoo-kim/awesome-context-engineering |
| Awesome-LLM-Compression | https://github.com/HuangOwen/Awesome-LLM-Compression |
| Prompt Compression Survey Repo | https://github.com/ZongqianLi/Prompt-Compression-Survey |
| awesome-claude-code | https://github.com/hesreallyhim/awesome-claude-code |

### Cross-Provider 프롬프팅 가이드

| 자료 | URL |
|---|---|
| OpenAI GPT-4.1 가이드 | https://cookbook.openai.com/examples/gpt4-1_prompting_guide |
| Google Gemini 프롬프팅 | https://ai.google.dev/gemini-api/docs/prompting-intro |
| Cross-Provider 비교 가이드 | https://dev.to/kenangain/one-stop-developer-guide-to-prompt-engineering-across-openai-anthropic-and-google-4bfb |

---

## 22. Individual Instruction Granularity — 단일 지시의 최적 길이

> 개별 지시문의 최적 단어/토큰 수에 대한 연구. **직접 측정한 연구는 부재 (research gap)**, 간접 증거로 수렴.

### 핵심 발견 요약

| 증거 | 시사점 | 출처 |
|---|---|---|
| SPA: 2-5 토큰 앵커로 Pass@1 12.9% 향상 | 키워드만으로 잠재 지식 활성화 가능 | SPA (ICML 2025) |
| "A single sentence is almost always sufficient" | 10-20 단어 문장이면 충분 | OpenAI GPT-4.1 가이드 |
| "5 bullets, each under 15 words" > "be concise" | 구체적 수치 제약이 모호한 형용사보다 효과적 | Anthropic 가이드 |
| 65.2% 요구사항이 명시 없이도 충족 | 이미 아는 내용은 0토큰도 충분 | Yang et al. 2025 |
| 강한 모델에서 제약이 오히려 해로움 | 모델이 강할수록 짧은 지시가 유리 | Prompting Inversion |
| P(all) = P(individual)^N | 불필요한 지시 하나가 전체 준수율 저하 | Curse of Instructions |

**실용적 결론**: 개별 지시의 최적 길이는 **5-25 단어** 범위이며, 핵심은 단어 수가 아니라 **의미의 구체성**. "Use ESLint" (2토큰)과 "Use ESLint for linting to ensure consistent style" (12토큰)은 모델이 ESLint를 이미 아는 경우 동등한 효과.

### DETAIL Framework (Kim, 2025)

> 프롬프트 구체성(specificity)이 성능에 미치는 영향을 체계적으로 측정.

- 제목: "DETAIL: Does Prompt Specificity Really Improve Task Performance?"
- 핵심: 30개 추론 태스크에서 고구체성 → 저구체성 프롬프트를 비교. **구체성은 절차적 태스크와 소형 모델에서 가장 유익**. 일부 태스크는 모호한 프롬프트가 오히려 모델이 효율적 내부 표현을 구축하도록 허용하여 유리.
- 의의: "항상 구체적일수록 좋다"는 통념에 반하는 nuanced 결과.

| 자료 | URL |
|---|---|
| arXiv | https://arxiv.org/abs/2512.02246 |

**prompt-distill 적용**: Tier 2 앵커의 적절한 구체성 수준 결정에 참고. 절차적/조건적 지시(Tier 3)는 구체적으로, 일반 지식(Tier 2)은 간결하게.

### More Than a Score (Zi, Menon, Guha, 2025)

> 코드 생성에서 최소 → 최대 구체성 프롬프트의 partial order로 성능 변화 측정.

- 제목: "More Than a Score: Evaluating LLM Code with PartialOrderEval"
- 핵심: pass@1이 프롬프트 구체성에 비례하여 증가. 핵심 개선 요인은 **명시적 I/O 명세, 엣지 케이스 처리, 단계별 분해** — 단순 단어 수 증가가 아님.

| 자료 | URL |
|---|---|
| arXiv | https://arxiv.org/abs/2508.03678 |

### Show and Tell (Bohr, 2025)

> 지시(추상적 규칙) vs 예시(구체적 시연)의 스타일 제어 효과 비교.

- 제목: "Show and Tell: Prompt Strategies for Style Control in Multi-Turn LLM Code Generation"
- 핵심: 160개 프로그램 쌍 비교. **지시 기반 프롬프트가 간결하지만 견고한 구현을 유도**. 반복 개발에서 지시는 필수, 예시와 결합 시 최강.

| 자료 | URL |
|---|---|
| arXiv | https://arxiv.org/abs/2511.13972 |

### Principled Instructions Are All You Need (Bsharat et al., 2024)

> 26개 프롬프팅 원칙을 LLaMA/GPT 계열에서 체계적 검증.

- 핵심: 명확하고 간결한 행동 동사(action verbs)가 장황한 표현보다 일관되게 우수.

| 자료 | URL |
|---|---|
| arXiv | https://arxiv.org/abs/2312.16171 |

### Instruction Following 벤치마크에서의 복잡도별 준수율

| 벤치마크 | URL | 핵심 발견 |
|---|---|---|
| **IFEval** (Zhou et al., 2023) | https://arxiv.org/abs/2311.07911 | 25가지 검증 가능 지시 유형, ~500 프롬프트 |
| **MOSAIC** (2026) | https://arxiv.org/abs/2601.18554 | 최대 20개 제약, 위치 효과(primacy/recency) 발견 |
| **RECAST-30K** (2025) | https://arxiv.org/abs/2505.19030 | 30k 인스턴스 × 19개 제약 유형. 규칙 기반 제약이 모델 기반보다 어려움 |
| **Multi-Dim Constraint** (2025) | https://arxiv.org/abs/2505.07591 | Level I 77.67% → Level IV 32.96%로 급락 |
| **EIFBENCH** (EMNLP 2025) | https://arxiv.org/abs/2506.08375 | "Extremely Complex Instruction Following" |
| **LIFBench** (ACL 2025) | https://arxiv.org/abs/2411.07037 | 2,766 지시, 6개 길이 구간, 128k 토큰까지 |
| **AgentIF** (2025) | https://arxiv.org/html/2505.16944v1 | "지시 길이 증가에 따라 성능 저하" 직접 언급 |
| **Scaling Reasoning, Losing Control** (2025) | https://arxiv.org/abs/2505.14810 | 추론 특화 모델이 오히려 지시 준수에서 퇴보. 최고 50.71% |

### DECRIM: 지시의 원자적 분해 (Amazon, 2024)

> 지시를 task + context + constraints 3요소로 분해하는 최소 유효 단위 정의.

- 핵심: 302개 지시 → 1,055개 제약으로 분해. 이 3부분 분해가 신뢰할 수 있는 평가의 최소 단위.

| 자료 | URL |
|---|---|
| Amazon Science PDF | https://assets.amazon.science/54/04/2dd88903469b9c7e2ef48769eb1c/llm-self-correction-with-decrim-decompose-critique-and-refine-for-enhanced-following-of-instructions-with-multiple-constraints.pdf |

### 제공업체별 개별 지시 가이드라인

**Anthropic**:
- "5 bullets, each under 15 words" > "be concise" — 구체적 수치 제약 선호
- "sequential steps using numbered lists or bullet points"
- https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/be-clear-and-direct

**OpenAI**:
- "A single sentence firmly and unequivocally clarifying your desired behavior is almost always sufficient"
- "It's generally not necessary to use all-caps or other incentives like bribes or tips"
- https://cookbook.openai.com/examples/gpt4-1_prompting_guide

**Google**:
- "State your request in brief -- but specific -- language"
- 컨텍스트를 먼저 배치하고 구체적 지시는 프롬프트 끝에
- https://ai.google.dev/gemini-api/docs/prompting-strategies

### 연구 공백 (Research Gap)

**단일 지시의 단어 수를 의미 내용을 고정한 채 체계적으로 변화시켜 준수율을 측정한 통제 실험은 아직 없음.** DETAIL 프레임워크가 가장 가까우나 전체 프롬프트 수준에서 작동. 이는 prompt-distill이 실험으로 기여할 수 있는 영역.

**prompt-distill 적용**:
- Tier 2 앵커의 최적 길이: 2-5 토큰 (SPA 근거) — 모델이 이미 아는 개념 활성화에 충분
- Tier 3 축약의 최적 길이: 8-20 단어 (DETAIL + OpenAI "single sentence" 근거) — 프로젝트 특화 규칙에 적절
- Tier 4 보존: 길이 제한 없음, 의미 완전성이 우선
- 벤치마크 활용: InFoBench DRFR로 압축 전후 개별 지시 준수율 검증 가능
