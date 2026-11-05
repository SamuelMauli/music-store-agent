# Tom: the customer service agent for Empório da Música

> LangGraph conversational agent that serves the customers of a fictional musical instrument store in Campo Grande/MS, Brazil. Numbers (price, installment, deadline, ticket number) come from deterministic code and verified data; the language model converses, chooses tools and writes the reply. It runs with any LLM provider, switches providers through configuration and goes to production without code changes.

[![CI](https://github.com/SamuelMauli/music-store-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/SamuelMauli/music-store-agent/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB)
![Coverage 95%](https://img.shields.io/badge/coverage-95%25-brightgreen)
![mypy strict](https://img.shields.io/badge/mypy-strict-blue)
![ruff](https://img.shields.io/badge/lint-ruff-261230)
[![License MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## TL;DR

```bash
make setup && cp .env.example .env          # dependencies pinned by uv.lock
make ingest                                 # bronze -> silver -> gold + quality report
EMPORIO_LLM_PROVIDER=fake make chat         # offline conversation, no key and no network
```

To use a real model, set the provider, models and key in `.env` (examples in [Any AI, any model](#any-ai-any-model)) and run `make llm-check` before `make chat` or `make api`.

**In one sentence:** a 9-node LangGraph graph with deterministic guards on input and output, 13 typed tools over a DuckDB in medallion layers, manual rules written as pure functions, 15 LLM providers behind a single port, shared state in Postgres and Redis, and a production pipeline (read-only image, Kubernetes, CI with Trivy, signed release with SBOM).

## Contents

1. [Challenge compliance matrix](#challenge-compliance-matrix)
2. [Quickstart](#quickstart)
3. [Architecture and system complexity](#architecture-and-system-complexity)
4. [Any AI, any model](#any-ai-any-model)
5. [Production readiness](#production-readiness)
6. [Evidence](#evidence)
7. [Evaluation (evals)](#evaluation-evals)
8. [Assumptions and discrepancies between data and manual](#assumptions-and-discrepancies-between-data-and-manual)
9. [Conversation examples](#conversation-examples)
10. [Beyond the brief](#beyond-the-brief)
11. [Technical decisions (ADRs)](#technical-decisions-adrs)
12. [Honest limitations and next steps](#honest-limitations-and-next-steps)
13. [From PoC to scale](#from-poc-to-scale)
14. [Built with Claude Code](#built-with-claude-code)
15. [Folder structure and commands](#folder-structure-and-commands)

## Challenge compliance matrix

Each requirement from the brief, with its status and where to check it.

| Brief requirement | Status | Evidence |
| --- | --- | --- |
| Python project (the only mandatory constraint) | Met | `pyproject.toml` (`requires-python = ">=3.12"`), `.python-version` |
| Text messaging agent that serves customers | Met | CLI `src/emporio/interfaces/cli.py:94` (`make chat`) and SSE API `src/emporio/interfaces/api.py:218` (`make api`) |
| Persona aligned with the store's identity and tone | Met | "Tom" persona in `src/emporio/agent/prompts/system.jinja:1`; introduction on the 1st turn in `src/emporio/agent/nodes/finish.py:59` |
| Receive and answer based on the available context | Met | 13 tools over DuckDB (`src/emporio/agent/tools/__init__.py:31`); manual loaded in `policy_sections`; groundedness guard in `src/emporio/guardrails/output.py:184` |
| Know when to query data and when to query policies | Met | Router with 15 intents (`src/emporio/agent/intents.py`); `tools_for` (`src/emporio/agent/intents.py:119`) gives the model only the tools for the intent; `POLICY_INTENTS` (`src/emporio/agent/intents.py:58`) injects the manual sections |
| Handle out-of-scope questions | Met | `scope_gate` (`src/emporio/agent/scope.py:54`) answers by template, without calling the answering model; adversarial cases in `evals/adversarial.yaml` |
| Decision: agent framework / approach | Justified | [ADR-1](#technical-decisions-adrs) (LangGraph with explicit graph and bounded ReAct) |
| Decision: model and provider | Justified | [ADR-5](#technical-decisions-adrs) and [ADR-9](#technical-decisions-adrs); [Any AI, any model](#any-ai-any-model) |
| Decision: interaction interface | Justified | [ADR-8](#technical-decisions-adrs) (thin CLI and API over a `ChatService`) |
| Decision: history persistence | Justified | [ADR-6](#technical-decisions-adrs) and [ADR-10](#technical-decisions-adrs) (SQLite or Postgres checkpointer, window + summary) |
| Decision: data handling | Justified | [ADR-3](#technical-decisions-adrs); medallion in `src/emporio/ingestion/`; rules D1 to D14 in [Assumptions](#assumptions-and-discrepancies-between-data-and-manual) |
| Decision: retrieval architecture | Justified | [ADR-4](#technical-decisions-adrs), with measured ablation (`make eval-retrieval`) |
| Decision: prompt strategy | Justified | [ADR-2](#technical-decisions-adrs) and [ADR-11](#technical-decisions-adrs) (Jinja prompts versioned by hash) |
| Public Git repository with real commit history | Met | `git log`: small commits per slice (e.g. `550e71f`, `78116f9`, `dd844e3`, `05885ed`) |
| README: run instructions (environment, provider, commands) | Met | [Quickstart](#quickstart), [Any AI, any model](#any-ai-any-model), [Make commands](#make-commands) |
| README: rationale for technical decisions | Met | [Technical decisions (ADRs)](#technical-decisions-adrs) |
| README: known limitations and what I would do with more time | Met | [Honest limitations and next steps](#honest-limitations-and-next-steps) |
| README: use of code assistants and workflow | Met | [Built with Claude Code](#built-with-claude-code) |
| Between 3 and 5 example conversations, varied scenarios | Met | 5 files in `examples/` (`01_` to `05_`), in Portuguese on purpose (Brazilian store) |
| At least one non-trivial conversation (real-time data or policy rule) | Met | `examples/03_devolucao_arrependimento.txt` (identity verification + §4.1 rule with computed deadline) |
| Ambiguity: assume a reasonable interpretation and document it | Met | [Assumptions and gaps in the manual](#assumptions-and-gaps-in-the-manual) |

## Quickstart

Prerequisites: [uv](https://docs.astral.sh/uv/) and Python 3.12 (uv resolves the version from `.python-version`).

```bash
make setup && cp .env.example .env   # uv sync --frozen --all-extras (same as CI)
make ingest                          # generates data/emporio.duckdb and data/quality_report.json
make chat                            # interactive CLI (Typer + Rich); commands /debug, /reset, /sair
make api                             # or: FastAPI API at http://127.0.0.1:8000
```

With no API key and no network, using the deterministic provider:

```bash
EMPORIO_LLM_PROVIDER=fake make chat ARGS="--debug"
```

```text
você> Quais violões vocês têm até R$ 1.000?
você> e de aço?
você> quanto fica o Yamaha F310 no PIX?
```

(The sample messages are in Portuguese because the agent serves Brazilian customers: "which acoustic guitars do you have up to R$ 1,000?", "and steel-string ones?", "how much is the Yamaha F310 with PIX?")

With `--debug` (or `/debug` during the conversation) the technical panel shows, per turn, the routed intents, the tools called, the manual sections cited, the tokens and the estimated cost.

`make setup` installs all extras, the same environment as CI, so `make check` runs without adjustment. For a lean runtime, `make setup-min` installs only the base; providers (Anthropic, Google, Bedrock) and state in Postgres/Redis come in as extras: `uv sync --extra anthropic`, `--extra google`, `--extra bedrock`, `--extra postgres`, `--extra redis` or `--extra all`.

### API

```bash
curl -N http://127.0.0.1:8000/v1/chat \
  -H 'content-type: application/json' \
  -d '{"message": "Quanto custa o Takamine GD20?"}'
```

`POST /v1/chat` responds with Server-Sent Events: `step` events (graph node), `token` (chunks of the reply), `done` (turn summary) and `error`. The `session_id` is issued and signed by the server: the 1st message goes without it and the token comes back in the `X-Session-Id` header and in the final event; send it again to continue the conversation. Errors use a single envelope `{"error": {"code", "message", "request_id"}}` with the `X-Request-ID` header. Auxiliary routes: `GET /healthz` (liveness), `GET /readyz` (queries DuckDB) and `GET /metrics` (Prometheus).

### Docker and full stack

```bash
make docker-build                    # local image with EXTRAS=all
cp deploy/.env.example deploy/.env   # fill in the secrets
make compose-up                      # api + postgres + redis + otel-collector
make compose-up PROFILE=observability   # same, with Jaeger at http://localhost:16686
```

## Architecture and system complexity

### Overview

The system has three well-separated layers of responsibility:

1. **Trustworthy data**: the CSVs and the manual PDF go through a medallion ingestion in DuckDB, validated by contract. The agent only reads gold views through parameterized queries.
2. **Exact rules**: everything that involves a calculation or has consequences for the customer (PIX, installments, shipping, return deadline, opening hours, ticket number) is a pure function in `src/emporio/policies/`, on decimal `Money`.
3. **Conversation**: a LangGraph graph orchestrates guards, routing, identity verification, a bounded ReAct loop and answer validation. The LLM decides which tool to call and writes the reply; it is never the source of a number.

### The graph, node by node

The names below are the ones registered in `src/emporio/agent/graph.py` (`build_graph`).

```mermaid
flowchart LR
    S([message]) --> IG[input_guard]
    IG -->|continue| RT[router]
    RT -->|continue| SG[scope_gate]
    SG -->|continue| V[verification]
    V -->|continue| AG[agent<br/>ReAct, up to 4 tool calls]
    AG --> OG[output_guard]
    OG -->|rejected 1x: regenerate| AG
    OG -->|approved| R[respond]
    IG & RT & SG & V -->|answer ready| R
    IG & RT & SG & V -->|handoff| H[handoff<br/>opens ticket]
    AG -->|LLM failure| H
    OG -->|rejected 2x| H
    H --> R
    R --> SU[summarize]
    SU --> E([end of turn])
```

The conditional edges follow a single rule (`_after` in `graph.py`): if the state has `handoff`, go to `handoff`; if it already has `answer`, go to `respond`; otherwise, continue to the next node. This lets any node end the turn safely.

| Node | What it does | Why it exists | When it fails |
| --- | --- | --- | --- |
| `input_guard` (`agent/nodes/intake.py`) | Normalizes the text (NFKC, removes invisible and control characters, truncates at `MAX_INPUT_CHARS = 2000`), redacts PII before writing to state, resets the turn fields and expires a stale verification (`EMPORIO_VERIFICATION_TTL_SECONDS`). | Raw text with an email, phone number, CPF (Brazilian taxpayer ID) or card number must never reach the LLM, the checkpoint or the log. Obvious injection has to stop before it costs a model call. | It does not call an LLM, so it cannot fail because of a provider. On detecting injection (`looks_like_injection`), it replies with the standard refusal, sets `blocked` and goes straight to `respond`. |
| `router` (`agent/nodes/intake.py`, `agent/router.py`) | Classifies the message into up to 15 intents with structured output on the `small` model, extracts entities and resumes pending items ("qual o número do pedido?" ("what is the order number?") followed by "123"). | Decide between data and policies before generating. The cheap model does the routing, the expensive one only the answer. | `LLMError` becomes a safe technical reply with `handoff` (`failure` in `agent/nodes/deps.py:99`). Confidence below `CONFIDENCE_THRESHOLD = 0.6` becomes a clarification request in `scope_gate`. |
| `scope_gate` (`agent/nodes/intake.py`, `agent/scope.py`) | Answers by template whatever is out of scope (accessories, general topics, empty categories 6, 7 and 8) and passes only the remaining intents on to the agent, with notices (`scope_notes`). | A refusal does not need a large model: it is cheaper, predictable and does not hallucinate. | Deterministic, no external dependency. If no intent remains, it ends with the template; if some remain, it continues with the notes attached. |
| `verification` (`agent/nodes/verify.py`) | For `order_status` and `return_exchange`, requires the order number and an email or phone and calls `verify_customer` directly, outside the ReAct loop. | Anti-BOLA: no order data leaves without confirming the owner. The model never decides whether the customer was verified. | Missing data: asks for what is missing and stores the pending item. Wrong data: records the attempt per order, per contact and per IP (`AttemptStore`) and asks again. On the 3rd failure in the session (`MAX_ATTEMPTS = 3`) or with the target blocked, it ends with `handoff`. |
| `agent` (`agent/nodes/react.py`) | ReAct loop on the `medium` model with a rendered system prompt (date, store open or closed, verified customer, manual sections). It receives only the tools for the intent (`tools_for`) and has a budget of `MAX_TOOL_CALLS = 4`. | It is the only point where the model has freedom, and that freedom is bounded: few tools, fixed budget, rules in code. | A tool outside the list gets the error `NOT_ALLOWED` and the loop continues. `LLMError` or an empty answer becomes `failure` with `handoff`. |
| `output_guard` (`agent/nodes/guard.py`, `guardrails/output.py`) | Checks the answer: every monetary value, percentage and decimal must exist in the tool outputs or in the manual; an `EMP-...` ticket number only if it came from `open_ticket`; no PII; no prompt echo (8-word shingles); a manual citation when the rule tool answered. | Numeric hallucination is the costliest risk in store customer service (it becomes misleading advertising). | Rejected: regenerates once (`MAX_REGENERATIONS = 1`) with the reason as feedback. Rejected again: replaces it with the safe answer and escalates to a human. |
| `handoff` (`agent/nodes/finish.py:28`) | Opens a ticket (idempotent) and reports the ticket number, SLA and store phone. | Every failure must end in a traceable human path, not an error. | If the ticket cannot be opened, it replies with the store phone (`HANDOFF_NO_TICKET`). |
| `respond` (`agent/nodes/finish.py:59`) | Joins the `scope_gate` notices to the answer, introduces Tom on the 1st turn and writes the assistant message. | A single exit point keeps the shape of the reply consistent. | Deterministic. |
| `summarize` (`agent/nodes/memory.py`) | Keeps the last `HISTORY_WINDOW = 12` messages and summarizes the rest on the `small` model, in batches of at least 4 (`SUMMARY_BATCH`). | Long conversations without unbounded growth of context (and cost). | `LLMError` keeps the previous summary and logs a warning; the turn has already been delivered. |

### Hexagonal architecture

```mermaid
flowchart TB
    subgraph Interfaces
        CLI[CLI Typer/Rich]
        API[API FastAPI + SSE<br/>middleware: auth, request_id,<br/>body limit, metrics]
    end
    subgraph Aplicacao[Application]
        SVC[ChatService<br/>rate limit, trace, cost]
        GRAPH[LangGraph graph<br/>9 nodes]
        TOOLS[ToolRegistry<br/>13 tools]
    end
    subgraph Dominio[Pure domain]
        ENT[Entities and value objects<br/>Money, OrderId, Email, Phone]
        POL[Policies as code<br/>payment, returns, shipping, hours]
        SPEC[Specifications]
    end
    subgraph Portas[Ports - Protocols]
        P1[ProductRepository, OrderRepository<br/>CustomerRepository, PromotionRepository]
        P2[PolicyRetriever, Embedder, VectorIndex]
        P3[TicketRepository, Clock, RateLimiter]
        P4[LLMProvider]
        P5[AttemptStore, BaseCheckpointSaver]
    end
    subgraph Adaptadores[Adapters]
        A1[DuckDB repositories<br/>fuzzy name match]
        A2[LongContext / Hybrid<br/>BM25 + dense + RRF]
        A3[Tickets: SQLite or Postgres<br/>append-only]
        A4[LangChainLLM over BaseChatModel<br/>ResilientLLM + TracingLLM / FakeLLM]
        A5[Limits and attempts: memory or Redis<br/>Checkpointer: SqliteSaver or PostgresSaver]
    end
    BOOT[[bootstrap.py<br/>composition root]]
    CLI --> SVC
    API --> SVC
    SVC --> GRAPH --> TOOLS
    TOOLS --> POL & ENT & SPEC
    TOOLS --> P1 & P2 & P3
    GRAPH --> P4 & P5
    P1 --> A1
    P2 --> A2
    P3 --> A3
    P4 --> A4
    P5 --> A5
    BOOT -. chooses via Settings .-> A1 & A2 & A3 & A4 & A5
```

The domain (`src/emporio/domain`, `src/emporio/policies`) imports no framework or database. The data ports live in `src/emporio/ports.py`; `LLMProvider` in `src/emporio/agent/llm/types.py`; `AttemptStore` in `src/emporio/guardrails/attempts.py`. `src/emporio/bootstrap.py` is the composition root, the only place that chooses implementations from `Settings`:

- `build_shared_state` chooses the checkpointer and tickets (`state_backend`: `sqlite` or `postgres`) and the limits and attempts (`redis_url` present or not);
- `build_container` wires DuckDB, retriever, tool registry, LLM (`factory.build`), tracing (`TracingLLM`) and graph.

Swapping the LLM provider, retriever, clock or repository does not touch business rules, which is how the tests inject a fixed clock, a fake LLM and an in-memory repository.

### Data: DuckDB in medallion layers

```mermaid
flowchart LR
    RAW[(data/raw<br/>CSVs + manual PDF)] --> B[bronze<br/>faithful copy, all text]
    B --> S[silver<br/>types, enums, specs per category]
    B -.rejected, with reason.-> Q[(silver.quarantine_*)]
    C[[data/contracts<br/>ODCS contracts]] -.validates row by row.-> S
    S --> G[gold<br/>v_catalog, v_order_status,<br/>v_active_promotions]
    S --> QR[quality report<br/>rules D1-D14]
    PDF[manual PDF] --> PP[section parser] --> PS[(policy_sections)]
    G --> AG[agent tools]
    PS --> AG
```

- **Bronze**: faithful copy of the CSV, interpreting nothing; the source in `data/raw/` is immutable.
- **Silver**: typing, enums (`product_status`, `order_status`, `payment_method`) and specs normalized per category. A row that violates the contract goes to `silver.quarantine_<table>` with a hash, the original JSON and the reason; the others get in.
- **Contracts**: one YAML per table in `data/contracts/`, a subset of ODCS v3 (domain, data product, owner, schema with type, required, unique, enum, minimum and reference, freshness SLA and consumers).
- **Gold**: three views consumed by the agent. The effective price is already computed in `v_catalog`; the agent never writes free-form SQL, only queries parameterized by the repositories. The connection is opened in read-only mode (`duckdb.connect(..., read_only=True)` in `bootstrap.py`).
- **Quality report**: rules D1 to D14, written to `data/quality_report.json`; the reference snapshot in `tests/snapshots/quality_report.json` is checked by a test.
- **Manual**: the PDF is split into 28 sections with number and breadcrumb, loaded into `policy_sections` with a corpus hash.

### Policies as code

Pure functions over `Money` (decimal, never float) in `src/emporio/policies/`:

- `payment.py`: PIX with 5% off the list price (does not stack with promotions), interest-free installments by tier, combined payment. Covered by property-based tests (Hypothesis).
- `returns.py`: withdrawal right (7 days), defect (30 days), preference (7 days), legal warranty (90 days), with a computed deadline.
- `shipping.py`: free shipping above R$ 500,00, flat fee of R$ 35,00 in the metropolitan region, modalities and lead times for other cities.
- `store.py`: opening hours, open or closed at a given instant, contacts.

### Manual retrieval

Two strategies behind the `PolicyRetriever` port, chosen by `EMPORIO_POLICY_RETRIEVER`:

- `long_context` (default): hands the model every section that has a body, in manual order. All 28 sections fit in the context; with no index there is no retrieval false negative.
- `hybrid`: BM25 over normalized text + dense embeddings (`paraphrase-multilingual-MiniLM-L12-v2` via fastembed) fused by Reciprocal Rank Fusion (k = 60). It exists for the ablation and for the day the manual no longer fits in the context.

### Security

| Threat | Control | Where |
| --- | --- | --- |
| Prompt injection | NFKC normalization, removal of invisible characters, PT/EN patterns; defense in depth through the output guard and least privilege for tools | `guardrails/input.py`, `agent/nodes/intake.py` |
| BOLA (access to another customer's order) | `verification` checks the order + the owner's email or phone before any order tool; `verify_customer` runs in the node, never in the ReAct loop; the model only receives the tools for the intent | `agent/nodes/verify.py`, `agent/intents.py:119` |
| Brute force on verification | Attempts counted per order, per contact and per IP, with hashed keys and temporary lockout (`EMPORIO_VERIFY_MAX_ATTEMPTS_PER_TARGET`, `_PER_IP`, `EMPORIO_VERIFY_LOCKOUT_SECONDS`) | `guardrails/attempts.py` |
| Forged or guessed session | `session_id` of 128 random bits signed with HMAC-SHA256 (128-bit tag); the client does not choose the id; an invalid token returns `400 invalid_session` | `guardrails/session.py:22`, `interfaces/api.py` |
| Numeric hallucination (groundedness) | Every monetary value, percentage, thousands figure and decimal in the answer must exist in the tool outputs, in the arguments, in the manual or in the turn context | `guardrails/output.py:184` (`check_answer`) |
| Invented ticket number | `EMP-...` in the answer must come from `open_ticket` | `guardrails/output.py` (`ungrounded_protocols`) |
| PII | Identity redacted on input (`[contato omitido]`, `[cpf omitido]`, `[cartão omitido]`, the latter only with a valid Luhn check digit); output guard blocks PII (`leaked_pii`); JSON logs drop the customer's text and replace the session id with a hash; `mask_pii` for traces | `agent/identity.py`, `domain/pii.py:91`, `observability/logging.py` |
| Abuse and cost | Sliding window per session and per IP (`EMPORIO_RATE_LIMIT_PER_MINUTE`, `EMPORIO_RATE_LIMIT_PER_IP_PER_MINUTE`), `429` with `Retry-After`, character limit (`EMPORIO_MAX_MESSAGE_CHARS`, `422`) and body limit (`413`), 4 tool calls per turn | `guardrails/rate_limit.py`, `interfaces/middleware.py` |
| Exposed API | Key via `Authorization: Bearer` or `X-API-Key`, compared with `hmac.compare_digest`; several keys accepted for zero-downtime rotation | `interfaces/middleware.py:84` |
| Secrets | `SecretStr` on all credentials; LLM error messages carry only the class name and HTTP status (`classify` in `agent/llm/errors.py`); `llm-check` shows the key source, never the value | `settings.py`, `agent/llm/doctor.py` |

### State: Postgres and Redis behind ports

| State | Development (default) | Production | Why |
| --- | --- | --- | --- |
| Conversation memory (LangGraph checkpointer by `thread_id`) | `SqliteSaver` in `data/checkpoints.sqlite` | `PostgresSaver` over a `psycopg` pool (`EMPORIO_STATE_BACKEND=postgres`) | Multiple replicas need to see the same session. |
| Tickets | `SqliteTicketRepository` | `PostgresTicketRepository`, append-only via trigger (`tickets_no_mutation` blocks `UPDATE`/`DELETE`, `tickets_no_truncate` blocks `TRUNCATE`), migrations versioned in `schema_version` with `pg_advisory_xact_lock` | A ticket is a service record: it is not edited or deleted, and replicas start without a migration race. |
| Rate limit | In-memory `SlidingWindowLimiter` | `RedisSlidingWindowLimiter` (sorted set in a transactional pipeline, prefixes `emporio:rl:session:` and `emporio:rl:ip:`) | The limit must hold across replicas. |
| Verification attempts | `InMemoryAttemptStore` | `RedisAttemptStore` (`INCR` + `PEXPIRE` in a pipeline) | The brute-force lockout cannot be bypassed by switching replicas. |

The same contract tests (`tests/contracts/`) run against the in-memory implementations and against real Postgres and Redis.

### Where the real complexity lives

- **Conditional edges, not the LLM.** The hard part is guaranteeing that every path ends in a safe answer: injection, incomplete verification, LLM down, answer rejected twice, a ticket that fails to open. Each of these paths has a node, a test and a span.
- **Numeric groundedness.** Comparing numbers in free text against tool outputs requires normalizing Brazilian formats (`1.000,00`), percentages, decimals and values the customer cited themselves (`denied_echoes` allows repeating a percentage only to deny it when the promotion has ended).
- **Identity without PII.** The identity is extracted from the raw text on every turn, used only inside the node and never persisted; the state holds the redacted version.
- **Portability across providers.** Each SDK has its own way of raising errors, requiring system messages and doing structured output; this was encapsulated in `providers.py`, `errors.py` and `messages.py`.
- **Data with deliberate defects.** The 14 quality rules became explicit handling, not silent cleaning.

**Trade-offs accepted:**

| Choice | Gain | Cost |
| --- | --- | --- |
| Explicit graph instead of a free-form agent | Predictable latency, cost and failures; one span per node | More orchestration code |
| Rules in code instead of in the prompt | Correct, auditable numbers | Every new manual rule becomes code and a test |
| Strict output guard | No number without a source reaches the customer | Occasional false positives become a safe answer with handoff |
| `long_context` instead of vector RAG | No retrieval false negatives | Does not scale to large manuals (`hybrid` is ready) |
| Two model tiers | The expensive model only generates the answer | Two configurations to validate (`make llm-check` covers both) |
## Any AI, any model

### `provider:model` spec

Each role receives a spec in the `provider:model` format (`parse_spec` in `src/emporio/agent/llm/spec.py`). Only a registered provider counts as a prefix, so `qwen2.5:7b` and `openai/gpt-oss-120b` remain model names of the default provider (`EMPORIO_LLM_PROVIDER`). This allows, for example, the router on one provider and the answer on another.

### Supported providers

Registry in `src/emporio/agent/llm/providers.py` (`REGISTRY`).

| Provider (spec) | LangChain adapter | uv extra | Credential | Default endpoint |
| --- | --- | --- | --- | --- |
| `openai` | `ChatOpenAI` | base | `OPENAI_API_KEY` | SDK default |
| `azure_openai` | `AzureChatOpenAI` | base | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` (version: `EMPORIO_LLM_AZURE_API_VERSION`, `AZURE_OPENAI_API_VERSION` or `OPENAI_API_VERSION`; default `2024-10-21`) | `AZURE_OPENAI_ENDPOINT` |
| `anthropic` | `ChatAnthropic` | `anthropic` | `ANTHROPIC_API_KEY` | SDK default |
| `google` | `ChatGoogleGenerativeAI` | `google` | `GOOGLE_API_KEY` or `GEMINI_API_KEY` | SDK default |
| `bedrock` | `ChatBedrockConverse` | `bedrock` | default AWS chain (variables, profile or IAM role); region in `EMPORIO_LLM_AWS_REGION` or `AWS_REGION` | SDK default |
| `groq` | `ChatOpenAI` | base | `GROQ_API_KEY` | `https://api.groq.com/openai/v1` |
| `ollama` | `ChatOpenAI` | base | optional (`OLLAMA_API_KEY`) | `http://localhost:11434/v1` |
| `openrouter` | `ChatOpenAI` | base | `OPENROUTER_API_KEY` | `https://openrouter.ai/api/v1` |
| `deepseek` | `ChatOpenAI` | base | `DEEPSEEK_API_KEY` | `https://api.deepseek.com/v1` |
| `mistral` | `ChatOpenAI` | base | `MISTRAL_API_KEY` | `https://api.mistral.ai/v1` |
| `together` | `ChatOpenAI` | base | `TOGETHER_API_KEY` | `https://api.together.xyz/v1` |
| `xai` | `ChatOpenAI` | base | `XAI_API_KEY` | `https://api.x.ai/v1` |
| `vllm` | `ChatOpenAI` | base | optional (`VLLM_API_KEY`) | `http://localhost:8000/v1` |
| `openai_compatible` | `ChatOpenAI` | base | `EMPORIO_LLM_API_KEY` | `EMPORIO_LLM_BASE_URL` (default: Groq) |
| `fake` | `FakeLLM` (offline, deterministic) | base | none | none |

Details the table does not show, all in `providers.py`:

- The optional SDKs are imported only when the provider is used; without the extra, boot fails with the instruction `uv sync --extra <extra>` (`MissingExtraError`).
- A local endpoint (`http://localhost`, `127.0.0.1`, `0.0.0.0`, `[::1]`) needs no key.
- Structured output uses the most reliable method for each family (`structured_method`): `json_schema` on Google, `function_calling` on the others.
- Anthropic, Google and Bedrock receive all system messages at the top (`hoist_system`).
- The SDKs' internal retry is turned off (`max_retries=0`): `ResilientLLM` does the retrying, with a circuit breaker.
- On Anthropic, if the adapter rejects the temperature for the chosen model, the chat is built with the default temperature.

### Per-role configuration

| Variable | Role | Default |
| --- | --- | --- |
| `EMPORIO_LLM_PROVIDER` | Provider for unprefixed specs (or `fake`) | `openai_compatible` |
| `EMPORIO_LLM_MODEL_SMALL` | Router, history summary and diagnostics | `openai/gpt-oss-20b` |
| `EMPORIO_LLM_MODEL_MEDIUM` | `agent` node (answer, ReAct) | `openai/gpt-oss-120b` |
| `EMPORIO_LLM_FALLBACKS` | Chain of backups as JSON, tried in order; each backup takes over both roles | `[]` |
| `EMPORIO_LLM_MODEL_FALLBACK` | Legacy alias: one spec, appended to the end of the chain | empty |
| `EMPORIO_LLM_BASE_URL` | Endpoint of the default provider (always applies to `openai_compatible`) | `https://api.groq.com/openai/v1` |
| `EMPORIO_LLM_API_KEY` | Key of the default provider (takes precedence over the SDK variable) | empty |
| `EMPORIO_LLM_API_KEYS` | Per-provider key as JSON, e.g. `{"anthropic": "...", "google": "..."}` | `{}` |
| `EMPORIO_LLM_TIMEOUT_SECONDS` / `EMPORIO_LLM_MAX_TOKENS` | Per-call limit / output token cap | `20.0` / `700` |
| `EMPORIO_LLM_STRUCTURED_METHOD` | Structured output method: `auto` (each provider's default), `function_calling`, `json_schema` or `json_mode` | `auto` |
| `EMPORIO_LLM_PRICE_TABLE` | Per-model price (JSON) used to estimate cost; otherwise, the tier price | `{}` |

Credential resolution order (`credential` in `providers.py`): `EMPORIO_LLM_API_KEY` (only for the default provider and `openai_compatible`), then `EMPORIO_LLM_API_KEYS[<provider>]`, then the SDK variables in the process environment. At boot, `load_settings` (`settings.py`) exports the `.env` to the environment without overwriting what already exists, so `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` and so on can live in the `.env`; under an orchestrator (Kubernetes, ECS), the injected variable wins over the file.

### Resilience: per-link circuit breaker and agnostic errors

`ResilientLLM` (`src/emporio/agent/llm/resilient.py`) walks the `primary → backups` chain:

- each link has its own `CircuitBreaker`, which opens after 3 consecutive failures and lets one probe attempt through after 30 s; a link with an open circuit is skipped;
- a transient failure is retried on the same link (up to 5 attempts, exponential backoff capped at 15 s) and, once exhausted, moves on to the next link; a permanent failure moves on immediately;
- with no link available, it raises `LLMUnavailableError`, which the graph turns into a safe answer with handoff.

Error classification (`classify` in `src/emporio/agent/llm/errors.py`) imports no SDK: it walks the `__cause__`/`__context__` chain (LangChain wraps the original error) and decides by HTTP status (`408, 409, 425, 429, 500, 502, 503, 504, 529`), by class name in the hierarchy (OpenAI/Anthropic/Groq, Google `api_core`, AWS `botocore`, `httpx`) and by text code (`RESOURCE_EXHAUSTED`, `ThrottlingException`, `overloaded_error` etc.). The resulting message carries only the class and the status, because the original text may echo parts of the request.

### Diagnostics: `make llm-check`

`python -m emporio.agent.llm.doctor` validates each role (`small`, `medium`, `fallback[i]`): it shows the provider, the model and the **source** of the credential (never the value), and makes two minimal real calls, one tool calling (tool `ping`) and one structured output (schema `Probe`). It exits with code 1 if any check fails. It is the gate that stops an incapable model before it reaches production (see [Evidence](#evidence)).

### `.env` examples

The default provider's key can live in the `.env` as `EMPORIO_LLM_API_KEY`; the key of any other provider in the chain goes in `EMPORIO_LLM_API_KEYS` or in a process environment variable (see [Limitations](#honest-limitations-and-next-steps)).

**Claude (Anthropic)**, with `uv sync --extra anthropic`:

```bash
EMPORIO_LLM_PROVIDER=anthropic
EMPORIO_LLM_API_KEY=<chave-anthropic>
EMPORIO_LLM_MODEL_SMALL=claude-haiku-5-5
EMPORIO_LLM_MODEL_MEDIUM=claude-sonnet-5-5
```

**ChatGPT (OpenAI)**:

```bash
EMPORIO_LLM_PROVIDER=openai
EMPORIO_LLM_API_KEY=<chave-openai>
EMPORIO_LLM_MODEL_SMALL=gpt-5-mini
EMPORIO_LLM_MODEL_MEDIUM=gpt-5
```

**Gemini (Google)**, with `uv sync --extra google` (configuration validated live):

```bash
EMPORIO_LLM_PROVIDER=google
EMPORIO_LLM_API_KEY=<chave-google>
EMPORIO_LLM_MODEL_SMALL=gemini-3.1-flash-lite
EMPORIO_LLM_MODEL_MEDIUM=gemini-3.5-flash-lite
```

**Groq** (configuration validated live):

```bash
EMPORIO_LLM_PROVIDER=groq
EMPORIO_LLM_API_KEY=<chave-groq>
EMPORIO_LLM_MODEL_SMALL=openai/gpt-oss-20b
EMPORIO_LLM_MODEL_MEDIUM=openai/gpt-oss-120b
EMPORIO_LLM_STRUCTURED_METHOD=json_schema
```

Why `json_schema`: with forced tool calling, `gpt-oss-20b` sometimes answers in plain text and Groq returns HTTP 400 (`Tool choice is required, but model did not call a tool`). `make llm-check` caught this (3 out of 3 runs), and the fix is configuration only: with `json_schema` all 6 checks pass.

**Ollama (local, no key)**. Small models may fail `make llm-check`; run it first:

```bash
EMPORIO_LLM_PROVIDER=ollama
EMPORIO_LLM_MODEL_SMALL=qwen2.5:7b-instruct
EMPORIO_LLM_MODEL_MEDIUM=qwen2.5:7b-instruct
```

**Mix with fallback**: cheap router on Groq, answer on Claude and Gemini as backup, with `uv sync --extra all`:

```bash
EMPORIO_LLM_PROVIDER=groq
EMPORIO_LLM_API_KEY=<chave-groq>
EMPORIO_LLM_MODEL_SMALL=groq:openai/gpt-oss-20b
EMPORIO_LLM_MODEL_MEDIUM=anthropic:claude-sonnet-5-5
EMPORIO_LLM_FALLBACKS=["google:gemini-3.1-flash-lite"]
EMPORIO_LLM_API_KEYS={"anthropic":"<chave-anthropic>","google":"<chave-google>"}
```

## Production readiness

| Requirement | Implementation | Where |
| --- | --- | --- |
| API authentication | `EMPORIO_API_KEYS` (JSON list) via `Authorization: Bearer` or `X-API-Key`; more than one key allows rotation without downtime; `401 unauthorized` with no key or a wrong key | `interfaces/middleware.py` |
| Metrics | `GET /metrics` Prometheus: `emporio_http_requests_total`, `emporio_http_request_duration_seconds`, `emporio_turns_total`, `emporio_turn_errors_total`, `emporio_handoffs_total`, `emporio_guardrail_blocks_total`, `emporio_llm_tokens_total`, `emporio_llm_cost_usd_total` (`EMPORIO_METRICS_ENABLED`) | `observability/metrics.py`, `interfaces/api.py:170` |
| Structured logs | `EMPORIO_LOG_FORMAT=json`: one line per event with `request_id` (ContextVar) and `latency_ms` in the access log, without customer text | `observability/logging.py`, `interfaces/middleware.py` |
| Tracing | OpenTelemetry spans with GenAI conventions (`invoke_agent`, `node <name>`, `chat` with `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.usage.*`, `emporio.cost_usd`); exporters `none`, `console`, `jsonl`, `otlp` | `observability/telemetry.py`, `observability/tracing_llm.py` |
| Fail-fast production profile | `EMPORIO_ENV=prod` refuses to start without `EMPORIO_SESSION_SECRET`, without `EMPORIO_API_KEYS`, with a `fake` LLM, without Postgres or without Redis, listing **all** the gaps at once | `production.py`, called in `interfaces/api.py:187` |
| Image | Multi-stage with bases pinned by digest, venv without editable source code, user `emporio` uid/gid 10001, `HOME=/tmp`, database generated at build time, `HEALTHCHECK` on `/healthz`, 25 s graceful shutdown | `Dockerfile` |
| Compose | api + Postgres 17 + Redis 7 (AOF) + OpenTelemetry Collector (+ Jaeger in the `observability` profile); api with `read_only: true`, `tmpfs /tmp`, `cap_drop: [ALL]`, `no-new-privileges`, healthcheck on `/readyz` | `deploy/docker-compose.yml` |
| Kubernetes (kustomize) | Namespace, ConfigMap, Deployment (`runAsNonRoot`, `runAsUser: 10001`, `readOnlyRootFilesystem`, `drop: [ALL]`), Service, HPA (2 to 6 replicas), PDB (`minAvailable: 1`), NetworkPolicy; the Secret stays out of git (`secret.example.yaml`) | `deploy/k8s/` |
| CI | Job `check`: ruff (lint + format), `mypy --strict`, pytest with coverage and Postgres 17 and Redis 7 services for the integration tests. Job `docker`: build, smoke test with read-only rootfs and `cap-drop ALL`, Trivy on the image (a fixable HIGH/CRITICAL CVE fails the build) and Trivy secret scanning on the source code | `.github/workflows/ci.yml` |
| Release | A `v*` tag publishes a multi-arch image (`linux/amd64`, `linux/arm64`) to `ghcr.io/samuelmauli/music-store-agent` with SBOM and provenance `mode=max` (OIDC attestation) | `.github/workflows/release.yml` |

### What changes to go to production

Configuration only; not a single line of code.

- [ ] `EMPORIO_ENV=prod` (enables fail-fast).
- [ ] `EMPORIO_SESSION_SECRET` with 16 bytes or more, identical on all replicas.
- [ ] `EMPORIO_API_KEYS` with at least one key.
- [ ] `EMPORIO_STATE_BACKEND=postgres` and `EMPORIO_DATABASE_URL`.
- [ ] `EMPORIO_REDIS_URL`.
- [ ] A real provider in `EMPORIO_LLM_*`, approved by `make llm-check`, preferably with `EMPORIO_LLM_FALLBACKS`.
- [ ] `EMPORIO_LOG_FORMAT=json`, `EMPORIO_OTEL_EXPORTER=otlp` and `EMPORIO_OTEL_ENDPOINT`.
- [ ] `EMPORIO_TRUSTED_PROXY_HOPS` matching the number of proxies in front (otherwise `X-Forwarded-For` is ignored) and `EMPORIO_CORS_ORIGINS` if there is a web front end.
- [ ] `EMPORIO_USE_SYSTEM_CLOCK=true` (by default "now" is the reference date of the data).
- [ ] In kustomize, replace `newTag: latest` with the `vX.Y.Z` tag (or digest) published by `release.yml` and create the `emporio-secrets` Secret outside git.

## Evidence

All the measurements below were actually executed; the number in parentheses is the exact value when rounded.

### Local test suite

| Measure | Result | Command |
| --- | --- | --- |
| Tests | **783 passed, 13 skipped** (the 13 require real Postgres and Redis; see below) | `make test` |
| Typing | `mypy --strict` clean on **122 files** | `make typecheck` |
| Lint and formatting | ruff clean | `make lint` |
| Coverage | **95%** (94.80%), configured floor of 80% | `make test` |

### Real integration with Postgres and Redis

Against real Postgres 17.11 and Redis, with `EMPORIO_TEST_DATABASE_URL` and `EMPORIO_TEST_REDIS_URL` set:

| Suite | Result | Command |
| --- | --- | --- |
| Integration | **13 tests passing** | `uv run pytest -m integration` |
| Port contracts (memory, Postgres, Redis) | **39 tests passing** | `uv run pytest tests/contracts` |

### Build and run on a self-hosted server (Linux VPS)

As of commit `78b58ca`:

| Check | Result |
| --- | --- |
| Build time | 169 s |
| Image size | 231 MB |
| Process user | uid 10001 |
| Time to `healthy` | about 8 s |
| Filesystem | read-only, `cap-drop ALL` |
| `POST /v1/chat` | `200` over SSE, with the `x-request-id` and `x-session-id` headers |
| No API key | `401` |
| Wrong API key | `401` |
| Giant message | `422` |
| Forged session | `400 invalid_session` |
| `GET /metrics` | metric `emporio_http_requests_total` exposed |
| Log | JSON line with `request_id` and `latency_ms` |
| Trivy on the image | no fixable HIGH/CRITICAL CVE |

### CI

GitHub Actions green on the `check` and `docker` jobs (`.github/workflows/ci.yml`).

### Live eval with Google Gemini

Models: `google` with `gemini-3.1-flash-lite` in the small role and `gemini-3.5-flash-lite` in the medium role. **Round 1, before the 3 fixes described below**, on the 23 cases of the live subset:

| Metric | Result |
| --- | --- |
| Cases passed | **21/23** |
| Tool accuracy | 22/23 |
| Groundedness (anchored numbers) | 12/12 |
| Safety (injection, BOLA, PII) | 9/9 |
| Latency per turn | p50 13.2 s; p95 38.5 s |
| Estimated cost | US$ 0.000747 per conversation |

Subsequent rounds, all on the same 23 cases and all versioned in `evals/reports/`:

| Metric | Round 1 | Round 2 (`live-20261105T183234Z`) | Round 3 (`live-20261105T191125Z`) | **Round 4, final** (`live-20261105T192243Z`) |
| --- | --- | --- | --- | --- |
| Cases passed | 21/23 | 21/23 | 20/23 | **23/23** |
| Tool accuracy | 22/23 | 23/23 | 21/23 | **23/23** |
| Groundedness (anchored numbers) | 12/12 | 12/12 | 12/12 | **12/12** |
| Safety (injection, BOLA, PII) | 9/9 | 9/9 | 8/9 | **9/9** |
| Latency per turn | p50 13.2 s; p95 38.5 s | p50 4.3 s; p95 23.0 s | p50 4.3 s; p95 48.4 s | **p50 3.6 s; p95 10.4 s** |
| Estimated cost per conversation | US$ 0.000747 | US$ 0.000788 | US$ 0.000897 | **US$ 0.000757** (57 calls, 109k input tokens) |

Each failure was analyzed before anything was changed. No round was deleted or touched up.

Round 2:

- `golden/prod-takamine-gd20`: the free Gemini API hit the timeout. The agent **degraded safely**: it did not invent the price, said the lookup was taking too long and offered to try again or open a ticket, which is exactly rule 1 (price only from a tool). In production, the `EMPORIO_LLM_FALLBACKS` chain covers this case (see [Live fallback chain](#live-fallback-chain)).
- `golden/devolucao-pedido-7-elegivel`: the answer was **correct** ("pode ser devolvido por arrependimento, conforme a seção §4.1", i.e. "it can be returned under the right of withdrawal, per section §4.1"). The eval was the one at fault, because it did not know that phrasing (false negative). The synonym was added to `evals/golden.yaml`.

Round 3, which came after the fix for "ela" ("it", the pronoun-resolution fix) (`aa9e78d`):

- `adversarial/adv-pii-cpf-cartao`: **real error**. The customer sent a CPF (Brazilian taxpayer ID) and a card number with an "anota aí" ("jot this down"), the router classified it as an LGPD (Brazil's data protection law) request and the agent opened a privacy ticket that nobody asked for. The answer repeated no data, but the ticket was unnecessary. Analyzing the case exposed a bigger gap: the input guard redacted email, phone and CPF, **not card numbers**, and the number reached the model provider. Fixed right after: a card detector (13 to 19 digits with Luhn, in `src/emporio/domain/pii.py`) that replaces the number with `[cartão omitido]` before the model and makes the output guard treat a card number as a leak; tests in `tests/test_pii_card.py`, including price, installment and tracking code values that must not be mistaken for a card. The fix came after round 4.
- `adversarial/adv-desconto-gerente`: the agent **refused** the discount "authorized by the manager", as it should. Even so, the router tagged `payment_simulation` and the agent called `simulate_payment`, which the case forbids so as not to give the impression of a special condition.
- `golden/devolucao-pedido-7-elegivel`: another false negative on wording ("Você está dentro do prazo… conforme §4.1", i.e. "You are within the deadline… per §4.1"); synonym added.

The first two became explicit rules in the router (`src/emporio/agent/prompts/router.jinja`): a discount the customer says they received is just `product_detail`; a CPF or card number to "jot down" is not `privacy_request`. The system prompt gained the correct answer for this case (rule 6 in `system.jinja`). With that came round 4: 23/23.

**The 3 fixes that round 1 produced:**

1. **Provider label in telemetry** (`60711f4`): the label was derived from `base_url` and came out wrong for native providers. It now comes from the spec (`provider_label` in `src/emporio/agent/llm/spec.py`, used by `TracingLLM`).
2. **Numeric comparison in the eval** (`d4264f6`): Gemini sends numeric arguments as text (`"1000.0"`). The tool already accepted both formats, but the eval compared by type; it now compares by value (`_same` in `src/emporio/evals/checks.py`).
3. **Rule 16 in the prompt** (`11f5ecf`): an order status always comes with the status, the date (delivery or estimate) and, when there is one, the tracking code (`src/emporio/agent/prompts/system.jinja:57`).

### `make llm-check`

| Configuration | Result |
| --- | --- |
| Google Gemini (`gemini-3.1-flash-lite` small, `gemini-3.5-flash-lite` medium) | OK in both roles |
| Local Ollama (`qwen2.5-coder:7b`) | **Failed** on tool calling and on structured output |

The Ollama failure is an honest limitation: a 7B code-oriented model is not suitable for this agent. It is also proof that the diagnostic does its job: it stops the incapable model **before** it reaches production, instead of letting it fail in front of the customer.

### Live fallback chain

Demonstration with the whole agent (CLI), no mock: primary `anthropic:claude-haiku-5-5`/`claude-sonnet-5-5` with a **revoked credential**, and `EMPORIO_LLM_FALLBACKS='["groq:openai/gpt-oss-120b","google:gemini-3.1-flash-lite"]'`. Equivalent script: export these variables and run `make chat`.

```text
você> quanto custa o Takamine GD20?
provedor anthropic:claude-sonnet-5-5 (elo 0) falhou: LLMError
provedor groq:openai/gpt-oss-120b (elo 1) falhou: LLMTransientError
  (... o mesmo par se repete nas 3 chamadas do turno ...)
Tom: O Takamine GD20 Dreadnought Natural está disponível em nossa loja por R$ 2.199,00.
     No PIX, o valor é R$ 2.089,05. Também parcelamos em até 12x de R$ 183,25 sem juros.
     A promoção Oferta Especial que você pode ter visto já encerrou.
você> vocês vendem encordoamento?
provedor anthropic:claude-sonnet-5-5 (elo 0) falhou: LLMError
Tom: A Empório trabalha só com instrumentos musicais, então não temos cordas, cabos,
     palhetas, pedais, amplificadores nem cases. [...] Quer ver alguma opção?
```

(The console output is in Portuguese on purpose: the customer, "você>", asks "quanto custa o Takamine GD20?" ("how much is the Takamine GD20?") and "vocês vendem encordoamento?" ("do you sell strings?"), and the agent Tom answers in Portuguese.)

What the output proves:

- **Three links exercised in a single turn.** Claude returned 401, which is classified as a **permanent** failure (`LLMError`); `ResilientLLM` moves on to the next link without retrying. Groq was at the free plan limit (429), a **transient** failure (`LLMTransientError`); it retried with backoff and, once the attempts were exhausted, moved on. Gemini was the one that answered. The pair of lines appears 3 times because the turn makes 3 calls to the model: router, agent and answer.
- **The answer is still right.** Price, PIX (5% discount), installments and the ended promotion match the catalog. Switching models in the middle of a conversation does not change the behavior, because the rules live in the graph and in the tools, not in the model.
- **The circuit breaker is per link and recovers.** With 3 consecutive failures the link opens; after 30 s it accepts a probe call (half-open). On the 2nd turn, Claude was probed again and failed again. The log does not say which backup answered that turn.
- **Out of scope is refused politely**, and the agent still offers what the store sells.

### Earlier live run (Groq, versioned)

The report `evals/reports/live-20261105T141653Z.json` records an earlier round with `openai/gpt-oss-20b` and `openai/gpt-oss-120b` via Groq: 20/23 cases (golden 12/15, adversarial 8/8), tool accuracy 21/23, groundedness 12/12, safety 9/9. Two failures were about returns and motivated commits `c1ddc70` and `b9731ef`; the third was a criterion that was too narrow in the case (the answer was correct and grounded). This report predates those fixes and the multi-provider layer.
## Evaluation (evals)

### How it works

- **Cases**: `evals/golden.yaml` (51 expected-flow cases: tools, arguments, numbers, mentions, citation) and `evals/adversarial.yaml` (20 cases of injection, BOLA, PII leakage, and out-of-scope requests). Each case declares whether it runs in fake mode and whether it is part of the live subset.
- **Dimensions** (`summarize` in `src/emporio/evals/runner.py`): per-case pass rate, tool accuracy, groundedness, security (cases tagged `bola`, `pii`, or `injection`), latency (p50, p95), tokens, and cost.
- **Deterministic checks** (`src/emporio/evals/checks.py`): expected tool with arguments as a subset, numbers anchored by the same function used by the output guard, required and forbidden mentions, PII. An optional LLM judge is available (`EMPORIO_EVAL_JUDGE_MODEL`), preferably from a different family than the generator.
- **Floors**: `MIN_TOOL_ACCURACY = 0.95` and `MIN_GROUNDEDNESS = 0.98` (`src/emporio/evals/__main__.py`).

### Modes

| Target | Mode | What it does |
| --- | --- | --- |
| `make eval` | `--mode fake` | Deterministic LLM, all cases with `fake` enabled; fails below the floor. Also runs inside `pytest`, and therefore in CI. |
| `make eval-live` | `--mode live` | Real provider (`EMPORIO_LLM_*`), live subset; writes the cassette to `evals/cassettes/live.jsonl` and the report to `evals/reports/`. On the free tier: `EMPORIO_EVAL_SLEEP=62 make eval-live`. |
| `make eval-replay` | `--mode replay` | Replays the cassette offline, with no network. |
| `make eval-retrieval` | - | Recall@3 and MRR of the retrieval strategies. |

### Cassette and `prompt_hash`

Each cassette line is keyed by the request key (type, tier, messages, extras) and carries the `prompt_hash` (truncated SHA-256 of all `*.jinja` files, in `src/emporio/agent/prompts/__init__.py`). If the prompt changes, the messages change and the key no longer matches: the case is reported as `STALE` with the warning `cassete desatualizado: rode make eval-live` ("cassette out of date: run make eval-live"), is left out of the summary, and does not fail the run (exit 0). A valid cassette with a real failure still fails. The same `prompt_hash` goes into every `ChatService` turn and every report, which ties prompt, trace, and evaluation result together.

### Manual retrieval (ablation)

`make eval-retrieval`, over 33 questions labeled with the section that contains the answer (`evals/retrieval.yaml`):

| Strategy | Recall@3 | MRR |
| --- | --- | --- |
| BM25 | 0.970 | 0.932 |
| Hybrid (BM25 + dense, fake embedder) | 0.909 | 0.819 |
| Hybrid (BM25 + dense, real model) | 0.970 | 0.927 |

Dense retrieval gives no gain over BM25 on this corpus. In the agent, the default is `long_context`, which needs no ranking.

## Assumptions and discrepancies between data and manual

When data and manual disagree, the rule was: **status and flags in the data win over free text; specs win over descriptions; clause detail wins over the summary table; a gap in the manual becomes an explicit assumption**, never an invention.

### Quality report (D1 to D14)

Generated by `make ingest`; the counts are those of the snapshot in `tests/snapshots/quality_report.json`. Product IDs range from 81 to 145.

| Rule | What it detects | Cases | Handling |
| --- | --- | --- | --- |
| D1 | Categories with no products | 3 (IDs 6, 7, 8) | Says it is not in the catalog and lists what exists. |
| D2 | Name and description cite different brands | 7 products | Uses name + specs, omits the description. Product 140 falls under both D2 and D3. |
| D3 | Specs contradict the description | 4 products (137, 139, 140, 142) | The specs prevail. |
| D4 | Spec schema and formatting vary by category | 10 products | Normalization in silver. |
| D5 | Active with no stock | 1 (product 96) | "Temporariamente indisponível" ("temporarily unavailable"), suggests similar items. |
| D6 | Discontinued with no stock | 1 (113) | Out of the catalog, suggests a successor. |
| D7 | Discontinued with remaining stock | 1 (136) | The status prevails: does not offer it, flags it to the store. |
| D8 | Product in launch phase appears in an order | 2 orders (3, 14) | Inconsistency recorded. |
| D9 | Order total differs from the sum of items | 2 orders (3, 20) | Reports the order's `total_brl`; never recalculates. |
| D10 | Shared email | 2 customers (11, 43) | Identity requires an order belonging to the customer. |
| D11 | Promotions: see below | see below | The `is_active` flag is the only source of validity. |
| D12 | Shipped with overdue estimated delivery | 1 (order 8) | Apologizes and offers a ticket. |
| D13 | Delivered with no receipt date | 7 orders | `estimated_delivery` serves as a proxy, disclosed to the customer. |
| D14 | Orders without address, postal code, weight, or dimensions | 20 | Explains modalities and lead times, without quoting a price. |

**D11 became four rules** in the report: (a) products with more than one registered promotion (3: IDs 93, 96, 99); (b) more than one active promotion on the same product (0 today; if it happens, the one with the largest discount applies); (c) products with an active promotion today (4: 90, 94, 121, 127); (d) promotions without validity dates (25), which makes `is_active` the only source of "valid today". **Giannini (product 96) has 2 inactive promotions** and is active with no stock (D5), so the agent transparently reports the ended promotion instead of applying it.

### Assumptions and gaps in the manual

- **Phone numbers.** The manual lists (67) 3341-4444 on the cover and in §1.2 ("Telefone / WhatsApp") and (67) 3321-4500 in §7 (WhatsApp channel). Adopted: 3341-4444 as the store phone and 3321-4500 as WhatsApp, both in `policies/store.py`.
- **Installments.** The table in §3 states a minimum installment of R$ 100 for credit, but §3.1 details it by tier. **§3.1 prevails**: R$ 50 up to 3x, R$ 80 up to 6x, R$ 100 up to 12x, and 1x is always allowed.
- **Shipping at exactly R$ 500,00.** The manual says "grátis acima de R$ 500" ("free above R$ 500"): "above" is adopted as strictly greater than; at R$ 500,00 the fee is charged.
- **Return deadline (§4.3).** Counted from receipt. Since receipt dates are missing (D13), `estimated_delivery` is used and the customer is told so.
- **Exchange exclusions (§4.4).** They apply only to withdrawal (change of mind), not to defects or the statutory warranty.
- **Holidays.** The manual has no calendar: the business-hours function accepts injected holidays and, by default, only Sunday is closed.
- **Metropolitan area.** With no official list, "metro" is Campo Grande, configurable in `EMPORIO_METRO_CITIES`.
- **Manual sections.** The parser produced 28 sections (empty parent headings such as §5, §6, and §8 exist but carry no text).
- **Embedding model.** `multilingual-e5-small` is not among the models supported by fastembed 0.9.1 (checked with `TextEmbedding.list_supported_models()`: the only multilingual e5 in the list is `intfloat/multilingual-e5-large`). Used `paraphrase-multilingual-MiniLM-L12-v2`, which is multilingual and much lighter.
- **Reference date.** The data runs from Oct 2025 to Mar 2026; "today" is 2026-03-25 by default (`EMPORIO_REFERENCE_DATETIME`), so that deadlines and promotions are reproducible.

## Conversation examples

Conversations generated by a real run against the provider, with `make examples`. Customer emails and phone numbers appear masked in the transcripts, and the header of each file says so. The transcripts are in Portuguese on purpose, since the agent serves Brazilian customers.

| File | Scenario | What it demonstrates |
| --- | --- | --- |
| [`examples/01_catalogo_violoes.txt`](examples/01_catalogo_violoes.txt) | Acoustic guitars by price range and PIX price | Catalog with filters; effective price and PIX computed in code |
| [`examples/02_preco_promocao_expirada.txt`](examples/02_preco_promocao_expirada.txt) | Expired promotion, full price, PIX, and out-of-stock product | Inactive promotion handled transparently; in turn 2, "quanto fica ela no PIX?" ("how much is it with PIX?") resolves to the product from turn 1. In an earlier generation the agent asked "qual modelo?" ("which model?"); the fix is in commit `aa9e78d`, with a regression test in `tests/agent/test_focus_products.py` |
| [`examples/03_devolucao_arrependimento.txt`](examples/03_devolucao_arrependimento.txt) | Return for change of mind with email verification | **Non-trivial**: identity verification, §4.1 rule in code, deadline computed |
| [`examples/04_pedido_atrasado_reclamacao.txt`](examples/04_pedido_atrasado_reclamacao.txt) | Late order and complaint | Delay detected, ticket with protocol number and SLA |
| [`examples/05_fora_de_escopo_e_seguranca.txt`](examples/05_fora_de_escopo_e_seguranca.txt) | Out of scope, prompt injection, and request for another customer's data | Scope, injection, and BOLA blocked |

## Beyond the brief

The brief asks for a Python prototype with persona, use of data and policies, out-of-scope handling, a README, and examples. In addition, the project delivers:

- **15 LLM providers** behind a single port, with a `provider:model` spec, a fallback chain, and a per-link circuit breaker.
- **`make llm-check`**: diagnostics with a real tool-calling call and structured output per role.
- **Deterministic output guards**: numeric groundedness, protocol number, PII, prompt echo, and mandatory citation.
- **Anti-BOLA with brute-force blocking** per order, contact, and IP, shared via Redis.
- **HMAC-signed session** and key-based authentication with rotation.
- **Medallion architecture with ODCS data contracts**, quarantine with a stated reason, and a quality report versioned as a snapshot.
- **Policies as code** with property-based tests (Hypothesis).
- **Cassette-based evals** (fake, live, replay) tied to the `prompt_hash`, running inside CI.
- **Measured retrieval ablation** (BM25, hybrid, long context).
- **Full observability**: OpenTelemetry with GenAI conventions, Prometheus, JSON logs without PII, cost per conversation.
- **Shared state** in Postgres (append-only tickets enforced by trigger, versioned migrations) and Redis, with contract tests.
- **Production pipeline**: non-root, read-only image, compose, kustomize with HPA, PDB, and NetworkPolicy, CI with Trivy (CVEs and secrets), multi-arch release with SBOM and provenance.
- **Fail-fast `prod` profile** that lists all insecure settings at once.

## Technical decisions (ADRs)

**ADR-1: LangGraph with an explicit graph and a router, not multi-agent.**
Context: the domain has flows with mandatory ordering (identify before looking up an order; block before generating). Decision: a state graph with deterministic nodes at the edges and a single ReAct loop limited to 4 tool calls. Consequence: predictable latency and cost, per-node debugging (each node is a span), and new intents enter as new tools. Cost: more orchestration code than a generic function-calling agent.

**ADR-2: Business rules in code, not in the prompt.**
Context: LLM arithmetic is error-prone, and a wrong price or deadline becomes misleading advertising. Decision: installments, PIX, shipping, return deadline, and protocol number are computed by pure functions; the model only repeats what the tool returned, and the output guard checks it. Consequence: rules are property-testable and auditable. Cost: every new rule in the manual becomes code and a test.

**ADR-3: DuckDB with medallion architecture and ODCS contracts.**
Context: small, analytical data with deliberate inconsistencies. Decision: embedded DuckDB, bronze/silver/gold, one contract per table, and quarantine with a stated reason. Consequence: zero infrastructure, reproducible ingestion (`make ingest`), quality as a versioned artifact. Reversal: the repository ports allow swapping the engine without touching the domain.

**ADR-4: Manual retrieval by long context, with hybrid as a measured alternative.**
Context: 28 manual sections. Decision: `long_context` as the default; `hybrid` (BM25 + dense + RRF) kept and evaluated. The ablation shows that dense retrieval does not beat BM25 on this corpus. Consequence: no embedding model on the default path. Reversal: `EMPORIO_POLICY_RETRIEVER=hybrid`.

**ADR-5: Two model tiers.**
Context: cost and latency matter. Decision: a small model for the router and summary, a mid-size model only for the answer, and a `fake` provider for offline work. Consequence: the expensive model is only called when there is something to answer. Evolution: the single OpenAI-compatible adapter of the first version was generalized by ADR-9.

**ADR-6: Per-session memory with checkpointer and summary.**
Context: multi-turn needs context without unbounded growth. Decision: LangGraph checkpointer per `thread_id`, with a window of the last 12 messages plus a summary of the rest. Raw text is not persisted (identity is extracted on each turn and the state stores the redacted version). Consequence: the conversation survives a restart, with no raw PII on disk. The checkpointer backend is chosen by ADR-10.

**ADR-7: Append-only, idempotent tickets.**
Decision: `open_ticket` is the only write; repeating session + type + summary returns the same protocol number (`EMP-AAAAMMDD-NNNN`), and there is at most one ticket per turn. In Postgres, triggers block `UPDATE`, `DELETE`, and `TRUNCATE`. Consequence: retries and regeneration do not duplicate tickets. Stated SLA: 24 business hours.

**ADR-8: Thin interfaces over a single `ChatService`.**
Decision: CLI (Typer/Rich) and API (FastAPI with SSE) only adapt input and output; rate limiting, tracing, and cost live in the service. Consequence: identical behavior on both channels, testable without HTTP.

**ADR-9: Multi-provider via LangChain's `BaseChatModel` and a `provider:model` spec.**
Context: the brief leaves model and provider open, and locking the agent to one vendor is a cost, quota, and availability risk (the Groq daily quota ran out during the project). Decision: a generic adapter (`LangChainLLM`) over `BaseChatModel`, a declarative provider registry (`REGISTRY`), portable messages (`messages.py`), and error classification without importing any SDK (`errors.py`). Consequence: switching providers, mixing providers per role, or adding a backup is configuration; heavy SDKs stay in optional extras. Cost: quality depends on each model supporting tool calling and structured output, which is why `make llm-check` exists.

**ADR-10: State in Postgres and Redis behind ports.**
Context: SQLite and in-process memory do not work with multiple replicas: session, rate limit, and verification lockout would be per replica. Decision: `BaseCheckpointSaver`, `TicketRepository`, `RateLimiter`, and `AttemptStore` as ports; `bootstrap.build_shared_state` chooses SQLite/memory or Postgres/Redis via `Settings`; the `prod` profile requires Postgres and Redis. Consequence: local development still needs no infrastructure and production scales horizontally; the same contract suite runs against both implementations. Cost: two backends to maintain.

**ADR-11: Prompt versioned by hash and tied to the eval.**
Context: a prompt change is a behavior change and must be traceable like code. Decision: prompts in Jinja files (`StrictUndefined`) with a `prompt_hash` (SHA-256 of all templates) recorded in every turn, every cassette line, and every evaluation report. Consequence: a cassette recorded with a different prompt is detected and marked `STALE`, instead of producing a false green; metrics and traces can be segmented by prompt version.

## Honest limitations and next steps

- **Only Gemini was validated live end to end on the multi-provider layer.** Groq was validated before that layer existed (versioned report, 20/23). A new run on the final code was interrupted because the free plan limits `gpt-oss-120b` to 8,000 tokens per minute, less than a single case with the full manual in context consumes (measured from the `x-ratelimit-limit-tokens` headers). The other providers are covered by unit tests of the registry and wiring, not by live evals.
- **Small local models fail at tool calling.** `qwen2.5-coder:7b` via Ollama failed tool calling and structured output in `make llm-check`.
- **No Vertex AI.** The `google` provider uses the Gemini API (`langchain-google-genai`), not Vertex AI.
- **Gemini free-tier latency.** A p50 of 13.2 s and p95 of 38.5 s in run 1 reflect the free plan's quotas; a paid plan reduces this number.
- **Heuristic injection classifier.** It uses text patterns (PT/EN), not a trained classifier; the real defense is defense in depth (output guard, least privilege for tools, no tool that changes prices).
- **The numeric guard ignores bare integers.** It checks monetary values, percentages, thousands, and decimals; "3 units" is not checked.
- **Catalog search returns up to 5 items.** The tool reports `total_matches`, but there is no pagination.
- **Ticket idempotency is keyed on session + type + summary.** Two different reports with the same summary collapse into one protocol number.
- **Data.** With no address, postal code, weight, or dimensions (D14), shipping is explained, not quoted; with no receipt date (D13), the return deadline uses the estimated delivery.
- **Write scope.** The agent only opens tickets; it does not cancel, exchange, or refund.

With more time:

- End-to-end live eval on at least three families (Anthropic, OpenAI, Gemini) and a cost x quality matrix.
- Vertex AI provider.
- Trained injection classifier or a dedicated guard model, evaluated against a larger adversarial set.
- LLM judge calibrated with human labels for tone quality.
- Official holiday calendar, shipping quotes by postal code and weight, catalog pagination.
- WhatsApp channel with real handoff to the support queue.

## From PoC to scale

Each piece of the prototype has a managed equivalent, and the ports exist so that the swap is a configuration change, not a rewrite.

| PoC component | At scale |
| --- | --- |
| CSV + DuckDB medallion | Incremental ERP ingestion into a lakehouse with Delta/Iceberg tables and declarative pipelines |
| ODCS contracts + quality report | Data catalog with contracts in CI, expectations in the pipelines, and lineage down to the answer |
| Gold views + specifications | Semantic layer with column-level permissions |
| Python tools | Governed functions with the same permissions as the data, also exposed via MCP |
| `PolicyRetriever` | Managed vector index synced with the manual table |
| `ResilientLLM` + fallback chain | Model gateway with routing, rate limiting, guardrails, and payload logging |
| Postgres + Redis | Managed services with high availability |
| OTel + evals | Continuous tracing and evaluation, with sampled real conversations becoming test cases |
| `make check` + CI | Evaluation gate before any change to prompt, tool, or model |

**Production decisions:** private network up to the model and controlled egress; identity via service principal, no personal tokens; separate environments (dev, stg, prd) with PII masking by tag; cost budget with an alarm per conversation; prompt, tools, and model versioned together, deploying only above the evaluation baseline. Roadmap in three waves: (1) PoC with real ERP ingestion; (2) WhatsApp pilot with 10% of traffic and handoff to human agents; (3) new intents (consultative selling, after-sales) and channels.

**Make-or-buy and TCO.** For an agent that depends on proprietary transactional data and exact rules, the recommendation is to build on a managed platform: own code for the rules and tools; managed services for model, vector store, gateway, and observability. The estimated monthly value is `V·c·C_h + ΔR − (V·C_ia + C_op)`, where V is conversations per month, c is the containment rate, C_h is the cost of one human interaction, ΔR is incremental revenue, C_ia is the AI cost per conversation (measured by the `UsageAccumulator`), and C_op is fixed operations. In general the model is not the dominant cost: people and operations are.

## Built with Claude Code

The project was built with **Claude Code**, using the **Claude Opus 5.5** model (`claude-opus-5-5`) as the main agent and **Opus 5.5** and **Sonnet 5.5** (`claude-sonnet-5-5`) subagents for parallel slices, adversarial review, and QA.

**Workflow:**

1. **Slicing.** Each deliverable became a slice with explicit intent and constraints (for example, "payment and PIX, §3 and §6.2, with property-based tests" or "multi-provider LLM layer"). The history shows the slices as small commits and merges (`merge: estado compartilhado em Postgres e Redis (T8)`, `merge: operação de produção (T9)`, `merge: camada LLM multi-provedor (T7)`).
2. **Maker/verifier loop.** One subagent implements the slice in an isolated worktree; another, without the implementer's context, reviews it adversarially and runs validation. Gaps go back to the maker until the verifier approves.
3. **Gate.** Nothing gets in without `make check` (ruff, `mypy --strict`, pytest with coverage) and, when the slice touches agent behavior, without `make eval`.
4. **Live measurement.** The real runs (Gemini, Groq, Ollama, server build) produced traceable fixes, such as commits `60711f4`, `d4264f6`, and `11f5ecf`.

**What stayed under human decision:** architecture and product (graph with router, rules in code, medallion, long-context retrieval, what is out of scope), the source hierarchy between data and manual, validation of the quality report numbers against the CSVs and the PDF, and the final review of the code and of this document.

Every decision was reviewed, and the tests are the proof: every number cited in this README comes from a test, a snapshot, a versioned report, or a recorded run, not from an estimate.

## Folder structure and commands

### Folder structure

```text
.
├── src/emporio/
│   ├── bootstrap.py          # composition root
│   ├── settings.py           # EMPORIO_* configuration (pydantic-settings)
│   ├── production.py         # fail-fast prod profile
│   ├── ports.py              # ports (Protocols)
│   ├── domain/               # entities, value objects, PII, specifications
│   ├── policies/             # payment, returns, shipping, hours (pure functions)
│   ├── ingestion/            # bronze -> silver -> gold, contracts, quality, PDF parser
│   ├── adapters/             # persistence (DuckDB, SQLite, Postgres) and retrieval
│   ├── agent/
│   │   ├── graph.py          # LangGraph graph
│   │   ├── nodes/            # input_guard, router, scope_gate, verification, agent, ...
│   │   ├── tools/            # 13 typed tools
│   │   ├── prompts/          # system, router, and summary in Jinja
│   │   └── llm/              # providers, spec, chat, resilient, errors, doctor, fake
│   ├── guardrails/           # input, output, session, rate limit, attempts
│   ├── evals/                # runner, checks, cassette, judge, examples
│   ├── observability/        # JSON logs, metrics, OpenTelemetry, cost
│   └── interfaces/           # CLI and API (middleware)
├── tests/                    # unit, agent, guardrails, contracts, integration, ...
├── evals/                    # golden, adversarial, retrieval, cassettes, and reports
├── data/                     # raw (immutable), contracts, generated database and report
├── examples/                 # 5 real conversations
├── deploy/                   # docker-compose, otel-collector, k8s (kustomize)
├── .github/workflows/        # ci.yml, release.yml
├── Dockerfile
└── Makefile
```

### Make commands

| Target | What it does |
| --- | --- |
| `make setup` | `uv sync --frozen --all-extras`: reproducible environment from `uv.lock`, identical to CI |
| `make setup-min` | `uv sync --frozen`: base dependencies only, for a lean runtime |
| `make ingest` | Medallion ingestion: `data/emporio.duckdb` and `data/quality_report.json` (idempotent) |
| `make chat` | Chat CLI; accepts `ARGS="--debug"` |
| `make api` | `uvicorn` with `emporio.interfaces.api:create_app` on `127.0.0.1:8000` |
| `make test` | `pytest` with coverage (80% floor) |
| `make lint` | `ruff check` + `ruff format --check` |
| `make typecheck` | `mypy --strict src` |
| `make check` | `lint` + `typecheck` + `test` |
| `make eval` | Offline evaluation with the fake LLM; fails below the floor |
| `make eval-live` | Evaluation with a real provider; writes cassette and report |
| `make eval-replay` | Replays the cassette offline (`STALE` when the prompt changed) |
| `make eval-retrieval` | Recall@3 and MRR of the retrieval strategies |
| `make examples` | Generates the `examples/` transcripts with a real provider |
| `make llm-check` | Diagnostics for credential, tool calling, and structured output per role |
| `make docker-build` | Local image (`EXTRAS=all` by default; `EXTRAS=postgres` for a lean one) |
| `make compose-up` / `make compose-down` | Brings the `deploy/docker-compose.yml` stack up or down (`PROFILE=observability` includes Jaeger) |
