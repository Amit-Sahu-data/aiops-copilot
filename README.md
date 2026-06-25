# AIOps Copilot

A production-grade multi-agent LLM system for autonomous Kubernetes operations, 
built as a portfolio project demonstrating real-world LLMOps engineering.

## What it does

Given a natural language incident description ("My model server keeps getting 
OOMKilled"), the system autonomously:

1. Investigates the Kubernetes cluster (pod status, resource limits, deployment info)
2. Fetches and analyzes application logs, including pre-crash logs from previous 
   container instances
3. Synthesizes a root cause assessment with explicit evidence requirements
4. Searches internal runbooks for approved remediation procedures
5. Proposes a specific, runbook-grounded fix — and pauses for human approval 
   before executing any write operation

## Architecture
User Query

│

▼

Supervisor (LangGraph orchestrator)

│

├──► Kubernetes Agent    → get_pod_status, get_pod_resource_limits, get_deployment_info

│

├──► Log Analysis Agent  → get_pod_logs (with previous=True for crash logs)

│

├──► Root Cause Agent    → evidence-gated synthesis, no tools

│

└──► Remediation Agent   → search_runbooks, then one of:

restart_pod | patch_memory_limit | scale_deployment

│

▼

Human Approval Gate (interrupt/resume)

│

approve ──┤── deny

│

Executes on cluster / Declines cleanly

**Key design decisions:**
- Supervisor uses `with_structured_output` (Pydantic) to prevent routing drift
- Hard code-level guards prevent agents re-running after they've been called
- RCA agent has an explicit LATENCY EVIDENCE RULE: OOMKill evidence never 
  establishes a latency root cause without direct latency measurements
- Remediation agent has a code-level RCA gate: if RCA found no root cause, 
  the LLM call is skipped entirely and a hardcoded decline is returned
- Human approval only fires for write tools (`restart_pod`, `patch_memory_limit`, 
  `scale_deployment`) — read-only tools pass through silently

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent framework | LangGraph (multi-agent graph with interrupt/resume) |
| LLM | GPT-4o-mini via LangChain ChatOpenAI |
| Infrastructure | Kubernetes (Minikube locally, AKS in production) |
| Vector store | ChromaDB + sentence-transformers (all-MiniLM-L6-v2) |
| Checkpointing | SQLite via langgraph-checkpoint-sqlite |
| Observability | LangFuse (per-agent token costs, latency, trace inspection) |
| Resilience | Tenacity (retry with transient-error classification) |
| Evaluation | Custom eval framework (10 scenarios, 7 assertion layers) |

## Production Hardening

### Retries
All Kubernetes API calls use `tenacity` with exponential backoff. Critically, 
retries are classified by error type:
- **Transient** (429, 500, 502, 503, 504): retried up to 3 times with backoff
- **Permanent** (400, 404): fail immediately — retrying won't help
- Write tools (`restart_pod`, `patch_memory_limit`, `scale_deployment`) are 
  idempotent, making retries safe

### Timeouts
- LLM calls: `ChatOpenAI(timeout=30, max_retries=2)`
- Kubernetes API calls: `_request_timeout=10` per call (not global config, 
  which caused `LocationValueError` in this client version)

### Persistent Checkpointing
`SqliteSaver` persists graph state across process restarts. A human approval 
pause survives indefinitely — the operator can resume hours later with the 
same `thread_id`. Proven end-to-end: process A pauses at approval, exits; 
process B resumes cleanly without re-running the investigation.

**Key lesson:** Resume must call `graph.invoke(None, config=config)` — injecting 
a fresh state dict corrupts the message sequence and causes OpenAI 400 errors.

## Evaluation Suite

10 scenarios across 4 categories, run via `python eval.py`:

| Category | Scenarios | Key assertions |
|----------|-----------|----------------|
| happy_path | OOMKill approve, OOMKill deny | Full pipeline runs, correct tools called, write executes/doesn't execute |
| safety | No fix without root cause, runbook before write, RCA always has 3 sections | Safety invariants hold across 3 runs each |
| adversarial | Latency not linked to OOMKill, dangerous request rejected | System resists speculative reasoning and unsafe requests |
| regression | One test per real bug found during development | Bugs don't silently reappear |

**Design principles:**
- `forbidden_writes` checks tool *execution* (ToolMessage content), not tool 
  *proposal* — a denied action legitimately appears as a proposed call
- `require_oomkill` scenarios skip cleanly when no incident exists rather than 
  failing misleadingly
- Stability testing: safety scenarios require 3/3 passes (100%), adversarial 
  scenarios use configurable thresholds
- Exits with code 1 on any failure — CI/CD ready

## Real Bugs Found During Development

These became regression tests:

1. **Supervisor routing drift** — LLM wrapped routing decisions in prose; fixed 
   with `with_structured_output` (Pydantic)
2. **Agent re-running redundantly** — fixed with `agents_called` hard guards in 
   routing logic
3. **RCA contradicting itself between sections** — fixed by explicitly mandating 
   "No root cause established" as valid output
4. **Remediation bypassing RCA gate** — prompt-only instruction was violated once; 
   fixed with a code-level gate that skips the LLM call entirely
5. **Approval gate firing on read-only tools** — fixed by scoping to `WRITE_TOOLS` set
6. **Checkpoint resume crash** — resuming with fresh state dict corrupted message 
   ordering; fixed by passing `None` as input on resume

## Running Locally

```bash
# Prerequisites: Minikube, Docker Desktop, Python 3.11+

# Start cluster
minikube start --driver=docker --cpus=2 --memory=3072

# Apply manifests
kubectl apply -f k8s/dummy-model-deployment.yaml
kubectl apply -f k8s/dummy-model-servicemonitor.yaml

# Set up Python environment
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Build runbook index
cd agents
python build_doc_index.py

# Run
python main.py

# Eval suite (state-independent)
python eval.py safety adversarial regression

# Full eval suite (requires live OOMKill)
python eval.py
```

## Environment Variables
OPENAI_API_KEY=sk-...

LANGFUSE_PUBLIC_KEY=pk-lf-...

LANGFUSE_SECRET_KEY=sk-lf-...

LANGFUSE_HOST=https://cloud.langfuse.com

## CI/CD

GitHub Actions runs the state-independent eval scenarios on every push to `main`. 
OOMKill-dependent scenarios require a live cluster and are covered by the online 
monitoring layer (LangFuse) rather than CI — this is the correct production pattern: 
offline CI for logic correctness, online monitoring for infrastructure-dependent behavior.