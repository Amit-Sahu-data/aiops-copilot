import os
from dotenv import load_dotenv
from typing import Annotated, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from k8s_tools import (
    get_pod_status, get_pod_resource_limits, get_deployment_info, get_pod_logs,
    restart_pod, patch_memory_limit, scale_deployment,search_runbooks,
)

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=30, max_retries=2)

import os
from langfuse.langchain import CallbackHandler

from langfuse.langchain import CallbackHandler


from langfuse.langchain import CallbackHandler

def get_langfuse_handler(session_id: str = None):
    trace_id = session_id.replace("-", "") if session_id else None
    trace_context = {"trace_id": trace_id} if trace_id else {}
    return CallbackHandler(trace_context=trace_context)

# ---- State ----
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    next_agent: str
    iteration_count: int
    agents_called: list[str]


# ---- Supervisor (structured output) ----
class SupervisorDecision(BaseModel):
    next_agent: Literal[
        "kubernetes_agent", "log_analysis_agent", "root_cause_agent", "remediation_agent", "FINISH"
    ] = Field(description="The next agent to call, or FINISH if the investigation is complete.")


SUPERVISOR_SYSTEM_PROMPT = """You are a supervisor coordinating an AIOps incident investigation team.
Based on the conversation so far, decide which specialist agent should act next.

Available agents:
- kubernetes_agent: handles questions about pod status, restarts, OOMKills, resource limits, deployments
- log_analysis_agent: handles questions about application logs, error messages, and request-level details
- root_cause_agent: synthesizes findings into a final assessment. Call this LAST among investigation agents.
- remediation_agent: proposes a fix for the root cause. Only call this AFTER root_cause_agent has run.
- FINISH: choose this once the investigation (and remediation, if applicable) is complete.

RULES:
- If the question mentions multiple symptoms, ensure both kubernetes_agent AND log_analysis_agent
  have been called before root_cause_agent.
- Do not call the same agent twice in a row if it already gave its answer.
- Always call root_cause_agent before remediation_agent.
- Only call remediation_agent if the user's question implies they want a fix, not just a diagnosis.

Output only your routing decision.
"""

supervisor_llm = llm.with_structured_output(SupervisorDecision)


def supervisor_node(state: AgentState):
    iteration_count = state.get("iteration_count", 0) + 1
    agents_called = state.get("agents_called", [])

    if iteration_count > 10:
        return {"next_agent": "FINISH", "iteration_count": iteration_count, "agents_called": agents_called}

    context_note = f"\n\nAgents already called so far: {agents_called if agents_called else 'none'}."
    messages = [SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT + context_note)] + state["messages"]
    decision: SupervisorDecision = supervisor_llm.invoke(messages)

    print(f"\n[DEBUG] Supervisor decision: {decision.next_agent} | agents_called so far: {agents_called} | iteration: {iteration_count}")

    return {"next_agent": decision.next_agent, "iteration_count": iteration_count, "agents_called": agents_called}


# ---- Kubernetes Agent ----
k8s_tools = [get_pod_status, get_pod_resource_limits, get_deployment_info]
k8s_llm = llm.bind_tools(k8s_tools)

K8S_SYSTEM_PROMPT = """You are a Kubernetes diagnostics agent. Use the available tools to inspect
pod status, restarts, resource limits, and deployments. Report factual findings clearly,
including specific numbers (restart counts, memory limits, exit codes).
Only state conclusions that are directly supported by the tool output. Do NOT speculate about
causes outside your data."""


def kubernetes_agent_node(state: AgentState):
    messages = [SystemMessage(content=K8S_SYSTEM_PROMPT)] + state["messages"]
    response = k8s_llm.invoke(messages)
    agents_called = state.get("agents_called", [])
    if "kubernetes_agent" not in agents_called:
        agents_called = agents_called + ["kubernetes_agent"]
    return {"messages": [response], "agents_called": agents_called}


k8s_tool_node = ToolNode(k8s_tools)


# ---- Log Analysis Agent ----
log_tools = [get_pod_status, get_pod_logs]
log_llm = llm.bind_tools(log_tools)

LOG_SYSTEM_PROMPT = """You are a log analysis agent. First use get_pod_status to find the pod name
and check its restart count. If restarts > 0, you MUST call get_pod_logs with previous=True to see
logs from before the crash. Look for error messages, exceptions, warnings, or unusual patterns.
Quote specific log lines that are relevant. If logs are empty or unremarkable, say so clearly."""


def log_analysis_agent_node(state: AgentState):
    messages = [SystemMessage(content=LOG_SYSTEM_PROMPT)] + state["messages"]
    response = log_llm.invoke(messages)
    agents_called = state.get("agents_called", [])
    if "log_analysis_agent" not in agents_called:
        agents_called = agents_called + ["log_analysis_agent"]
    return {"messages": [response], "agents_called": agents_called}


log_tool_node = ToolNode(log_tools)


# ---- Root Cause Analysis Agent ----
RCA_SYSTEM_PROMPT = """You are a Root Cause Analysis agent. You do NOT call any tools yourself.
Your job is to synthesize the findings already gathered by the kubernetes_agent and log_analysis_agent
in the conversation above, and produce a final root cause assessment.

CRITICAL RULES:
1. Before claiming a causal link to ANY symptom the user mentioned, you MUST check: did kubernetes_agent
   or log_analysis_agent retrieve actual evidence that directly measures or mentions that specific symptom?
   - If YES: state the causal link and quote the specific evidence.
   - If NO: you MUST NOT claim a causal link. State explicitly in UNVERIFIED / NEEDS MORE DATA that the
     symptom was never directly measured, and that any connection to other findings is SPECULATION.
2. A pod restart, OOMKill, or memory warning is evidence of a memory/stability problem — it is NOT
   automatically evidence of a latency problem unless directly measured.
3. Separate your output into three sections:
   - CONFIRMED FINDINGS (directly stated by tool output, no inference)
   - LIKELY ROOT CAUSE: state a root cause ONLY if Rule 1's evidence bar is met. If NOT met, this
     section must say exactly: "No root cause for the user's stated symptom is established by current
     evidence." It is correct and expected for this section to be short when evidence is insufficient.
   - UNVERIFIED / NEEDS MORE DATA (any user-stated symptom without direct evidence)
4. Do not soften or hide gaps in the evidence.
5. LATENCY EVIDENCE RULE:
If the user's symptom involves latency, response times, or slowness:
- You MUST have direct latency evidence (slow request logs, p99 response time metrics, 
  timeout errors) to establish a latency root cause.
- OOMKill evidence, memory pressure, or restart history NEVER establishes a latency 
  root cause on its own — a pod can OOMKill and recover without any measurable latency impact.
- If you only have memory/restart evidence and the user asked about latency, 
  the LIKELY ROOT CAUSE section MUST state: 
  "No root cause for the user's stated symptom is established by current evidence."
"""


def rca_agent_node(state: AgentState):
    messages = [SystemMessage(content=RCA_SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    agents_called = state.get("agents_called", [])
    if "root_cause_agent" not in agents_called:
        agents_called = agents_called + ["root_cause_agent"]
    return {"messages": [response], "agents_called": agents_called}


# ---- Remediation Agent ----
remediation_tools = [restart_pod, patch_memory_limit, scale_deployment, search_runbooks]
remediation_llm = llm.bind_tools(remediation_tools)

REMEDIATION_SYSTEM_PROMPT = """You are a Remediation agent. Based on the root cause analysis above,
propose ONE specific remediation action using the available tools (restart_pod, patch_memory_limit,
scale_deployment). 

BEFORE proposing any action, you MUST call search_runbooks to check if there is an established,
approved procedure for this type of incident. Follow the runbook's guidance on specific values
(e.g., standard memory limit increments) rather than picking arbitrary numbers yourself.

Only propose an action if the root cause agent established a clear, evidence-based root cause AND
a runbook supports that type of action — if the root cause was unverified or no runbook supports
the proposed action, state that no safe remediation can be proposed and do NOT call any write tool.
Be specific: include exact pod/deployment names and parameter values from the conversation above."""

def remediation_agent_node(state: AgentState):
    # Code-level gate: check if RCA explicitly found no established root cause.
    # If so, skip the LLM call entirely and short-circuit to a safe decline.
    rca_messages = [m for m in state["messages"] if isinstance(m, AIMessage) and "LIKELY ROOT CAUSE" in (m.content or "")]
    if rca_messages:
        last_rca_content = rca_messages[-1].content
        if "No root cause for the user's stated symptom is established by current evidence" in last_rca_content:
            agents_called = state.get("agents_called", [])
            if "remediation_agent" not in agents_called:
                agents_called = agents_called + ["remediation_agent"]
            decline_message = AIMessage(
                content="No remediation action is being proposed. Root Cause Analysis did not establish "
                        "a clear, evidence-based root cause for the reported symptom, so no safe fix can be recommended. "
                        "Further investigation is needed before any remediation action should be taken."
            )
            return {"messages": [decline_message], "agents_called": agents_called}

    messages = [SystemMessage(content=REMEDIATION_SYSTEM_PROMPT)] + state["messages"]
    response = remediation_llm.invoke(messages)
    agents_called = state.get("agents_called", [])
    if "remediation_agent" not in agents_called:
        agents_called = agents_called + ["remediation_agent"]


    # Defense-in-depth: if the model produced prose that sounds like it's proposing
    # or about to take an action, but didn't actually call a tool, force a retry with
    # an explicit correction rather than letting it silently end the run.
    intent_phrases = ["i will proceed", "please confirm", "would you like me to", "shall i"]
    if not response.tool_calls and any(p in (response.content or "").lower() for p in intent_phrases):
        correction = SystemMessage(content=(
            "Your previous response described an intended action in text instead of calling a tool, "
            "or asked the user for confirmation, which this system does not support. Either call the "
            "appropriate write tool now, or state plainly that no remediation is being proposed."
        ))
        retry_messages = messages + [response, correction]
        response = remediation_llm.invoke(retry_messages)

    return {"messages": [response], "agents_called": agents_called}


WRITE_TOOLS = {"restart_pod", "patch_memory_limit", "scale_deployment"}


def human_approval_node(state: AgentState):
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]

    if tool_call["name"] not in WRITE_TOOLS:
        # Read-only tool (e.g., search_runbooks) — no approval needed, fall through silently
        return {}

    approval_request = {
        "action": tool_call["name"],
        "arguments": tool_call["args"],
        "message": f"Remediation agent wants to call `{tool_call['name']}` with arguments: {tool_call['args']}. Approve this action?",
    }

    decision = interrupt(approval_request)

    if decision != "approve":
        denial_message = ToolMessage(
            content=f"Action '{tool_call['name']}' was NOT approved by the human operator. No changes were made to the cluster.",
            tool_call_id=tool_call["id"],
        )
        return {"messages": [denial_message]}

    return {}


remediation_tool_node = ToolNode(remediation_tools)


# ---- Routing logic ----
def route_from_supervisor(state: AgentState) -> Literal[
    "kubernetes_agent", "log_analysis_agent", "root_cause_agent", "remediation_agent", "__end__"
]:
    decision = state["next_agent"]
    agents_called = state.get("agents_called", [])
    iteration_count = state.get("iteration_count", 0)

    # Hard rule: once remediation has run (proposed a fix or declined), it is always the final word.
    if "remediation_agent" in agents_called:
        print("[DEBUG] remediation_agent already ran -> forcing FINISH, no further agents allowed")
        return "__end__"

    # Hard rule: once RCA has run, only remediation_agent or FINISH may follow.
    if "root_cause_agent" in agents_called and decision not in ("remediation_agent", "FINISH"):
        decision = "FINISH"

    if decision in agents_called:
        decision = "FINISH"

    if decision == "FINISH":
        if "kubernetes_agent" not in agents_called and iteration_count <= 10:
            return "kubernetes_agent"
        if "log_analysis_agent" not in agents_called and iteration_count <= 10:
            return "log_analysis_agent"
        if "root_cause_agent" not in agents_called and iteration_count <= 10:
            return "root_cause_agent"
        print("[DEBUG] Allowing FINISH")
        return "__end__"

    return decision


def route_from_k8s_agent(state: AgentState) -> Literal["tools", "supervisor"]:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "supervisor"


def route_from_log_agent(state: AgentState) -> Literal["log_tools", "supervisor"]:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "log_tools"
    return "supervisor"


def route_from_remediation_agent(state: AgentState) -> Literal["human_approval", "supervisor"]:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "human_approval"
    return "supervisor"


def route_from_approval(state: AgentState) -> Literal["remediation_tools", "remediation_agent"]:
    last_message = state["messages"][-1]
    if isinstance(last_message, ToolMessage) and "NOT approved" in last_message.content:
        return "remediation_agent"
    return "remediation_tools"

# ---- Build the graph ----
graph_builder = StateGraph(AgentState)

graph_builder.add_node("supervisor", supervisor_node)
graph_builder.add_node("kubernetes_agent", kubernetes_agent_node)
graph_builder.add_node("tools", k8s_tool_node)
graph_builder.add_node("log_analysis_agent", log_analysis_agent_node)
graph_builder.add_node("log_tools", log_tool_node)
graph_builder.add_node("root_cause_agent", rca_agent_node)
graph_builder.add_node("remediation_agent", remediation_agent_node)
graph_builder.add_node("human_approval", human_approval_node)
graph_builder.add_node("remediation_tools", remediation_tool_node)

graph_builder.set_entry_point("supervisor")

graph_builder.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {
        "kubernetes_agent": "kubernetes_agent",
        "log_analysis_agent": "log_analysis_agent",
        "root_cause_agent": "root_cause_agent",
        "remediation_agent": "remediation_agent",
        "__end__": END,
    },
)

graph_builder.add_conditional_edges(
    "kubernetes_agent", route_from_k8s_agent, {"tools": "tools", "supervisor": "supervisor"}
)
graph_builder.add_edge("tools", "kubernetes_agent")

graph_builder.add_conditional_edges(
    "log_analysis_agent", route_from_log_agent, {"log_tools": "log_tools", "supervisor": "supervisor"}
)
graph_builder.add_edge("log_tools", "log_analysis_agent")

graph_builder.add_edge("root_cause_agent", "remediation_agent")

graph_builder.add_conditional_edges(
    "remediation_agent",
    route_from_remediation_agent,
    {"human_approval": "human_approval", "supervisor": "supervisor"},
)
graph_builder.add_conditional_edges(
    "human_approval",
    route_from_approval,
    {"remediation_tools": "remediation_tools", "remediation_agent": "remediation_agent"},
)
graph_builder.add_edge("remediation_tools", "remediation_agent")


POSTGRES_URL = os.getenv("POSTGRES_URL")
SQLITE_PATH = os.path.join(os.path.dirname(__file__), "agent_checkpoints.db")

if POSTGRES_URL:
    import psycopg
    from langgraph.checkpoint.postgres import PostgresSaver
    postgres_conn = psycopg.connect(POSTGRES_URL, autocommit=True)
    checkpointer = PostgresSaver(postgres_conn)
    checkpointer.setup()
    print("[INFO] Using Postgres checkpointer")

else:
    from langgraph.checkpoint.sqlite import SqliteSaver
    import sqlite3
    conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    print("[INFO] Using SQLite checkpointer (fallback)")

graph = graph_builder.compile(checkpointer=checkpointer)