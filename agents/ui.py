"""
AIOps Copilot - Streamlit UI
Real-time chat interface for the AIOps agent with approval workflow.
"""

import streamlit as st
import requests
import time

API_BASE = "http://localhost:8088"

st.set_page_config(
    page_title="AIOps Copilot",
    page_icon="🤖",
    layout="wide",
)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.title("🤖 AIOps Copilot")
st.caption("Autonomous Kubernetes Incident Investigation & Remediation")

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "status" not in st.session_state:
    st.session_state.status = None
if "pending_action" not in st.session_state:
    st.session_state.pending_action = None


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.header("Investigation Status")

    if st.session_state.thread_id:
        st.code(f"Thread ID:\n{st.session_state.thread_id}", language=None)
        st.markdown(f"**Status:** `{st.session_state.status}`")

        if st.session_state.status == "running":
            st.info("🔄 Agents are investigating...")
        elif st.session_state.status == "awaiting_approval":
            st.warning("⏳ Awaiting your approval")
        elif st.session_state.status == "completed":
            st.success("✅ Investigation complete")
        elif st.session_state.status == "failed":
            st.error("❌ Investigation failed")

    st.divider()
    st.header("Quick Questions")
    quick_questions = [
        "My model server keeps getting OOMKilled. Can you investigate and fix it?",
        "My model server response times have doubled. Investigate.",
        "My model server keeps crashing. What's wrong?",
        "Why is my model server consuming so much memory?",
    ]
    for q in quick_questions:
        if st.button(q[:50] + "...", use_container_width=True):
            st.session_state.quick_question = q

    st.divider()
    if st.button("🔄 New Investigation", use_container_width=True):
        st.session_state.thread_id = None
        st.session_state.messages = []
        st.session_state.status = None
        st.session_state.pending_action = None
        st.rerun()


# ─────────────────────────────────────────────
# CHAT HISTORY
# ─────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ─────────────────────────────────────────────
# APPROVAL UI
# ─────────────────────────────────────────────

if st.session_state.status == "awaiting_approval" and st.session_state.pending_action:
    st.divider()
    st.warning("### ⚠️ Human Approval Required")
    st.markdown(f"**Proposed action:** {st.session_state.pending_action}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve", type="primary", use_container_width=True):
            with st.spinner("Submitting approval..."):
                response = requests.post(
                    f"{API_BASE}/approve/{st.session_state.thread_id}",
                    json={"decision": "approve"},
                )
            st.session_state.messages.append({
                "role": "user",
                "content": "✅ **Approved** — proceeding with remediation."
            })
            st.session_state.status = "running"
            st.session_state.pending_action = None
            st.rerun()

    with col2:
        if st.button("❌ Deny", use_container_width=True):
            with st.spinner("Submitting denial..."):
                response = requests.post(
                    f"{API_BASE}/approve/{st.session_state.thread_id}",
                    json={"decision": "deny"},
                )
            st.session_state.messages.append({
                "role": "user",
                "content": "❌ **Denied** — remediation cancelled."
            })
            st.session_state.status = "running"
            st.session_state.pending_action = None
            st.rerun()


# ─────────────────────────────────────────────
# POLLING (auto-refresh when running)
# ─────────────────────────────────────────────

def poll_status():
    if st.session_state.thread_id and st.session_state.status == "running":
        try:
            response = requests.get(f"{API_BASE}/status/{st.session_state.thread_id}")
            data = response.json()
            st.session_state.status = data["status"]

            if data["status"] == "awaiting_approval":
                st.session_state.pending_action = data.get("pending_action", "")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⏳ **Approval required:** {st.session_state.pending_action}"
                })

            elif data["status"] == "completed":
                final_answer = data.get("final_answer", "Investigation complete.")
                agents = data.get("agents_called", [])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"{final_answer}\n\n---\n**Agents called:** {' → '.join(agents)}"
                })

            elif data["status"] == "failed":
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ Investigation failed: {data.get('error', 'Unknown error')}"
                })

        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ Error polling status: {str(e)}"
            })


# ─────────────────────────────────────────────
# INPUT HANDLING
# ─────────────────────────────────────────────

# Handle quick question from sidebar
question = None
if "quick_question" in st.session_state:
    question = st.session_state.quick_question
    del st.session_state.quick_question

# Handle chat input
chat_input = st.chat_input("Describe the incident (e.g. 'My model server keeps OOMKilling')")
if chat_input:
    question = chat_input

if question:
    # Reset state for new investigation
    st.session_state.messages = []
    st.session_state.thread_id = None
    st.session_state.status = None
    st.session_state.pending_action = None

    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})

    # Start investigation via API
    with st.spinner("Starting investigation..."):
        try:
            response = requests.post(
                f"{API_BASE}/investigate",
                json={"question": question},
            )
            data = response.json()
            st.session_state.thread_id = data["thread_id"]
            st.session_state.status = "running"
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"🔍 Investigation started. Thread ID: `{st.session_state.thread_id}`\n\nAgents are now investigating your incident..."
            })
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ Failed to start investigation: {str(e)}"
            })

    st.rerun()

# Auto-poll when running
if st.session_state.status == "running":
    poll_status()
    if st.session_state.status == "running":
        time.sleep(3)
        st.rerun()
    else:
        st.rerun()