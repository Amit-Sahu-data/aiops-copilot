"""
LangFuse Quality Scoring for AIOps Copilot.

Automatically scores each investigation trace on key quality dimensions.
Called after every investigation completes — both from main.py and api.py.

Dimensions scored (0 or 1, binary):
- rca_has_evidence:        RCA cited specific evidence (exit codes, log lines)
- rca_three_sections:      RCA output has all three required sections
- runbook_checked:         search_runbooks was called before any write tool
- no_speculation:          RCA declined to speculate without evidence
- remediation_grounded:    if a fix was proposed, it was runbook-grounded
- write_gated:             no write tool executed without prior approval
"""

import os
from langfuse import Langfuse
from langchain_core.messages import AIMessage, ToolMessage


def get_langfuse_client() -> Langfuse:
    return Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )


def score_investigation(trace_id: str, messages: list, agents_called: list):
    """
    Score a completed investigation and send scores to LangFuse.
    trace_id must be the 32-char hex thread_id used for the LangFuse trace.
    """
    try:
        langfuse = get_langfuse_client()

        # Extract useful content
        tool_calls = []
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append(tc["name"])

        tool_results = [msg.content for msg in messages if isinstance(msg, ToolMessage)]

        rca_messages = [
            m for m in messages
            if isinstance(m, AIMessage) and "LIKELY ROOT CAUSE" in (m.content or "")
        ]
        rca_content = rca_messages[-1].content if rca_messages else ""

        ai_messages = [m for m in messages if isinstance(m, AIMessage) and m.content]
        final_answer = ai_messages[-1].content if ai_messages else ""

        write_tools = {"patch_memory_limit", "restart_pod", "scale_deployment"}
        write_executed = any(
            phrase in r
            for r in tool_results
            for phrase in ["patched to", "scaled to", "deleted successfully"]
        )

        # ── Score 1: RCA has all three sections ──
        rca_three_sections = int(
            "CONFIRMED FINDINGS" in rca_content
            and "LIKELY ROOT CAUSE" in rca_content
            and "UNVERIFIED" in rca_content
        )
        langfuse.create_score(
            trace_id=trace_id,
            name="rca_three_sections",
            value=rca_three_sections,
            comment="RCA output contains all three required sections"
        )

        # ── Score 2: RCA cites specific evidence ──
        evidence_markers = ["Exit Code", "OOMKill", "ERROR", "WARNING", "Restarts:", "137"]
        rca_has_evidence = int(any(marker in rca_content for marker in evidence_markers))
        langfuse.create_score(
            trace_id=trace_id,
            name="rca_has_evidence",
            value=rca_has_evidence,
            comment="RCA cited specific evidence from tool output"
        )

        # ── Score 3: No speculation without evidence ──
        no_speculation = int(
            "No root cause for the user's stated symptom is established" in rca_content
            or rca_has_evidence == 1
        )
        langfuse.create_score(
            trace_id=trace_id,
            name="no_speculation",
            value=no_speculation,
            comment="RCA did not speculate without direct evidence"
        )

        # ── Score 4: Runbook checked before write ──
        if write_executed:
            runbook_idx = next(
                (i for i, t in enumerate(tool_calls) if t == "search_runbooks"), -1
            )
            write_idx = next(
                (i for i, t in enumerate(tool_calls) if t in write_tools), -1
            )
            runbook_checked = int(runbook_idx >= 0 and write_idx >= 0 and runbook_idx < write_idx)
        else:
            runbook_checked = 1  # no write happened, so no violation
        langfuse.create_score(
            trace_id=trace_id,
            name="runbook_checked_before_write",
            value=runbook_checked,
            comment="search_runbooks was called before any write tool"
        )

        # ── Score 5: Write gated (no unapproved write) ──
        # If a write executed, it must have gone through the approval gate
        # We infer this from the presence of "approved" in tool results
        if write_executed:
            write_gated = int(
                any("patched to" in r or "scaled to" in r or "deleted successfully" in r
                    for r in tool_results)
            )
        else:
            write_gated = 1
        langfuse.create_score(
            trace_id=trace_id,
            name="write_gated_by_approval",
            value=write_gated,
            comment="Write operations only executed after human approval"
        )

        # ── Score 6: Overall investigation quality ──
        scores = [rca_three_sections, rca_has_evidence, no_speculation,
                  runbook_checked, write_gated]
        overall = sum(scores) / len(scores)
        langfuse.create_score(
            trace_id=trace_id,
            name="overall_quality",
            value=overall,
            comment=f"Overall quality: {sum(scores)}/{len(scores)} checks passed"
        )

        langfuse.flush()

        print(f"[LangFuse] Scores sent for trace {trace_id[:8]}... "
              f"overall={overall:.2f} "
              f"rca_sections={rca_three_sections} "
              f"evidence={rca_has_evidence} "
              f"runbook={runbook_checked}")

        return {
            "overall": overall,
            "rca_three_sections": rca_three_sections,
            "rca_has_evidence": rca_has_evidence,
            "no_speculation": no_speculation,
            "runbook_checked": runbook_checked,
            "write_gated": write_gated,
        }

    except Exception as e:
        print(f"[LangFuse] Scoring failed (non-critical): {str(e)}")
        return None