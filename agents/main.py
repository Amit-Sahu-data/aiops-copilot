import uuid

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from graph import graph, get_langfuse_handler


def run_query(question: str, show_full_trace: bool = False, existing_thread_id: str = None):
    print(f"\n{'='*60}\nQUERY: {question}\n{'='*60}")

    thread_id = existing_thread_id or str(uuid.uuid4())
    print(f"[INFO] thread_id: {thread_id}")

    langfuse_handler = get_langfuse_handler(
    session_id=thread_id,
    user_id="amit",
    incident_type="OOMKilled Investigation",
)

    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [langfuse_handler],
    }

    if existing_thread_id:
        print("[INFO] Resuming from saved checkpoint...")
        result = graph.invoke(None, config=config)
    else:
        result = graph.invoke(
            {
                "messages": [HumanMessage(content=question)],
                "next_agent": "",
                "iteration_count": 0,
                "agents_called": [],
            },
            config=config,
        )

    while "__interrupt__" in result:
        interrupt_info = result["__interrupt__"][0].value

        print("\n" + "!" * 60)
        print("HUMAN APPROVAL REQUIRED")
        print("!" * 60)
        print(interrupt_info["message"])
        print(f"\n[To resume later, rerun with thread_id={thread_id}]")

        user_input = input(
            "\nType 'approve' to proceed, 'later' to exit and resume later, or anything else to deny: "
        ).strip().lower()

        if user_input == "later":
            print("Exiting. Run again with the same thread_id to resume.")
            return

        decision = "approve" if user_input == "approve" else "deny"

        result = graph.invoke(
            Command(resume=decision),
            config=config,
        )

    if show_full_trace:
        print("\n=== FULL INTERNAL TRACE (for debugging) ===")
        for msg in result["messages"]:
            role = msg.__class__.__name__
            print(f"\n[{role}]: {msg.content}")

    final_answer = result["messages"][-1].content

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(final_answer)


if __name__ == "__main__":
    import sys

    thread_id_arg = sys.argv[1] if len(sys.argv) > 1 else None

    run_query(
        "My model server keeps getting OOMKilled. Can you investigate and fix it?",
        show_full_trace=True,
        existing_thread_id=thread_id_arg,
    )