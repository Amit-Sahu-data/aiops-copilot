import os
import requests
from dotenv import load_dotenv

load_dotenv()


def _send(payload: dict):
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    print(f"[Slack DEBUG] webhook_url = {webhook_url[:20] if webhook_url else 'None'}")
    if not webhook_url:
        print("[Slack] No webhook URL configured")
        return
    try:
        print(f"[Slack DEBUG] Sending payload: {payload}")
        response = requests.post(webhook_url, json=payload, timeout=5)
        print(f"[Slack] Response: {response.status_code} {response.text}")
    except Exception as e:
        print(f"[Slack] Failed: {e}")


def send_investigation_started(thread_id: str, question: str):
    _send({"text": f"🔍 *AIOps Investigation Started*\n*Incident:* {question}\n*Thread:* `{thread_id[:8]}...`"})


def send_approval_request(thread_id: str, question: str, proposed_action: str, api_base_url: str):
    _send({"text": f"⚠️ *AIOps Approval Required*\n*Incident:* {question}\n*Proposed Action:* {proposed_action}\n*Approve:* `POST {api_base_url}/approve/{thread_id}` with `{{\"decision\": \"approve\"}}`"})


def send_investigation_complete(thread_id: str, question: str, final_answer: str, agents_called: list):
    preview = final_answer[:300] + "..." if len(final_answer) > 300 else final_answer
    _send({"text": f"✅ *AIOps Investigation Complete*\n*Incident:* {question}\n*Agents:* {' → '.join(agents_called)}\n*Result:* {preview}"})