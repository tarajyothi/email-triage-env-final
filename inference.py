"""
=============================================================================
  inference.py — OpenEnv Inference Runner
  OpenEnv Hackathon — Round 1 Submission
=============================================================================

Runs all three tasks (easy / medium / hard) using an LLM agent via the
OpenAI-compatible client.  Falls back to the rule-based baseline_agent if
the API is unavailable or returns an unparseable response.

Environment variables:
    API_BASE_URL   — base URL for the OpenAI-compatible endpoint
    MODEL_NAME     — model identifier (e.g. "gpt-4o-mini")
    HF_TOKEN       — Hugging Face token (passed as Bearer if API_BASE_URL
                     points to a HF Inference Endpoint)

Output format is STRICT — do not modify the [START] / [STEP] / [END] blocks.
"""

import os
import json

from openai import OpenAI
from graders import run_task, TASKS
from email_triage_env import baseline_agent, EmailTriageEnv

# =============================================================================
# CLIENT SETUP
# =============================================================================

API_BASE_URL = os.environ.get("API_BASE_URL", "")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")

# Build client only if a base URL is configured
_client: OpenAI | None = None
if API_BASE_URL:
    _client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN or "no-key",
    )


# =============================================================================
# LLM AGENT
# =============================================================================

_SYSTEM_PROMPT = """
You are an email triage agent. Given an email observation, return a JSON
object with exactly these keys:
  - priority    : integer 0 (low), 1 (medium), or 2 (high)
  - category    : integer 0 (bug), 1 (billing), 2 (general), or 3 (spam)
  - action_type : integer 0 (ignore), 1 (respond), or 2 (escalate)

Return ONLY the JSON object. No explanation, no markdown.
""".strip()


def _parse_llm_response(text: str) -> dict | None:
    """Parse LLM JSON response; return None if malformed."""
    try:
        text  = text.strip().strip("```json").strip("```").strip()
        data  = json.loads(text)
        keys  = {"priority", "category", "action_type"}
        if not keys.issubset(data):
            return None
        action = {k: int(data[k]) for k in keys}
        env    = EmailTriageEnv()
        return action if env.action_space.contains(action) else None
    except Exception:
        return None


def llm_agent(observation: dict) -> dict:
    """
    Query the LLM for a triage decision; fall back to baseline on failure.

    Parameters
    ----------
    observation : dict
        {email_text, sender_type, urgency_flag}

    Returns
    -------
    dict
        {priority, category, action_type}
    """
    if _client is None:
        return baseline_agent(observation)

    user_msg = (
        f"Email: {observation['email_text']!r}\n"
        f"Sender: {observation['sender_type']}\n"
        f"Urgent: {observation['urgency_flag']}\n"
        "Respond with JSON only."
    )

    try:
        response = _client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0,
            max_tokens=64,
        )
        text   = response.choices[0].message.content or ""
        action = _parse_llm_response(text)
        return action if action is not None else baseline_agent(observation)
    except Exception:
        return baseline_agent(observation)


# =============================================================================
# INFERENCE RUNNER
# =============================================================================

def run_inference() -> None:
    """Run all three tasks and print results in strict OpenEnv log format."""

    agent_fn = llm_agent  # uses baseline fallback when API is unavailable

    for task_name in ["easy", "medium", "hard"]:

        # ── [START] ────────────────────────────────────────────────────
        print("[START]")
        print(f"task: {task_name}")
        print()

        result = run_task(task_name, agent_fn=agent_fn)

        # ── [STEP] per email ───────────────────────────────────────────
        for step in result["steps"]:
            action = step["action"]
            reward = step["reward"]

            action_str = (
                f"priority={action['priority']} "
                f"category={action['category']} "
                f"action_type={action['action_type']}"
            )

            print("[STEP]")
            print(f"action: {action_str}")
            print(f"reward: {reward:+d}")
            print()

        # ── [END] ──────────────────────────────────────────────────────
        print("[END]")
        print(f"score: {result['score']:.2f}")
        print()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_inference()
