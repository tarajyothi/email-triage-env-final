
# AI Email Triage System — OpenEnv RL Environment

**OpenEnv Hackathon — Round 1 Submission**

## Problem Description

Email overload is one of the most persistent productivity problems in knowledge-work
organisations. An intelligent triage agent that correctly classifies, prioritises, and
routes emails can save significant human time and prevent critical issues from being missed.

This environment provides a structured, evaluable sandbox for training and benchmarking
such an agent using Reinforcement Learning.

---

## Why This Is Real-World

Unlike synthetic classification benchmarks, this environment introduces:

- **Multi-dimensional decisions** — the agent must simultaneously decide priority,
  category, and action type, mirroring compound real-world judgement calls.
- **Conflicting signals** — a boss can send a casual low-urgency note *or* a critical
  deadline request; the agent must resolve ambiguity correctly.
- **Operational cost modelling** — over-escalation is penalised (costly in practice),
  while correct urgent handling earns a bonus.
- **Edge cases** — empty emails, phishing disguised as billing, urgent-looking spam.

---

## Observation Space

| Field         | Type   | Values                                              |
|---------------|--------|-----------------------------------------------------|
| `email_text`  | str    | Raw email body (may be empty)                       |
| `sender_type` | str    | `"boss"` \| `"customer"` \| `"employee"` \| `"spam"` \| `"system"` |
| `urgency_flag`| int    | `0` (normal) \| `1` (urgent)                        |

---

## Action Space

| Key           | Type | Values                                              |
|---------------|------|-----------------------------------------------------|
| `priority`    | int  | `0` (low) \| `1` (medium) \| `2` (high)             |
| `category`    | int  | `0` (bug) \| `1` (billing) \| `2` (general) \| `3` (spam) |
| `action_type` | int  | `0` (ignore) \| `1` (respond) \| `2` (escalate)     |

---

## Reward Design

| Event                                          | Reward |
|------------------------------------------------|--------|
| Correct priority                               |  +10   |
| Correct category                               |   +8   |
| Correct action_type                            |  +12   |
| Wrong priority                                 |   −5   |
| Wrong category                                 |   −4   |
| Wrong action_type                              |   −6   |
| BONUS — correct escalation on urgent email     |   +5   |
| PENALTY — escalation on non-urgent email       |   −6   |

**Maximum per step (no bonus):** 30 points  
**Normalised score:** `total_reward / (steps × 30)` ∈ [0.0, 1.0]

---

## Task Descriptions

### `easy`
Spam detection and low-urgency system email classification.  
Sender signals are unambiguous; no conflicting cues.  
**Emails:** 4 (spam ×2, system maintenance, system backup)

### `medium`
Mixed real-world emails across billing, general, and employee senders.  
Requires contextual understanding but no urgent escalation pressure.  
**Emails:** 7 (boss casual, employee general, employee bug, customer billing ×2, boss billing, customer refund)

### `hard`
Urgent and conflicting-signal emails.  Boss directives, critical system failures,
urgent customer billing disputes, and employee escalation requests.  
Correct escalation decisions are critical to score well.  
**Emails:** 7 (empty edge case, system critical, customer billing urgent, customer bug urgent,
boss urgent, system error, employee urgent demo failure)

---

## Project Structure

```
.
├── email_triage_env.py   # Core Gym-style environment + dataset + baseline agent
├── graders.py            # Task definitions, TaskEnv, grade_episode()
├── inference.py          # OpenEnv inference runner (strict log format)
├── app.py                # FastAPI HF Space (/reset, /step, /state, /tasks)
├── openenv.yaml          # OpenEnv spec
├── Dockerfile            # Container build
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## Setup Instructions

**Python 3.10+, zero mandatory external dependencies for baseline mode.**

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full baseline evaluation (no API key needed)
python inference.py

# Run the original full-dataset evaluation
python email_triage_env.py

# Run the grader directly
python graders.py

# Serve the FastAPI app locally
uvicorn app:app --host 0.0.0.0 --port 7860
```

---

## How to Run Inference

### Baseline (rule-based, no API key)
```bash
python inference.py
```

### With an LLM agent
```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-key-here"
python inference.py
```

### Docker
```bash
docker build -t email-triage-env .
docker run --rm email-triage-env

# With LLM agent
docker run --rm \
  -e API_BASE_URL="https://api.openai.com/v1" \
  -e MODEL_NAME="gpt-4o-mini" \
  -e HF_TOKEN="your-key" \
  email-triage-env
```

---

## Example Output

```
[START]
task: easy

[STEP]
action: priority=0 category=3 action_type=0
reward: +30

[STEP]
action: priority=0 category=2 action_type=0
reward: +30

...

[END]
score: 1.00

[START]
task: hard
...
[END]
score: 0.71
```

---

## API Endpoints (HF Space)

| Method | Path     | Description                        |
|--------|----------|------------------------------------|
| POST   | `/reset` | Reset env for a task               |
| POST   | `/step`  | Submit action, get reward + info   |
| GET    | `/state` | Current session state              |
| GET    | `/tasks` | List available tasks               |
| GET    | `/health`| Liveness check                     |

---

## Why This Environment Is Useful

1. **REALISTIC TASK** — Email triage is a genuine workplace pain-point that every knowledge worker faces daily.
2. **MULTI-DIMENSIONAL DECISIONS** — The agent must simultaneously decide priority, category, and action, mirroring real-world compound decisions.
3. **EDGE CASES BUILT IN** — The dataset intentionally includes ambiguous signals that separate naive rule-followers from genuine decision-makers.
4. **SHAPED REWARD** — The reward function penalises over-escalation while incentivising correct urgency recognition.
5. **DETERMINISTIC & REPRODUCIBLE** — Fixed dataset, no randomness — every evaluation run is identical.
6. **ZERO DEPENDENCIES** — Core environment runs anywhere Python 3.7+ is available.
7. **EXTENSIBLE TO LLM AGENTS** — Structured observation + action format allows easy integration with LLM-based decision systems and tool-using agents.
8. **COMPATIBLE WITH RL BENCHMARKING** — Gymnasium-style API, directly usable with standard RL libraries and OpenEnv pipelines.
=======
---
title: Email Triage Ai
emoji: 😻
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
>>>>>>> 491dfcb5daf545b359bca7c3df51d98bff83d49d
