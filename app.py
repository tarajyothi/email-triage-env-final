"""
=============================================================================
  app.py — Hugging Face Space API
  OpenEnv Hackathon — Round 1 Submission
=============================================================================

Minimal FastAPI application exposing /reset and /step endpoints.
Designed for deployment as a Hugging Face Space (Docker SDK).

Endpoints:
    POST /reset        — reset the environment for a given task
    POST /step         — submit one action, receive observation + reward
    GET  /state        — return current environment state
    GET  /tasks        — list available tasks
    GET  /health       — liveness check
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from graders import TaskEnv, TASKS
from email_triage_env import EmailTriageEnv

app = FastAPI(
    title="Agentic Email Triage — OpenEnv API",
    description="RL environment for email triage. Supports easy / medium / hard tasks.",
    version="1.0.0",
)

# =============================================================================
# SESSION STORE (single-session; stateless per request otherwise)
# =============================================================================

_session: dict = {
    "env":         None,
    "task":        None,
    "observation": None,
    "step_count":  0,
    "done":        False,
}


# =============================================================================
# PYDANTIC SCHEMAS
# =============================================================================

class ResetRequest(BaseModel):
    task: str = Field("easy", description="One of: easy, medium, hard")


class StepRequest(BaseModel):
    priority:    int = Field(..., ge=0, le=2, description="0=low, 1=medium, 2=high")
    category:    int = Field(..., ge=0, le=3, description="0=bug, 1=billing, 2=general, 3=spam")
    action_type: int = Field(..., ge=0, le=2, description="0=ignore, 1=respond, 2=escalate")


class ObservationResponse(BaseModel):
    email_text:   str
    sender_type:  str
    urgency_flag: int


class StepResponse(BaseModel):
    observation: Optional[ObservationResponse]
    reward:      int
    done:        bool
    difficulty:  str
    is_correct:  bool
    explanation: list[str]


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/health")
def health():
    """Liveness check."""
    return {"status": "ok"}


@app.get("/tasks")
def list_tasks():
    """Return available task names and descriptions."""
    return {
        name: {"description": cfg["description"], "num_emails": len(cfg["indices"])}
        for name, cfg in TASKS.items()
    }


@app.post("/reset", response_model=ObservationResponse)
def reset(request: ResetRequest):
    """
    Reset the environment for the specified task.
    Returns the first observation.
    """
    if request.task not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task '{request.task}'. Choose from: {list(TASKS)}")

    env         = TaskEnv(request.task)
    observation = env.reset()

    _session["env"]         = env
    _session["task"]        = request.task
    _session["observation"] = observation
    _session["step_count"]  = 0
    _session["done"]        = False

    return ObservationResponse(**observation)


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    """
    Submit one action for the current email.
    Returns next observation, reward, done flag, and explanation.
    """
    if _session["env"] is None:
        raise HTTPException(status_code=400, detail="Environment not initialised. Call /reset first.")
    if _session["done"]:
        raise HTTPException(status_code=400, detail="Episode is complete. Call /reset to start a new one.")

    action = {
        "priority":    request.priority,
        "category":    request.category,
        "action_type": request.action_type,
    }

    env = _session["env"]
    try:
        next_obs, reward, done, info = env.step(action)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    _session["observation"] = next_obs
    _session["step_count"] += 1
    _session["done"]        = done

    return StepResponse(
        observation=ObservationResponse(**next_obs) if next_obs else None,
        reward=reward,
        done=done,
        difficulty=info["difficulty"],
        is_correct=info["is_correct"],
        explanation=info["explanation"],
    )


@app.get("/state")
def state():
    """Return the current session state."""
    obs = _session["observation"]
    return {
        "task":        _session["task"],
        "step_count":  _session["step_count"],
        "done":        _session["done"],
        "observation": obs,
    }
