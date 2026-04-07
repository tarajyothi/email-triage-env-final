"""
=============================================================================
  app.py — Hugging Face Space API
  OpenEnv Hackathon — Round 1 Submission
=============================================================================
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

# ✅ ADD THIS (HOME ROUTE)
@app.get("/")
def home():
    return {"message": "Email Triage API is running 🚀"}


# =============================================================================
# SESSION STORE
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
    priority:    int = Field(..., ge=0, le=2)
    category:    int = Field(..., ge=0, le=3)
    action_type: int = Field(..., ge=0, le=2)


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
    return {"status": "ok"}


@app.get("/tasks")
def list_tasks():
    return {
        name: {"description": cfg["description"], "num_emails": len(cfg["indices"])}
        for name, cfg in TASKS.items()
    }


@app.post("/reset", response_model=ObservationResponse)
def reset(request: ResetRequest):
    if request.task not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task '{request.task}'")

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
    if _session["env"] is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    if _session["done"]:
        raise HTTPException(status_code=400, detail="Episode complete")

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
    return {
        "task":        _session["task"],
        "step_count":  _session["step_count"],
        "done":        _session["done"],
        "observation": _session["observation"],
    }