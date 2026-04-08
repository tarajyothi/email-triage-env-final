from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Dict

app = FastAPI()

TASKS = ["easy", "medium", "hard"]

_session: Dict = {
    "env": None,
    "task": None,
    "observation": None,
    "step_count": 0,
    "done": False
}

class TaskEnv:
    def __init__(self, task):
        self.task = task

    def reset(self):
        return {"message": f"Task '{self.task}' started"}

    def step(self, action):
        return {
            "observation": {"msg": "step done"},
            "reward": 1,
            "done": False,
            "info": {}
        }

class StepRequest(BaseModel):
    priority: int
    category: int
    action_type: int


@app.get("/")
def home():
    return {"message": "Email Triage API is running 🚀"}


@app.post("/reset")
async def reset(request: Request):
    try:
        body = await request.json()
    except:
        body = {}

    task = body.get("task", "easy")

    if task not in TASKS:
        task = "easy"

    env = TaskEnv(task)
    observation = env.reset()

    _session["env"] = env
    _session["task"] = task
    _session["observation"] = observation
    _session["step_count"] = 0
    _session["done"] = False

    return observation


@app.post("/step")
def step(request: StepRequest):
    if _session["env"] is None:
        raise HTTPException(status_code=400, detail="Call /reset first")

    if _session["done"]:
        raise HTTPException(status_code=400, detail="Episode complete")

    action = {
        "priority": request.priority,
        "category": request.category,
        "action_type": request.action_type
    }

    result = _session["env"].step(action)

    _session["step_count"] += 1
    _session["done"] = result["done"]

    return result
