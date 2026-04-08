from fastapi import FastAPI, Body
from email_triage_env import EmailTriageEnv

app = FastAPI()

env = EmailTriageEnv()


@app.get("/")
def root():
    return {"status": "ok"}


# ✅ IMPORTANT: body must be OPTIONAL
@app.post("/reset")
def reset(body: dict = Body(default={})):
    try:
        obs = env.reset()
    except Exception:
        obs = {}

    return {
        "observation": obs if obs is not None else {},
        "reward": 0.0,
        "done": False,
        "info": {}
    }


# ✅ IMPORTANT: body must be OPTIONAL
@app.post("/step")
def step(action: dict = Body(default={})):
    try:
        obs, reward, done, info = env.step(action)
    except Exception:
        obs, reward, done, info = {}, 0.0, False, {}

    return {
        "observation": obs if obs is not None else {},
        "reward": float(reward),
        "done": bool(done),
        "info": info if isinstance(info, dict) else {}
    }


@app.get("/state")
def state():
    return {"state": "running"}
