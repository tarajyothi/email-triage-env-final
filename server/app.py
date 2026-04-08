from fastapi import FastAPI, Body
from email_triage_env import EmailTriageEnv

app = FastAPI()
env = EmailTriageEnv()

@app.get("/")
def root():
    return {"status": "ok"}

# 🔥 REQUIRED: body is MANDATORY using Body(...)
@app.post("/reset")
def reset(body: dict = Body(...)):
    obs = env.reset()
    return {
        "observation": obs,
        "reward": 0.0,
        "done": False,
        "info": {}
    }

@app.post("/step")
def step(action: dict = Body(...)):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs,
        "reward": float(reward),
        "done": bool(done),
        "info": info if isinstance(info, dict) else {}
    }

@app.get("/state")
def state():
    return {"state": "running"}
