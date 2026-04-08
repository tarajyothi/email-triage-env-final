from fastapi import FastAPI
from email_triage_env import EmailTriageEnv

app = FastAPI()

env = EmailTriageEnv()

@app.get("/")
def root():
    return {"status": "ok"}

# ✅ FIXED: accept body + return full structure
@app.post("/reset")
def reset(body: dict = {}):
    obs = env.reset()
    return {
        "observation
