FROM python:3.10-slim

# ── metadata ──────────────────────────────────────────────────────────────
LABEL maintainer="OpenEnv Contributor"
LABEL description="Agentic Email Triage RL Environment — OpenEnv Hackathon"

# ── system setup ──────────────────────────────────────────────────────────
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── dependencies ──────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── application code ──────────────────────────────────────────────────────
COPY email_triage_env.py .
COPY graders.py           .
COPY inference.py         .
COPY app.py               .
COPY openenv.yaml         .

# ── environment variables (overridable at runtime) ────────────────────────
ENV API_BASE_URL=""
ENV MODEL_NAME="gpt-4o-mini"
ENV HF_TOKEN=""

# ── default command: run inference ────────────────────────────────────────
CMD ["python", "inference.py"]

# ── HF Space alternative: serve the FastAPI app ───────────────────────────
# To deploy as a Hugging Face Space, override CMD at runtime:
#   CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
