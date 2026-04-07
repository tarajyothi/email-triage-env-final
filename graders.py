"""
=============================================================================
  graders.py — Task Grader System
  OpenEnv Hackathon — Round 1 Submission
=============================================================================

Defines three task configurations (easy / medium / hard) and a grader that
scores an agent's episode results as a normalised float in [0.0, 1.0].
"""

from email_triage_env import EmailTriageEnv, baseline_agent, DATASET

# =============================================================================
# TASK DEFINITIONS
# =============================================================================

# Each task selects a deterministic slice of the full 18-email dataset.
# Indices are 0-based references into DATASET.
#
#   easy   — spam + low-urgency system emails: unambiguous signals only
#   medium — mixed real-world emails: billing, general, employee
#   hard   — urgent + conflicting signals: boss, system critical, customer crisis

TASKS = {
    "easy": {
        "description": "Classify obvious spam and low-urgency system emails.",
        "indices": [1, 7, 9, 16],   # spam×2, system maintenance, system backup
    },
    "medium": {
        "description": "Handle mixed real-world emails across billing, general, and employee senders.",
        "indices": [3, 5, 10, 11, 13, 14, 17],  # boss casual, employee general, employee bug,
                                                  # customer billing ×2, boss billing, customer refund
    },
    "hard": {
        "description": "Triage urgent and conflicting-signal emails requiring careful escalation.",
        "indices": [0, 2, 4, 6, 8, 12, 15],     # empty, system critical, customer billing urgent,
                                                  # customer bug urgent, boss urgent, system error, employee urgent
    },
}

# Maximum reward per step (all three components correct, no bonuses)
MAX_REWARD_PER_STEP = 10 + 8 + 12  # = 30


# =============================================================================
# TASK ENVIRONMENT
# =============================================================================

class TaskEnv(EmailTriageEnv):
    """
    Wraps EmailTriageEnv to run on a fixed task-specific subset of DATASET.

    Parameters
    ----------
    task_name : str
        One of "easy", "medium", "hard".
    """

    def __init__(self, task_name: str):
        super().__init__()
        if task_name not in TASKS:
            raise ValueError(f"Unknown task '{task_name}'. Choose from: {list(TASKS)}")
        self._task_name = task_name
        self._dataset   = [DATASET[i] for i in TASKS[task_name]["indices"]]
        self._max_steps = len(self._dataset)

    @property
    def task_name(self) -> str:
        return self._task_name


# =============================================================================
# GRADER
# =============================================================================

def grade_episode(step_results: list) -> float:
    """
    Compute a normalised score in [0.0, 1.0] for a completed episode.

    The maximum possible score assumes every step is answered correctly with
    no escalation bonuses (conservative baseline).  Scores above 1.0 are
    clamped, so escalation bonuses can only help, not inflate beyond the cap.

    Parameters
    ----------
    step_results : list[dict]
        Each dict must contain at least {"reward": int}.

    Returns
    -------
    float
        Normalised score ∈ [0.0, 1.0].
    """
    if not step_results:
        return 0.0

    total_reward  = sum(r["reward"] for r in step_results)
    max_possible  = len(step_results) * MAX_REWARD_PER_STEP

    # Clamp to [0.0, 1.0] — negative totals score 0, bonuses cannot exceed 1
    return max(0.0, min(1.0, total_reward / max_possible))


def run_task(task_name: str, agent_fn=None) -> dict:
    """
    Run one complete task episode and return graded results.

    Parameters
    ----------
    task_name : str
        One of "easy", "medium", "hard".
    agent_fn  : callable, optional
        agent_fn(observation: dict) -> action: dict.
        Defaults to baseline_agent if not provided.

    Returns
    -------
    dict
        {
          "task":         str,
          "score":        float,    # normalised ∈ [0.0, 1.0]
          "total_reward": int,
          "steps":        list[dict]
        }
    """
    if agent_fn is None:
        agent_fn = baseline_agent

    env         = TaskEnv(task_name)
    observation = env.reset()
    steps       = []

    while observation is not None:
        action = agent_fn(observation)
        next_obs, reward, done, info = env.step(action)
        steps.append({
            "observation": observation,
            "action":      action,
            "reward":      reward,
            "info":        info,
        })
        observation = next_obs

    score = grade_episode(steps)

    return {
        "task":         task_name,
        "score":        score,
        "total_reward": sum(s["reward"] for s in steps),
        "steps":        steps,
    }


# =============================================================================
# STANDALONE GRADER ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  OPENENV GRADER — ALL TASKS")
    print("=" * 60)

    for task in ["easy", "medium", "hard"]:
        result = run_task(task)
        print(f"\n  Task   : {result['task']}")
        print(f"  Reward : {result['total_reward']:+d}")
        print(f"  Score  : {result['score']:.4f}")

    print("\n" + "=" * 60)
