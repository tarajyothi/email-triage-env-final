"""
=============================================================================
  Agentic Email Workflow RL Environment
  OpenEnv Hackathon — Round 1 Submission
  Author: OpenEnv Contributor
  File:   email_triage_env.py
=============================================================================

A Gymnasium-style Reinforcement Learning environment that simulates an email
triage assistant. An agent reads incoming emails and must decide:
  - Priority   (low / medium / high)
  - Category   (bug / billing / general / spam)
  - Action     (ignore / respond / escalate)

The environment is fully deterministic, requires no external libraries, and is
designed to be easily evaluated by automated graders.
"""

# =============================================================================
# SECTION 1 — SPACE DEFINITIONS
# =============================================================================

class Space:
    """Lightweight base class mimicking gymnasium.spaces.Space interface."""

    def contains(self, x):
        raise NotImplementedError


class DictSpace(Space):
    """
    Represents a space of dictionaries, where each key maps to its own sub-space.

    Parameters
    ----------
    spaces : dict[str, Space]
        Mapping of field names to their respective spaces.
    """

    def __init__(self, spaces: dict):
        self.spaces = spaces

    def contains(self, x: dict) -> bool:
        if not isinstance(x, dict):
            return False
        return all(
            key in x and space.contains(x[key])
            for key, space in self.spaces.items()
        )

    def __repr__(self):
        inner = ", ".join(f"{k}: {v}" for k, v in self.spaces.items())
        return f"DictSpace({{{inner}}})"


class DiscreteSpace(Space):
    """
    Represents a discrete set of integer values [0, n).

    Parameters
    ----------
    n : int
        Number of valid discrete values.
    """

    def __init__(self, n: int):
        self.n = n

    def contains(self, x) -> bool:
        return isinstance(x, int) and 0 <= x < self.n

    def __repr__(self):
        return f"Discrete({self.n})"


class TextSpace(Space):
    """Represents a space accepting any string value."""

    def contains(self, x) -> bool:
        return isinstance(x, str)

    def __repr__(self):
        return "Text()"


# =============================================================================
# SECTION 2 — DATASET  (fully deterministic, fixed order)
# =============================================================================

# Each record defines one email and its ground-truth labels.
# Fields:
#   text          — raw email body shown to the agent
#   sender_type   — one of ["boss", "customer", "employee", "spam", "system"]
#   urgency_flag  — 0 (normal) or 1 (urgent)
#   priority      — 0 (low), 1 (medium), 2 (high)
#   category      — 0 (bug), 1 (billing), 2 (general), 3 (spam)
#   action_type   — 0 (ignore), 1 (respond), 2 (escalate)

DATASET = [
    # ── 1. Edge case: empty email ────────────────────────────────────────────
    {
        "text": "",
        "sender_type": "system",
        "urgency_flag": 0,
        "priority": 0,
        "category": 2,
        "action_type": 0,
    },
    # ── 2. Obvious spam ──────────────────────────────────────────────────────
    {
        "text": "WIN MONEY NOW!!! Click here to claim your $10,000 prize!!!",
        "sender_type": "spam",
        "urgency_flag": 0,
        "priority": 0,
        "category": 3,
        "action_type": 0,
    },
    # ── 3. Urgent system failure ─────────────────────────────────────────────
    {
        "text": "CRITICAL: Production database is down. All services are unavailable. Immediate action required.",
        "sender_type": "system",
        "urgency_flag": 1,
        "priority": 2,
        "category": 0,
        "action_type": 2,
    },
    # ── 4. Boss — casual/low urgency (conflicting signal) ────────────────────
    {
        "text": "Hey, just checking in. No rush — let me know how the Q3 report is coming along when you get a chance!",
        "sender_type": "boss",
        "urgency_flag": 0,
        "priority": 1,
        "category": 2,
        "action_type": 1,
    },
    # ── 5. Customer billing complaint ────────────────────────────────────────
    {
        "text": "I was charged twice for my subscription this month. Please refund the duplicate charge immediately.",
        "sender_type": "customer",
        "urgency_flag": 1,
        "priority": 2,
        "category": 1,
        "action_type": 2,
    },
    # ── 6. Employee — routine general question ────────────────────────────────
    {
        "text": "Hi, could you remind me what day the all-hands meeting is scheduled for this month?",
        "sender_type": "employee",
        "urgency_flag": 0,
        "priority": 0,
        "category": 2,
        "action_type": 1,
    },
    # ── 7. Customer — bug report ─────────────────────────────────────────────
    {
        "text": "The login button on the mobile app is broken. I cannot access my account at all. This is urgent!",
        "sender_type": "customer",
        "urgency_flag": 1,
        "priority": 2,
        "category": 0,
        "action_type": 2,
    },
    # ── 8. System — routine health-check (low priority) ──────────────────────
    {
        "text": "Scheduled maintenance complete. All systems operational. No action needed.",
        "sender_type": "system",
        "urgency_flag": 0,
        "priority": 0,
        "category": 2,
        "action_type": 0,
    },
    # ── 9. Boss — urgent request ─────────────────────────────────────────────
    {
        "text": "I need the investor deck ready by 3 PM today. This is critical for the board meeting. Please escalate.",
        "sender_type": "boss",
        "urgency_flag": 1,
        "priority": 2,
        "category": 2,
        "action_type": 2,
    },
    # ── 10. Spam disguised as billing ────────────────────────────────────────
    {
        "text": "Your invoice is ready! Click here to update your payment info and avoid service interruption: spam-link.biz",
        "sender_type": "spam",
        "urgency_flag": 0,
        "priority": 0,
        "category": 3,
        "action_type": 0,
    },
    # ── 11. Employee — reports a bug ─────────────────────────────────────────
    {
        "text": "Found a bug in the export module — CSV exports are including extra blank rows. Not urgent but should be fixed.",
        "sender_type": "employee",
        "urgency_flag": 0,
        "priority": 1,
        "category": 0,
        "action_type": 1,
    },
    # ── 12. Customer — billing enquiry (non-urgent) ───────────────────────────
    {
        "text": "Could you explain what the 'platform fee' line item on my latest invoice is for?",
        "sender_type": "customer",
        "urgency_flag": 0,
        "priority": 1,
        "category": 1,
        "action_type": 1,
    },
    # ── 13. System — error alert (urgent) ────────────────────────────────────
    {
        "text": "ERROR: Payment gateway timeout. Transactions are failing. Immediate investigation required.",
        "sender_type": "system",
        "urgency_flag": 1,
        "priority": 2,
        "category": 0,
        "action_type": 2,
    },
    # ── 14. Boss — casual but billing-related (conflicting signal) ────────────
    {
        "text": "Hey, just noticed the team's SaaS subscription might be up for renewal soon. No rush — check when free.",
        "sender_type": "boss",
        "urgency_flag": 0,
        "priority": 1,
        "category": 1,
        "action_type": 1,
    },
    # ── 15. Customer — general feedback ──────────────────────────────────────
    {
        "text": "Love the new dashboard! The redesign is really clean and makes navigation much easier. Keep up the great work.",
        "sender_type": "customer",
        "urgency_flag": 0,
        "priority": 0,
        "category": 2,
        "action_type": 1,
    },
    # ── 16. Employee — urgent escalation request ──────────────────────────────
    {
        "text": "The client demo environment is completely down and the call starts in 20 minutes. Need help now!",
        "sender_type": "employee",
        "urgency_flag": 1,
        "priority": 2,
        "category": 0,
        "action_type": 2,
    },
    # ── 17. System — informational log (no action needed) ────────────────────
    {
        "text": "Daily backup completed successfully. 4,321 files backed up. Next backup scheduled for tomorrow.",
        "sender_type": "system",
        "urgency_flag": 0,
        "priority": 0,
        "category": 2,
        "action_type": 0,
    },
    # ── 18. Customer — refund request (billing, medium urgency) ──────────────
    {
        "text": "I cancelled my subscription 5 days ago but haven't received my refund yet. When can I expect it?",
        "sender_type": "customer",
        "urgency_flag": 0,
        "priority": 1,
        "category": 1,
        "action_type": 1,
    },
]


# =============================================================================
# SECTION 3 — ENVIRONMENT
# =============================================================================

class EmailTriageEnv:
    """
    Agentic Email Workflow RL Environment.

    Simulates an inbox-triage task where an RL agent reads emails and decides
    how to handle each one.  The episode runs through a fixed, deterministic
    dataset; no randomness is introduced at any stage.

    Observation Space
    -----------------
    Dict with:
        email_text   : str
        sender_type  : str  ∈ {"boss","customer","employee","spam","system"}
        urgency_flag : int  ∈ {0, 1}

    Action Space
    ------------
    Dict with:
        priority    : int  ∈ {0 (low), 1 (medium), 2 (high)}
        category    : int  ∈ {0 (bug), 1 (billing), 2 (general), 3 (spam)}
        action_type : int  ∈ {0 (ignore), 1 (respond), 2 (escalate)}

    Reward Structure
    ----------------
        +10  correct priority
        + 8  correct category
        +12  correct action_type
        - 5  wrong priority
        - 4  wrong category
        - 6  wrong action_type
        + 5  BONUS: escalated when urgency_flag == 1 and action correct
        - 6  PENALTY: escalated when urgency_flag == 0
    """

    # Human-readable label maps (for logging)
    PRIORITY_LABELS   = {0: "low", 1: "medium", 2: "high"}
    CATEGORY_LABELS   = {0: "bug", 1: "billing", 2: "general", 3: "spam"}
    ACTION_LABELS     = {0: "ignore", 1: "respond", 2: "escalate"}
    SENDER_TYPES      = ["boss", "customer", "employee", "spam", "system"]

    def __init__(self):
        """Initialise observation/action spaces and reset internal episode state."""
        # ── Observation space ──────────────────────────────────────────────
        self.observation_space = DictSpace({
            "email_text":   TextSpace(),
            "sender_type":  TextSpace(),
            "urgency_flag": DiscreteSpace(2),
        })

        # ── Action space ───────────────────────────────────────────────────
        self.action_space = DictSpace({
            "priority":    DiscreteSpace(3),
            "category":    DiscreteSpace(4),
            "action_type": DiscreteSpace(3),
        })

        self._dataset   = DATASET
        self._index     = 0
        self._max_steps = len(self._dataset)

    # ------------------------------------------------------------------
    def _validate_observation(self, observation: dict) -> None:
        """
        Validate that an observation conforms to observation_space.

        Parameters
        ----------
        observation : dict
            The observation dict to validate.

        Raises
        ------
        ValueError
            If the observation does not satisfy observation_space.contains().
        """
        if not self.observation_space.contains(observation):
            raise ValueError(
                f"Invalid observation produced by environment: {observation!r}\n"
                f"Expected structure: {self.observation_space}"
            )

    # ------------------------------------------------------------------
    def reset(self) -> dict:
        """
        Reset the environment to the beginning of the episode.

        Returns
        -------
        dict
            The first observation: {email_text, sender_type, urgency_flag}.
        """
        self._index = 0
        observation = self._make_observation(self._dataset[self._index])
        self._validate_observation(observation)
        return observation

    # ------------------------------------------------------------------
    def step(self, action: dict) -> tuple:
        """
        Apply an action to the current email and advance to the next one.

        Parameters
        ----------
        action : dict
            Must contain keys: "priority", "category", "action_type" (all int).

        Returns
        -------
        tuple : (next_state, reward, done, info)
            next_state : dict  — next email observation, or None if the episode ended
            reward     : int   — total reward earned for this step
            done       : bool  — True when the full dataset has been processed
            info       : dict  — ground truth labels, reward breakdown, difficulty, explanation
        """
        if not self.action_space.contains(action):
            raise ValueError(
                f"Invalid action: {action}. "
                f"Expected keys priority∈[0,2], category∈[0,3], action_type∈[0,2]."
            )

        if not all(k in action for k in ["priority", "category", "action_type"]):
            raise ValueError("Action missing required keys.")

        current = self._dataset[self._index]
        reward, explanation = self._compute_reward(action, current)

        info = {
            "ground_truth": {
                "priority":    current["priority"],
                "category":    current["category"],
                "action_type": current["action_type"],
            },
            "agent_action": action,
            "reward":       reward,
            "explanation":  explanation,
            "difficulty":   self._estimate_difficulty(current),
            "is_correct": (
                action["priority"]    == current["priority"] and
                action["category"]    == current["category"] and
                action["action_type"] == current["action_type"]
            ),
        }

        self._index += 1
        done = self._index >= self._max_steps

        if done:
            next_state = None
        else:
            next_state = self._make_observation(self._dataset[self._index])
            self._validate_observation(next_state)

        return next_state, reward, done, info

    # ------------------------------------------------------------------
    def _make_observation(self, record: dict) -> dict:
        """
        Build an observation dict from a dataset record.

        Parameters
        ----------
        record : dict
            A row from DATASET.

        Returns
        -------
        dict
            {email_text, sender_type, urgency_flag}
        """
        return {
            "email_text":   record["text"],
            "sender_type":  record["sender_type"],
            "urgency_flag": record["urgency_flag"],
        }

    # ------------------------------------------------------------------
    def _estimate_difficulty(self, record: dict) -> str:
        """
        Estimate the triage difficulty of a dataset record.

        Rules (in priority order):
          - Spam emails are always easy (sender signal is unambiguous).
          - Urgent emails from authoritative senders (system/boss) are hard.
          - Boss emails (any urgency) carry conflicting signals → medium.
          - All remaining cases default to medium.

        Parameters
        ----------
        record : dict
            A row from DATASET.

        Returns
        -------
        str
            One of "easy", "medium", or "hard".
        """
        if record["sender_type"] == "spam":
            return "easy"
        if record["urgency_flag"] == 1 and record["sender_type"] in ["system", "boss"]:
            return "hard"
        if record["sender_type"] == "boss":
            return "medium"
        return "medium"

    # ------------------------------------------------------------------
    def _compute_reward(self, action: dict, record: dict) -> tuple:
        """
        Calculate the reward for a single step.

        Parameters
        ----------
        action : dict
            Agent's chosen {priority, category, action_type}.
        record : dict
            Ground-truth dataset row.

        Returns
        -------
        tuple : (total_reward: int, explanation: list[str])
        """
        total       = 0
        explanation = []

        # ── Priority ───────────────────────────────────────────────────
        if action["priority"] == record["priority"]:
            total += 10
            explanation.append(
                f"[+10] Correct priority: "
                f"'{self.PRIORITY_LABELS[action['priority']]}'"
            )
        else:
            total -= 5
            explanation.append(
                f"[-5]  Wrong priority: "
                f"agent='{self.PRIORITY_LABELS[action['priority']]}' "
                f"expected='{self.PRIORITY_LABELS[record['priority']]}'"
            )

        # ── Category ───────────────────────────────────────────────────
        if action["category"] == record["category"]:
            total += 8
            explanation.append(
                f"[+8]  Correct category: "
                f"'{self.CATEGORY_LABELS[action['category']]}'"
            )
        else:
            total -= 4
            explanation.append(
                f"[-4]  Wrong category: "
                f"agent='{self.CATEGORY_LABELS[action['category']]}' "
                f"expected='{self.CATEGORY_LABELS[record['category']]}'"
            )

        # ── Action type ────────────────────────────────────────────────
        if action["action_type"] == record["action_type"]:
            total += 12
            explanation.append(
                f"[+12] Correct action_type: "
                f"'{self.ACTION_LABELS[action['action_type']]}'"
            )
        else:
            total -= 6
            explanation.append(
                f"[-6]  Wrong action_type: "
                f"agent='{self.ACTION_LABELS[action['action_type']]}' "
                f"expected='{self.ACTION_LABELS[record['action_type']]}'"
            )

        # ── Escalation bonus / penalty ─────────────────────────────────
        if action["action_type"] == 2:                   # agent escalated
            if record["urgency_flag"] == 1 and record["action_type"] == 2:
                total += 5
                explanation.append("[+5]  BONUS: Escalated correctly on urgent email.")
            elif record["urgency_flag"] == 0:
                total -= 6
                explanation.append("[-6]  PENALTY: Unnecessary escalation on non-urgent email.")

        return total, explanation


# =============================================================================
# SECTION 4 — GRADER FUNCTION
# =============================================================================

def evaluate_agent(env: EmailTriageEnv, agent_fn) -> int:
    """
    Run a complete episode and grade the agent's performance.

    Prints a per-step log (email preview, action taken, difficulty, reward,
    explanation) and a final summary (score, efficiency, correct decisions).

    Parameters
    ----------
    env      : EmailTriageEnv
        A freshly constructed (or reset) environment instance.
    agent_fn : callable
        A function  agent_fn(observation: dict) -> action: dict

    Returns
    -------
    int
        Total accumulated reward over the full episode.
    """
    observation      = env.reset()
    total_score      = 0
    step_number      = 0
    correct_steps    = 0
    difficulty_counts = {"easy": 0, "medium": 0, "hard": 0}
    separator        = "─" * 72

    print("\n" + "=" * 72)
    print("  AGENTIC EMAIL TRIAGE — EVALUATION RUN")
    print("=" * 72)

    while observation is not None:
        step_number += 1
        action = agent_fn(observation)
        next_observation, reward, done, info = env.step(action)

        total_score += reward
        if info["is_correct"]:
            correct_steps += 1
        difficulty_counts[info["difficulty"]] += 1

        # ── Per-step log ────────────────────────────────────────────────
        email_preview = (
            observation["email_text"][:80] + "…"
            if len(observation["email_text"]) > 80
            else (observation["email_text"] if observation["email_text"] else "[empty]")
        )

        print(f"\n{separator}")
        print(f"  Step {step_number:02d} | Sender: {observation['sender_type']:<10} "
              f"| Urgent: {observation['urgency_flag']}")
        print(f"  Email:  \"{email_preview}\"")
        print(f"  Action: priority={EmailTriageEnv.PRIORITY_LABELS[action['priority']]}, "
              f"category={EmailTriageEnv.CATEGORY_LABELS[action['category']]}, "
              f"action_type={EmailTriageEnv.ACTION_LABELS[action['action_type']]}")
        print(f"  Difficulty: {info['difficulty']}")
        print(f"  Reward: {reward:+d}")
        print(f"  Explanation:")
        for line in info["explanation"]:
            print(f"    {line}")
        print(f"  Running Total: {total_score:+d}")

        observation = next_observation

    print(f"\n{separator}")
    print(f"\n  ✅  FINAL SCORE: {total_score:+d}  (over {step_number} emails)")
    max_possible = step_number * (10 + 8 + 12)          # all correct, no bonuses
    print(f"  📊  Max Possible (no bonuses): {max_possible}")
    print(f"  📈  Efficiency: {100 * total_score / max_possible:.1f}%")
    print(f"  🎯  Correct Decisions: {correct_steps} / {step_number}")
    print(f"  🧩  Difficulty Breakdown: {difficulty_counts}")
    print("  🧠  Insight: High scores require correct escalation decisions, not just classification.")
    print("\n" + "=" * 72)
    print("  ✅ Evaluation complete. Environment behaving deterministically.")
    print("=" * 72 + "\n")

    return total_score


# =============================================================================
# SECTION 5 — BASELINE AGENT
# =============================================================================

def baseline_agent(observation: dict) -> dict:
    """
    Simple rule-based agent for benchmarking.

    Rules (in priority order):
    1. If sender_type is "spam"                      → low priority, spam category, ignore.
    2. If email contains billing keywords
       ["refund", "invoice", "payment", "charge"]    → medium priority, billing category, respond.
    3. If urgency_flag == 1 OR
       email contains crisis keywords                → high priority, general/bug category, escalate.
    4. Otherwise                                     → medium priority, general category, respond.

    Parameters
    ----------
    observation : dict
        {email_text: str, sender_type: str, urgency_flag: int}

    Returns
    -------
    dict
        {priority: int, category: int, action_type: int}
    """
    email_text  = observation["email_text"].lower()
    sender_type = observation["sender_type"]
    urgency     = observation["urgency_flag"]

    # Rule 1 — Spam sender
    if sender_type == "spam":
        return {"priority": 0, "category": 3, "action_type": 0}

    # Rule 2 — Billing keywords (checked before crisis to avoid mis-escalation)
    billing_keywords = {"refund", "invoice", "payment", "charge"}
    if any(kw in email_text for kw in billing_keywords):
        return {"priority": 1, "category": 1, "action_type": 1}

    # Rule 3 — Crisis keywords or urgency flag
    crisis_keywords = {"urgent", "down", "error", "critical", "fail", "broken",
                       "immediate", "unavailable", "crash", "timeout"}
    has_crisis_keyword = any(kw in email_text for kw in crisis_keywords)

    if urgency == 1 or has_crisis_keyword:
        # Resolve category: billing takes precedence over technical, else general
        tech_keywords    = {"database", "server", "gateway", "login", "app",
                            "export", "module", "payment", "transaction", "demo"}
        billing_keywords = {"refund", "invoice", "payment", "charge"}

        if any(kw in email_text for kw in billing_keywords):
            category = 1
        elif any(kw in email_text for kw in tech_keywords):
            category = 0
        else:
            category = 2

        return {"priority": 2, "category": category, "action_type": 2}

    # Rule 4 — Default: medium priority, respond
    return {"priority": 1, "category": 2, "action_type": 1}


# =============================================================================
# SECTION 6 — ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    env   = EmailTriageEnv()
    score = evaluate_agent(env, baseline_agent)


# =============================================================================
# README
# =============================================================================
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  README — Agentic Email Workflow RL Environment
  OpenEnv Hackathon — Round 1 Submission
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This environment simulates structured decision-making in AI agents using
multi-dimensional actions under deterministic evaluation.

PROBLEM STATEMENT
─────────────────
Email overload is one of the most persistent productivity problems in
organisations.  An intelligent triage agent that can correctly classify,
prioritise, and route emails could save significant human time and prevent
critical issues from being missed.

This environment provides a structured, evaluable sandbox for training and
testing such an agent using Reinforcement Learning.

ENVIRONMENT DESIGN
──────────────────
  State (Observation)
  ┌──────────────┬──────────────────────────────────────────────────┐
  │ Field        │ Values                                           │
  ├──────────────┼──────────────────────────────────────────────────┤
  │ email_text   │ raw string (may be empty)                        │
  │ sender_type  │ "boss" | "customer" | "employee" | "spam" |      │
  │              │ "system"                                         │
  │ urgency_flag │ 0 (normal) | 1 (urgent)                         │
  └──────────────┴──────────────────────────────────────────────────┘

  Action
  ┌─────────────┬──────────────────────────────────────────────────┐
  │ Key         │ Values                                           │
  ├─────────────┼──────────────────────────────────────────────────┤
  │ priority    │ 0 (low) | 1 (medium) | 2 (high)                 │
  │ category    │ 0 (bug) | 1 (billing) | 2 (general) | 3 (spam)  │
  │ action_type │ 0 (ignore) | 1 (respond) | 2 (escalate)         │
  └─────────────┴──────────────────────────────────────────────────┘

  Reward Function
  ┌───────────────────────────────────────────────────┬────────┐
  │ Event                                             │ Reward │
  ├───────────────────────────────────────────────────┼────────┤
  │ Correct priority                                  │  +10   │
  │ Correct category                                  │   +8   │
  │ Correct action_type                               │  +12   │
  │ Wrong priority                                    │   -5   │
  │ Wrong category                                    │   -4   │
  │ Wrong action_type                                 │   -6   │
  │ BONUS: correct escalation when urgency_flag=1     │   +5   │
  │ PENALTY: unnecessary escalation (urgency_flag=0)  │   -6   │
  └───────────────────────────────────────────────────┴────────┘

  Maximum score per step (all correct, no bonus): 30
  Maximum bonus per urgent email:                 +5

DATASET DESCRIPTION
───────────────────
  18 fixed emails, no randomness.  Covers:
  • Edge case: empty email body
  • Obvious spam with prize scam
  • Critical system failure (urgent)
  • Boss sending a casual, low-urgency note (conflicting signal)
  • Boss sending an urgent deadline request (conflicting signal)
  • Customer billing complaints (urgent and non-urgent)
  • Customer bug reports
  • Customer general feedback
  • Employee general enquiry
  • Employee bug report (non-urgent)
  • Employee urgent escalation
  • System health-check (no action needed)
  • System error alerts (urgent)
  • System routine backup log
  • Phishing spam disguised as billing

HOW TO RUN
──────────
  Python 3.7+, zero external dependencies.

  $ python email_triage_env.py

EXAMPLE USAGE
─────────────
  from email_triage_env import EmailTriageEnv, evaluate_agent, baseline_agent

  env   = EmailTriageEnv()
  score = evaluate_agent(env, baseline_agent)

  # Or step manually:
  obs = env.reset()
  while obs is not None:
      action = baseline_agent(obs)
      obs, reward, done, info = env.step(action)
      print(reward, info["explanation"])

  # Plug in your own agent:
  def my_agent(obs):
      return {"priority": 1, "category": 2, "action_type": 1}

  evaluate_agent(EmailTriageEnv(), my_agent)

EVALUATION METHOD
─────────────────
  evaluate_agent(env, agent_fn) runs a complete episode, prints per-step
  logs (email preview, action taken, reward, explanation), and returns the
  total accumulated reward.  The function also prints:
    • Running total at each step
    • Final score
    • Maximum possible score (no bonuses)
    • Efficiency percentage

WHY THIS ENVIRONMENT IS USEFUL
────────────────────────────────
  1. REALISTIC TASK — Email triage is a genuine workplace pain-point that
     every knowledge worker faces daily.

  2. MULTI-DIMENSIONAL DECISIONS — The agent must simultaneously decide
     priority, category, and action, mirroring real-world compound decisions.

  3. EDGE CASES BUILT IN — The dataset intentionally includes ambiguous
     signals (casual boss emails, urgent-looking spam) that separate naive
     rule-followers from genuine decision-makers.

  4. SHAPED REWARD — The reward function penalises over-escalation (costly
     in practice) while incentivising correct urgency recognition, creating
     a non-trivial optimisation landscape.

  5. DETERMINISTIC & REPRODUCIBLE — Fixed dataset and no randomness mean
     every evaluation run is identical, enabling fair comparison between
     agents.

  6. ZERO DEPENDENCIES — Runs anywhere Python 3.7+ is available, removing
     all setup friction for evaluators and contributors.

  7. EXTENSIBLE TO LLM AGENTS — The structured observation + action format
     allows easy integration with LLM-based decision systems and tool-using agents.

  8. COMPATIBLE WITH RL BENCHMARKING — The environment follows a Gymnasium-style API,
     making it directly usable with standard RL libraries and OpenEnv pipelines.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
