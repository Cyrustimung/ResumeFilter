"""
Stage 2: Reliability & Availability Scorer (20% of final score).

Evaluates behavioural signals such as response rate, activity recency,
notice period, and location fit to determine hiring feasibility.
"""

from backend.config import PREFERRED_CITIES, ACCEPTABLE_CITIES, RELIABILITY_WEIGHTS


def compute_reliability_score(features: dict) -> tuple[float, dict]:
    """Compute reliability score from behavioural and location features."""
    b: dict = features["behavioral"]
    p: dict = features["profile"]
    w = RELIABILITY_WEIGHTS

    # Response rate
    rr: float = b["response_rate"]
    response: float = 1.0 if rr >= 0.7 else 0.7 if rr >= 0.4 else 0.4 if rr >= 0.2 else 0.2 if rr >= 0.1 else 0.05

    # Activity recency
    days: int = b["last_active_days_ago"]
    recency: float = 1.0 if days <= 30 else 0.8 if days <= 90 else 0.5 if days <= 180 else 0.2 if days <= 365 else 0.05

    # Notice period
    np_days: int = b["notice_period_days"]
    notice: float = 1.0 if np_days <= 30 else 0.7 if np_days <= 60 else 0.4 if np_days <= 90 else 0.2

    # Location match
    loc: str = p["location"].lower()
    country: str = p["country"].lower()
    relocate: bool = b["willing_to_relocate"]
    if any(c in loc for c in PREFERRED_CITIES):
        location: float = 1.0
    elif any(c in loc for c in ACCEPTABLE_CITIES):
        location = 0.8
    elif country == "india":
        location = 0.6 if relocate else 0.4
    else:
        location = 0.3 if relocate else 0.1

    # Open to work
    open_score: float = 1.0 if b["open_to_work"] else 0.4

    # Interview completion
    interview: float = b["interview_completion_rate"]

    # Response time
    hrs: float = b["avg_response_time_hours"]
    time_score: float = 1.0 if hrs <= 24 else 0.7 if hrs <= 72 else 0.4 if hrs <= 168 else 0.2

    score: float = (
        w["response_rate"] * response +
        w["recency"] * recency +
        w["notice_period"] * notice +
        w["location"] * location +
        w["open_to_work"] * open_score +
        w["interview_completion"] * interview +
        w["response_time"] * time_score
    )

    return score, {
        "response_rate": round(response, 3),
        "recency": round(recency, 3),
        "notice_period": round(notice, 3),
        "location": round(location, 3),
        "open_to_work": round(open_score, 3),
        "interview_completion": round(interview, 3),
        "response_time": round(time_score, 3),
    }
