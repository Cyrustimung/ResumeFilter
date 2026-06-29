"""
Stage 3: Growth & Credibility Scorer (15% of final score).

Evaluates external validation signals: GitHub activity, platform assessments,
education quality, career velocity, and profile verification.
"""

from backend.config import (
    CREDIBILITY_WEIGHTS, RELEVANT_FIELDS, TIER_WEIGHTS, DEGREE_WEIGHTS,
    CORE_SKILLS, SECONDARY_SKILLS,
)


def compute_credibility_score(features: dict) -> tuple[float, dict]:
    """Compute credibility score from education, GitHub, assessments, and verification."""
    b: dict = features["behavioral"]
    edu: list[dict] = features["education"]
    career: dict = features["career"]
    w = CREDIBILITY_WEIGHTS

    # GitHub
    gh: float = b["github_activity_score"]
    github: float = 0.3 if gh < 0 else 1.0 if gh >= 70 else 0.7 if gh >= 40 else 0.5 if gh >= 10 else 0.35

    # Assessments
    scores: dict = b["skill_assessment_scores"]
    if scores:
        relevant = [v for k, v in scores.items() if k.lower() in CORE_SKILLS or k.lower() in SECONDARY_SKILLS]
        other = [v for k, v in scores.items() if k.lower() not in CORE_SKILLS and k.lower() not in SECONDARY_SKILLS]
        if relevant:
            assessment: float = sum(relevant) / (len(relevant) * 100)
        elif other:
            assessment = sum(other) / (len(other) * 100) * 0.5
        else:
            assessment = 0.3
    else:
        assessment = 0.3

    # Education
    best_edu: float = 0.2
    for e in edu:
        field_match: float = 1.0 if e["field"] in RELEVANT_FIELDS else 0.3
        tier: float = TIER_WEIGHTS.get(e["tier"], 0.4)
        degree: float = DEGREE_WEIGHTS.get(e["degree"], 0.4)
        score: float = 0.5 * field_match + 0.3 * tier + 0.2 * degree
        best_edu = max(best_edu, score)

    # Career velocity
    avg: float = career["avg_tenure"]
    velocity: float = 1.0 if avg >= 30 else 0.8 if avg >= 24 else 0.5 if avg >= 18 else 0.2

    # Verification
    verified: int = sum([b["verified_email"], b["verified_phone"], b["linkedin_connected"]])
    completeness: float = b["profile_completeness"] / 100.0
    verification: float = (verified / 3.0) * 0.5 + completeness * 0.5

    final_score: float = (
        w["github"] * github +
        w["assessments"] * assessment +
        w["education"] * best_edu +
        w["career_velocity"] * velocity +
        w["verification"] * verification
    )

    return final_score, {
        "github": round(github, 3),
        "assessments": round(assessment, 3),
        "education": round(best_edu, 3),
        "career_velocity": round(velocity, 3),
        "verification": round(verification, 3),
    }
