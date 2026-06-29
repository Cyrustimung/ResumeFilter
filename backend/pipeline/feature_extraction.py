"""
Feature Extraction: Parse raw candidate JSON into compact scoring features.

Transforms unstructured candidate records into a normalized feature dictionary
that downstream scorers consume. Handles title classification, career text
aggregation, skill normalization, and behavioural signal extraction.
"""

from datetime import date, datetime

from backend.config import TODAY, CONSULTING_FIRMS, AI_ML_TITLES, ADJACENT_TITLES, NON_TECH_TITLES


def _parse_date(date_str: str | None) -> date | None:
    """Parse a YYYY-MM-DD date string, returning None on failure."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _days_since(date_str: str | None) -> int:
    """Return the number of days between *date_str* and TODAY."""
    d = _parse_date(date_str)
    if not d:
        return 9999
    return max((TODAY - d).days, 0)


def _classify_title(title: str) -> str:
    """Classify a job title into AI_ML, ADJACENT, NON_TECH, or OTHER."""
    t = title.lower().strip()
    if t in AI_ML_TITLES:
        return "AI_ML"
    if t in ADJACENT_TITLES:
        return "ADJACENT"
    if t in NON_TECH_TITLES:
        return "NON_TECH"
    return "OTHER"


def _is_consulting(company: str, industry: str, size: str) -> bool:
    """Return True if the company is a consulting/IT-services firm."""
    if company.lower().strip() in CONSULTING_FIRMS:
        return True
    if industry == "IT Services" and size == "10001+":
        return True
    return False


def extract_features(candidate: dict) -> dict:
    """Convert a raw candidate JSON record into a compact feature dict for scoring."""
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    education = candidate.get("education", [])
    skills = candidate.get("skills", [])
    certs = candidate.get("certifications", [])
    signals = candidate.get("redrob_signals", {})

    # Profile
    years_exp: float = profile.get("years_of_experience", 0)
    current_title: str = profile.get("current_title", "Unknown")
    title_category: str = _classify_title(current_title)
    country: str = profile.get("country", "Unknown")
    location: str = profile.get("location", "Unknown")

    # Career
    total_career_months: int = sum(r.get("duration_months", 0) for r in career)
    product_months: int = 0
    consulting_months: int = 0
    career_texts: list[str] = []

    for role in career:
        dur = role.get("duration_months", 0)
        if _is_consulting(role.get("company", ""), role.get("industry", ""), role.get("company_size", "")):
            consulting_months += dur
        else:
            product_months += dur
        desc = role.get("description", "")
        if desc:
            career_texts.append(desc.lower())

    product_ratio: float = product_months / max(product_months + consulting_months, 1)
    tenures = [r.get("duration_months", 0) for r in career if r.get("duration_months", 0) > 0]
    avg_tenure: float = sum(tenures) / len(tenures) if tenures else 0

    role_titles = [r.get("title", "").lower() for r in career]
    ml_role_count: int = sum(1 for t in role_titles if _classify_title(t) == "AI_ML")
    has_recent_ml: bool = any(
        r.get("is_current", False) and _classify_title(r.get("title", "")) == "AI_ML"
        for r in career
    )

    combined_career_text: str = " ".join(career_texts)

    # Skills — normalize to lowercase for matching
    skill_list: list[dict] = [
        {
            "name": s.get("name", "").lower().strip(),
            "proficiency": s.get("proficiency", "beginner"),
            "endorsements": s.get("endorsements", 0),
            "duration_months": s.get("duration_months", 0),
        }
        for s in skills
    ]

    # Education
    edu_list: list[dict] = [
        {
            "field": e.get("field_of_study", "").lower().strip(),
            "tier": e.get("tier", "unknown"),
            "degree": e.get("degree", ""),
        }
        for e in education
    ]

    # Behavioral signals
    behavioral: dict = {
        "profile_completeness": signals.get("profile_completeness_score", 0),
        "last_active_days_ago": _days_since(signals.get("last_active_date")),
        "open_to_work": signals.get("open_to_work_flag", False),
        "response_rate": signals.get("recruiter_response_rate", 0),
        "avg_response_time_hours": signals.get("avg_response_time_hours", 999),
        "notice_period_days": signals.get("notice_period_days", 180),
        "interview_completion_rate": signals.get("interview_completion_rate", 0),
        "github_activity_score": signals.get("github_activity_score", -1),
        "willing_to_relocate": signals.get("willing_to_relocate", False),
        "verified_email": signals.get("verified_email", False),
        "verified_phone": signals.get("verified_phone", False),
        "linkedin_connected": signals.get("linkedin_connected", False),
        "skill_assessment_scores": signals.get("skill_assessment_scores", {}),
    }

    # Integrity raw data
    integrity_data: dict = {
        "years_exp": years_exp,
        "total_career_months": total_career_months,
        "career_history_raw": career,
        "skill_list_raw": skills,
    }

    return {
        "candidate_id": candidate.get("candidate_id", ""),
        "profile": {
            "name": profile.get("anonymized_name", ""),
            "headline": profile.get("headline", ""),
            "years_exp": years_exp,
            "current_title": current_title,
            "title_category": title_category,
            "country": country,
            "location": location,
        },
        "career": {
            "total_months": total_career_months,
            "product_ratio": product_ratio,
            "avg_tenure": avg_tenure,
            "ml_role_count": ml_role_count,
            "has_recent_ml": has_recent_ml,
            "combined_text": combined_career_text,
            "role_count": len(career),
        },
        "skills": skill_list,
        "education": edu_list,
        "certifications": certs,
        "behavioral": behavioral,
        "integrity_data": integrity_data,
    }
