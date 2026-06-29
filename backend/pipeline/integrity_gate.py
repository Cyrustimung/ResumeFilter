"""
Stage 0: Integrity Gate — Honeypot & Manipulation Detection.

Detects timeline fabrication, skill inflation, and impossible role overlaps.
Returns a multiplicative gate value (1.0 = clean, 0.3 = suspect, 0.0 = honeypot).
"""

from datetime import date, datetime

from backend.config import INTEGRITY, TODAY


def _parse_date(date_str: str | None) -> date | None:
    """Parse a YYYY-MM-DD date string, returning None on failure."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _check_timeline(data: dict) -> bool:
    """Returns True if timeline is SUSPICIOUS."""
    claimed = data["years_exp"] * 12
    actual = data["total_career_months"]
    if claimed > actual + INTEGRITY["timeline_tolerance_months"]:
        return True

    for role in data["career_history_raw"]:
        start = _parse_date(role.get("start_date"))
        end = _parse_date(role.get("end_date"))
        stated = role.get("duration_months", 0)
        if start and end:
            real = (end.year - start.year) * 12 + (end.month - start.month)
            if stated > real + 3:
                return True
        if start and start > TODAY and stated > 6:
            return True
    return False


def _check_skill_inflation(data: dict) -> bool:
    """Returns True if skills are suspiciously inflated."""
    inflation: int = 0
    high_zero: int = 0
    for s in data["skill_list_raw"]:
        prof = s.get("proficiency", "beginner")
        dur = s.get("duration_months", 0)
        endorse = s.get("endorsements", 0)
        if prof in ("expert", "advanced") and dur < INTEGRITY["min_duration_for_advanced"]:
            inflation += 1
        if prof in ("expert", "advanced") and endorse == 0:
            high_zero += 1

    if inflation > INTEGRITY["skill_inflation_threshold"]:
        return True
    if high_zero > INTEGRITY["high_prof_zero_endorse_threshold"]:
        return True
    return False


def _check_overlapping_dates(data: dict) -> bool:
    """Returns True if roles overlap impossibly."""
    tolerance: int = INTEGRITY["overlap_tolerance_months"]
    roles: list[tuple[date, date | None]] = []
    for r in data["career_history_raw"]:
        s = _parse_date(r.get("start_date"))
        e = _parse_date(r.get("end_date"))
        if s:
            roles.append((s, e))

    for i in range(len(roles)):
        for j in range(i + 1, len(roles)):
            sa, ea = roles[i]
            sb, eb = roles[j]
            ea_eff = ea or TODAY
            eb_eff = eb or TODAY
            overlap_start = max(sa, sb)
            overlap_end = min(ea_eff, eb_eff)
            if overlap_start < overlap_end:
                months = (overlap_end.year - overlap_start.year) * 12 + (overlap_end.month - overlap_start.month)
                if months > tolerance:
                    return True
    return False


def compute_integrity_gate(features: dict) -> float:
    """
    Compute integrity gate score for a candidate.

    Returns:
        1.0 = clean
        0.3 = suspect (one flag)
        0.0 = honeypot (2+ flags)
    """
    data = features["integrity_data"]
    flags: int = 0
    if _check_timeline(data):
        flags += 1
    if _check_skill_inflation(data):
        flags += 1
    if _check_overlapping_dates(data):
        flags += 1

    # Domain relevance gate: non-tech title + zero ML evidence
    # Instead of zeroing completely, return a heavy penalty (0.3) for non-tech no-ML
    # The scoring caps in fusion.py handle the fine-grained ceiling.
    # Only zero-out if BOTH domain mismatch AND integrity flags exist.
    if features["profile"]["title_category"] == "NON_TECH":
        career_text = features["career"]["combined_text"]
        ml_keywords = ["machine learning", "deep learning", "neural",
                       "embedding", "model training", "model serving",
                       "training pipeline", "inference", "nlp",
                       "retrieval", "search ranking", "ranking model",
                       "recommendation", "feature engineering",
                       "ml pipeline", "prediction", "classification",
                       "xgboost", "sklearn", "pytorch", "tensorflow"]
        has_any_ml = any(kw in career_text for kw in ml_keywords)
        if not has_any_ml:
            # Non-tech + no ML: severely penalize but let scoring caps do final work
            if flags >= 1:
                return 0.0  # Non-tech + no ML + integrity issue = honeypot
            return 0.3  # Non-tech + no ML = heavy penalty but not full rejection

    if flags >= 2:
        return 0.0
    if flags == 1:
        return 0.3
    return 1.0
