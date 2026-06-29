"""
Stage 1: Technical Fit Scorer (65% of final score).

Key improvements over naive keyword matching:
- Hard requirement gating: missing ML experience = multiplicative penalty
- Skill depth: 6 months vs 36 months MATTER
- Negative scoring: domain mismatch punishes, not just rewards
- Evidence over claims: career descriptions prove, skill lists just claim
- Job-specific scoring: we rank for Senior AI Engineer, not generic profiles
- TF-IDF similarity: semantic matching between career text and JD
- Fuzzy skill matching: catches skill name variations (sklearn ↔ scikit-learn)
"""

import math

from backend.config import (
    EVIDENCE_KEYWORDS, CORE_EVIDENCE_CATEGORIES, NICE_EVIDENCE_CATEGORIES,
    CORE_SKILLS, SECONDARY_SKILLS, SKILL_TRUST_WEIGHTS, SOFT_PENALTIES,
    HIGH_VALUE_AI_SKILLS, HARD_REQUIREMENT_KEYWORDS, HARD_PENALTIES,
    JD,
)
from backend.pipeline.text_similarity import compute_tfidf_similarity
from backend.pipeline.skill_matching import match_skills_to_jd, find_skill_in_text


# Hard requirement check — missing ML = hard penalty

# Stricter ML evidence keywords — "model" alone is too broad (matches business models, etc.)
_STRICT_ML_KEYWORDS: list[str] = [
    "machine learning", "deep learning", "neural network", "model training",
    "model serving", "model deployment", "inference", "prediction",
    "classification", "regression", "embedding", "nlp",
    "natural language processing", "computer vision", "recommendation",
    "ranking model", "search ranking", "retrieval", "feature engineering",
    "training pipeline", "ml pipeline", "ml model", "ai system",
    "xgboost", "sklearn", "scikit", "pytorch", "tensorflow",
    "fine-tuning", "fine tuning", "transformer", "bert", "gpt",
    "vector search", "semantic search", "a/b test",
]


def _has_genuine_ml_evidence(career_text: str) -> bool:
    """
    Check for genuine ML evidence using strict keywords.
    Avoids false positives from generic words like 'model' or 'ai' appearing
    in non-technical contexts (e.g. "AI-strategy advisory").
    """
    return any(kw in career_text for kw in _STRICT_ML_KEYWORDS)


def _check_hard_requirements(features: dict) -> tuple[float, list[str]]:
    """
    Check if candidate meets hard job requirements.
    Returns a multiplier (0.1 to 1.0) and a list of failed requirements.
    """
    career_text: str = features["career"]["combined_text"]
    profile: dict = features["profile"]
    years: float = profile["years_exp"]
    title_cat: str = profile["title_category"]

    failed: list[str] = []
    multiplier: float = 1.0

    # Check 1: Has GENUINE ML evidence anywhere in career? (strict keywords)
    has_ml = _has_genuine_ml_evidence(career_text)

    # Check 1b: For ADJACENT tech titles, also consider strong AI skill claims
    # This prevents penalizing transitioning engineers who have real skills
    # but whose job descriptions don't explicitly mention ML yet
    has_strong_ai_skills = False
    if not has_ml and title_cat == "ADJACENT":
        skill_names = {s["name"] for s in features["skills"]}
        ai_skill_overlap = skill_names & {s.lower() for s in HIGH_VALUE_AI_SKILLS}
        # Need at least 2 high-value AI skills with reasonable duration
        strong_ai_skills = [
            s for s in features["skills"]
            if s["name"] in {sk.lower() for sk in HIGH_VALUE_AI_SKILLS}
            and s["duration_months"] >= 12
        ]
        if len(strong_ai_skills) >= 2:
            has_strong_ai_skills = True

    if not has_ml and not has_strong_ai_skills:
        failed.append("ZERO_ML_EVIDENCE: No machine learning evidence found in any career description")
        if title_cat == "NON_TECH":
            multiplier = min(multiplier, HARD_PENALTIES["non_tech_title_zero_ml"])
        else:
            multiplier = min(multiplier, HARD_PENALTIES["zero_ml_evidence_in_career"])

    # Check 2: Severe underexperience (JD says 5-9 years)
    if years < 3:
        failed.append(f"UNDEREXPERIENCED: Only {years:.1f} years (JD requires 5-9)")
        multiplier = min(multiplier, HARD_PENALTIES["under_3_years_experience"])

    # Check 3: Non-tech title with zero ML is near-disqualifier
    # Now also checks for genuine ML, not just broad keyword match
    if title_cat == "NON_TECH" and not has_ml:
        failed.append(f"DOMAIN_MISMATCH: {profile['current_title']} with no ML evidence")
        multiplier = min(multiplier, HARD_PENALTIES["non_tech_title_zero_ml"])

    return multiplier, failed


# Career evidence — keyword QUALITY not frequency
def _extract_career_evidence(text: str) -> tuple[dict[str, float], dict[str, list[str]]]:
    """
    Extract evidence from career text.
    Returns category scores and matched keyword details.
    """
    evidence: dict[str, float] = {}
    matched_keywords: dict[str, list[str]] = {}

    for category, keywords in EVIDENCE_KEYWORDS.items():
        hits: list[str] = []
        for kw in keywords:
            if kw in text:
                hits.append(kw)
        evidence[category] = min(len(hits) / 3.0, 1.0) if hits else 0.0
        if hits:
            matched_keywords[category] = hits

    # Depth bonus: candidates hitting 4+ core categories have deep expertise
    core_hits = sum(1 for c in CORE_EVIDENCE_CATEGORIES if evidence.get(c, 0) > 0)
    if core_hits >= 4:
        for c in CORE_EVIDENCE_CATEGORIES:
            evidence[c] = min(evidence[c] * 1.25, 1.0)
    elif core_hits >= 3:
        for c in CORE_EVIDENCE_CATEGORIES:
            evidence[c] = min(evidence[c] * 1.10, 1.0)

    return evidence, matched_keywords


def _career_evidence_score(evidence: dict[str, float]) -> float:
    """Compute aggregate career evidence score from per-category evidence."""
    core = sum(evidence.get(c, 0) * 2.0 for c in CORE_EVIDENCE_CATEGORIES)
    nice = sum(evidence.get(c, 0) * 1.0 for c in NICE_EVIDENCE_CATEGORIES)
    max_possible = len(CORE_EVIDENCE_CATEGORIES) * 2.0 + len(NICE_EVIDENCE_CATEGORIES) * 1.0
    return (core + nice) / max_possible


# Skill depth — duration + endorsements + corroboration
def _compute_skill_trust_detailed(
    skill: dict,
    career_text: str,
    assessments: dict,
    total_career_months: int = 0,
) -> dict:
    """
    Compute trust for one skill claim with depth analysis.
    Duration is a key depth signal: 6mo ≠ 36mo.
    """
    w = SKILL_TRUST_WEIGHTS
    name: str = skill["name"]
    dur_raw: int = skill.get("duration_months", 0)

    # Inflation check
    inflation_penalty: float = 1.0
    if total_career_months > 0 and dur_raw > total_career_months * 0.9:
        inflation_penalty = 0.7

    # Minimum duration filter
    if dur_raw < w["min_meaningful_duration_months"]:
        if name in HIGH_VALUE_AI_SKILLS and dur_raw >= 3:
            depth_multiplier = 0.6
        else:
            depth_multiplier = max(dur_raw / w["min_meaningful_duration_months"], 0.2)
    else:
        depth_multiplier = 1.0

    prof_score: float = w["proficiency"].get(skill["proficiency"], 0.05)
    dur_score: float = min(dur_raw / w["duration_max_months"], 1.0) * w["duration_weight"]

    endorse_raw: int = skill.get("endorsements", 0)
    endorse_score: float = (
        min(math.log(endorse_raw + 1) / math.log(w["endorsement_log_base"]), 1.0) * w["endorsement_weight"]
        if endorse_raw > 0 else 0.0
    )

    # Platform assessment — hardest evidence
    assess_raw = assessments.get(name, -1)
    if assess_raw < 0:
        for k, v in assessments.items():
            if k.lower() == name:
                assess_raw = v
                break
    assess_score: float = (assess_raw / 100.0) * w["assessment_weight"] if assess_raw >= 0 else 0.0

    # Career corroboration (enhanced with fuzzy matching)
    corroborated: bool = find_skill_in_text(career_text, name)
    if not corroborated and " " in name:
        parts = name.split()
        corroborated = any(p in career_text for p in parts if len(p) > 3)

    if corroborated:
        evidence_bonus: float = w["evidence_corroboration_bonus"]
    elif name in HIGH_VALUE_AI_SKILLS and endorse_raw >= 10:
        evidence_bonus = w["high_value_no_corroboration"]
    elif name in HIGH_VALUE_AI_SKILLS and dur_raw >= 24:
        evidence_bonus = w["high_value_no_corroboration"]
    elif assess_raw >= 50:
        evidence_bonus = w["high_value_no_corroboration"]
    else:
        evidence_bonus = 0.0

    raw_trust: float = prof_score + dur_score + endorse_score + assess_score + evidence_bonus
    total: float = min(raw_trust * depth_multiplier * inflation_penalty, 1.0)

    return {
        "name": name,
        "trust_score": round(total, 3),
        "proficiency": skill["proficiency"],
        "duration_months": dur_raw,
        "endorsements": endorse_raw,
        "assessment_raw": assess_raw if assess_raw >= 0 else None,
        "career_corroborated": corroborated,
        "verdict": "VERIFIED" if corroborated and total > 0.6 else "PARTIALLY_VERIFIED" if total > 0.35 else "UNVERIFIED",
    }


def _skill_trust_score_detailed(
    skills: list[dict],
    career_text: str,
    assessments: dict,
    total_career_months: int = 0,
) -> tuple[float, dict]:
    """Compute aggregate skill trust score with per-skill audit detail."""
    from backend.pipeline.skill_matching import normalize_skill_name
    
    core_details: list[dict] = []
    sec_details: list[dict] = []

    for s in skills:
        name = s["name"]
        # Use fuzzy normalization to match skill variants
        normalized = normalize_skill_name(name)
        
        if normalized in CORE_SKILLS or name in CORE_SKILLS:
            core_details.append(_compute_skill_trust_detailed(s, career_text, assessments, total_career_months))
        elif normalized in SECONDARY_SKILLS or name in SECONDARY_SKILLS:
            sec_details.append(_compute_skill_trust_detailed(s, career_text, assessments, total_career_months))

    verified_core: int = sum(1 for d in core_details if d["verdict"] in ("VERIFIED", "PARTIALLY_VERIFIED"))
    core_avg: float = sum(d["trust_score"] for d in core_details) / len(core_details) if core_details else 0.0
    sec_avg: float = sum(d["trust_score"] for d in sec_details) / len(sec_details) if sec_details else 0.0

    # Bonus for having MULTIPLE verified relevant skills
    breadth_bonus: float = min(verified_core * 0.02, 0.10)

    ai_skill_count = len(core_details)
    if ai_skill_count >= 6:
        breadth_bonus += 0.08
    elif ai_skill_count >= 4:
        breadth_bonus += 0.05

    overall: float = min(0.7 * core_avg + 0.3 * sec_avg + breadth_bonus, 1.0)

    return overall, {
        "overall": round(overall, 4),
        "core_skills_found": len(core_details),
        "verified_core_count": verified_core,
        "core_avg_trust": round(core_avg, 3),
        "secondary_avg_trust": round(sec_avg, 3),
        "core_skill_audit": sorted(core_details, key=lambda x: -x["trust_score"])[:8],
        "secondary_skill_audit": sorted(sec_details, key=lambda x: -x["trust_score"])[:5],
    }


# Job-specific role alignment
def _role_alignment_detailed(features: dict) -> tuple[float, dict]:
    """
    Score alignment with THIS specific job: Senior AI Engineer.
    Not just "senior engineer" — specifically AI/ML focused.
    """
    profile: dict = features["profile"]
    career: dict = features["career"]

    # Title fit
    cat: str = profile["title_category"]
    title_score: float = {"AI_ML": 1.0, "ADJACENT": 0.55, "NON_TECH": 0.05}.get(cat, 0.25)

    # Experience band (JD: 5-9 years ideal)
    y: float = profile["years_exp"]
    if JD["experience_ideal"][0] <= y <= JD["experience_ideal"][1]:
        exp = 1.0; exp_verdict = "IDEAL"
    elif JD["experience_min"] <= y <= JD["experience_max"]:
        exp = 0.85; exp_verdict = "ACCEPTABLE"
    elif 4 <= y < JD["experience_min"] or JD["experience_max"] < y <= 12:
        exp = 0.6; exp_verdict = "STRETCH"
    elif y < 3:
        exp = 0.15; exp_verdict = "TOO_JUNIOR"
    else:
        exp = 0.35; exp_verdict = "TOO_SENIOR"

    # Career progression toward AI/ML
    ml_roles: int = career["ml_role_count"]
    has_recent: bool = career["has_recent_ml"]
    prog: float = 0.4 + (0.35 if has_recent else 0.0) + 0.25 * min(ml_roles / 3.0, 1.0)

    overall: float = 0.45 * title_score + 0.30 * exp + 0.25 * prog

    return overall, {
        "overall": round(overall, 3),
        "title": profile["current_title"],
        "title_category": cat,
        "title_score": round(title_score, 2),
        "years_experience": y,
        "experience_score": round(exp, 2),
        "experience_verdict": exp_verdict,
        "ml_role_count": ml_roles,
        "has_current_ml_role": has_recent,
        "progression_score": round(prog, 3),
    }


# Soft anti-pattern detection
def _soft_penalties(features: dict) -> tuple[float, list[dict]]:
    """Detect soft anti-patterns and return cumulative penalty + flag list."""
    career: dict = features["career"]
    penalty: float = 0.0
    flags: list[dict] = []

    if career["product_ratio"] == 0 and career["role_count"] > 1:
        penalty += SOFT_PENALTIES["consulting_only"]
        flags.append({"pattern": "CONSULTING_ONLY", "penalty": SOFT_PENALTIES["consulting_only"],
                      "reason": "Entire career at consulting/IT services firms."})
    elif career["product_ratio"] < 0.3:
        penalty += SOFT_PENALTIES["mostly_consulting"]
        flags.append({"pattern": "MOSTLY_CONSULTING", "penalty": SOFT_PENALTIES["mostly_consulting"],
                      "reason": f"Only {career['product_ratio']:.0%} at product companies."})

    if 0 < career["avg_tenure"] < 18 and career["role_count"] >= 3:
        penalty += SOFT_PENALTIES["title_chasing"]
        flags.append({"pattern": "TITLE_CHASING", "penalty": SOFT_PENALTIES["title_chasing"],
                      "reason": f"Avg tenure {career['avg_tenure']:.0f} months across {career['role_count']} roles."})

    text: str = features["career"]["combined_text"]
    ml_prod_keywords = ["model serving", "ml pipeline", "mlops", "model deployment",
                        "recommendation engine", "ranking system", "trained model"]
    if not any(kw in text for kw in ml_prod_keywords):
        penalty += SOFT_PENALTIES["no_ml_production"]
        flags.append({"pattern": "NO_ML_PRODUCTION", "penalty": SOFT_PENALTIES["no_ml_production"],
                      "reason": "No ML production evidence in career descriptions."})

    return penalty, flags


# MAIN COMPUTE FUNCTION
def compute_evidence_score(features: dict) -> tuple[float, dict]:
    """
    Compute technical fit score for one candidate.

    Pipeline:
    1. Check hard requirements → multiplicative penalty if missing ML
    2. Extract career evidence → quality, not just keyword frequency
    3. Score skill depth → duration + corroboration + endorsements
    4. Role alignment → specific to Senior AI Engineer
    5. Soft penalties → consulting, title-chasing
    6. TF-IDF similarity → semantic career-to-JD matching
    7. Fuzzy skill matching → catch skill name variations
    8. Combine with evidence being primary driver
    """
    text: str = features["career"]["combined_text"]
    assessments: dict = features["behavioral"]["skill_assessment_scores"]

    # Step 1: Hard requirements
    hard_multiplier, hard_failures = _check_hard_requirements(features)

    # Step 2: Career evidence (keyword-based)
    evidence_cats, matched_keywords = _extract_career_evidence(text)
    career_ev: float = _career_evidence_score(evidence_cats)

    # Step 3: Skill depth
    total_career_months: int = features["career"]["total_months"]
    skill_tr, skill_audit = _skill_trust_score_detailed(features["skills"], text, assessments, total_career_months)

    # Step 4: Role alignment
    role_al, role_audit = _role_alignment_detailed(features)

    # Step 5: Soft penalties
    soft_penalty, soft_flags = _soft_penalties(features)

    # Step 6: TF-IDF semantic similarity (new — provides signal beyond keywords)
    skills_text = " ".join(s["name"] for s in features["skills"])
    tfidf_scores = compute_tfidf_similarity(text, skills_text)
    tfidf_signal: float = tfidf_scores["combined_similarity"]

    # Step 7: Fuzzy skill matching (new — catches skill name variations)
    fuzzy_match = match_skills_to_jd(features["skills"])
    fuzzy_boost: float = 0.0
    # Bonus for high-value skill depth that fuzzy matching discovers
    if fuzzy_match["high_value_matches"] >= 4:
        fuzzy_boost = 0.08
    elif fuzzy_match["high_value_matches"] >= 2:
        fuzzy_boost = 0.04
    elif fuzzy_match["high_value_matches"] >= 1:
        fuzzy_boost = 0.02

    # Step 8: Combine — adaptive weighting based on evidence availability
    verified_assessments: int = sum(1 for v in assessments.values() if v >= 50)
    has_strong_assessments: bool = verified_assessments >= 2

    assessment_boost: float = 0.0
    if verified_assessments >= 3:
        assessment_boost = 0.06
    elif verified_assessments >= 2:
        assessment_boost = 0.04

    # TF-IDF provides a baseline semantic signal that helps differentiate
    # candidates who use different vocabulary but have the same expertise
    tfidf_contribution = tfidf_signal * 0.12  # Up to ~0.12 bonus

    if career_ev >= 0.25:
        raw = 0.40 * career_ev + 0.25 * skill_tr + 0.23 * role_al + tfidf_contribution + soft_penalty + assessment_boost + fuzzy_boost
    elif skill_tr > 0.35 and has_strong_assessments:
        raw = 0.20 * career_ev + 0.40 * skill_tr + 0.25 * role_al + tfidf_contribution + soft_penalty + assessment_boost + fuzzy_boost
    elif skill_tr > 0.4 and career_ev < 0.2:
        raw = 0.25 * career_ev + 0.35 * skill_tr + 0.25 * role_al + tfidf_contribution + soft_penalty + fuzzy_boost
    else:
        raw = 0.40 * career_ev + 0.25 * skill_tr + 0.23 * role_al + tfidf_contribution + soft_penalty + fuzzy_boost

    raw = max(0.0, min(1.0, raw))

    # Apply hard multiplier
    score: float = raw * hard_multiplier

    return score, {
        "career_evidence": round(career_ev, 4),
        "skill_trust": round(skill_tr, 4),
        "role_alignment": round(role_al, 4),
        "tfidf_similarity": tfidf_scores,
        "fuzzy_skill_match": fuzzy_match,
        "soft_penalty": round(soft_penalty, 4),
        "hard_multiplier": round(hard_multiplier, 3),
        "hard_failures": hard_failures,
        "evidence_categories": evidence_cats,
        "matched_keywords": matched_keywords,
        "skill_audit": skill_audit,
        "role_audit": role_audit,
        "anti_patterns": soft_flags,
        "evidence_summary": _build_evidence_summary(
            evidence_cats, matched_keywords, skill_audit,
            role_audit, soft_flags, hard_failures
        ),
    }


def _build_evidence_summary(
    evidence_cats: dict,
    matched_keywords: dict,
    skill_audit: dict,
    role_audit: dict,
    soft_flags: list,
    hard_failures: list,
) -> list[dict]:
    """Build a structured evidence trail summary for audit purposes."""
    findings: list[dict] = []

    if hard_failures:
        for f in hard_failures:
            findings.append({"type": "HARD_FAIL", "signal": "negative", "message": f, "score_impact": "PENALTY"})

    for cat, score in sorted(evidence_cats.items(), key=lambda x: -x[1]):
        if score > 0:
            kws = matched_keywords.get(cat, [])
            label = cat.replace("_", " ").title()
            findings.append({
                "type": "CAREER_EVIDENCE", "signal": "positive",
                "message": f"{label}: [{', '.join(kws[:4])}]",
                "score_impact": f"+{score:.2f}",
            })

    for cat in CORE_EVIDENCE_CATEGORIES:
        if evidence_cats.get(cat, 0) == 0:
            label = cat.replace("_", " ").title()
            findings.append({
                "type": "MISSING_EVIDENCE", "signal": "negative",
                "message": f"No {label} evidence in career",
                "score_impact": "0.00",
            })

    for skill in skill_audit["core_skill_audit"][:4]:
        findings.append({
            "type": f"SKILL_{skill['verdict']}", "signal": "positive" if skill["verdict"] == "VERIFIED" else "warning",
            "message": f"'{skill['name']}': {skill['verdict']} ({skill['duration_months']}mo, {skill['endorsements']} endorsements)",
            "score_impact": f"trust={skill['trust_score']:.2f}",
        })

    findings.append({
        "type": "ROLE_FIT", "signal": "positive" if role_audit["overall"] > 0.6 else "negative",
        "message": f"{role_audit['title_category']} role, {role_audit['years_experience']:.1f}y exp ({role_audit['experience_verdict']})",
        "score_impact": f"alignment={role_audit['overall']:.2f}",
    })

    for ap in soft_flags:
        findings.append({
            "type": "SOFT_PENALTY", "signal": "negative",
            "message": ap["reason"],
            "score_impact": f"penalty={ap['penalty']:.2f}",
        })

    return findings
