"""
Scoring weights and penalty configuration.

All multiplicative and additive weight constants used across the scoring pipeline.
Tune these values to adjust scoring behaviour system-wide.
"""

# SCORE WEIGHTS
# Technical fit is the ONLY primary driver.
# Reliability = tiebreaker, not driver.
WEIGHTS: dict[str, float] = {
    "technical_fit": 0.65,   # Can this person DO the job?
    "reliability": 0.20,     # Can we HIRE this person? (secondary)
    "credibility": 0.15,     # External validation (tertiary)
}

# Hard penalties for disqualifying patterns (applied multiplicatively)
HARD_PENALTIES: dict[str, float] = {
    "zero_ml_evidence_in_career": 0.20,   # multiplier — severe cut for zero ML evidence
    "non_tech_title_zero_ml": 0.10,        # multiplier — almost zero
    "under_3_years_experience": 0.50,      # multiplier for severe underexp
}

# Soft penalties (additive, applied to evidence score)
SOFT_PENALTIES: dict[str, float] = {
    "consulting_only": -0.15,
    "mostly_consulting": -0.08,
    "title_chasing": -0.12,
    "no_ml_production": -0.12,
}

# RELIABILITY (Stage 2)
RELIABILITY_WEIGHTS: dict[str, float] = {
    "response_rate": 0.20,
    "recency": 0.25,
    "notice_period": 0.20,
    "location": 0.15,
    "open_to_work": 0.10,
    "interview_completion": 0.07,
    "response_time": 0.03,
}

# CREDIBILITY (Stage 3)
CREDIBILITY_WEIGHTS: dict[str, float] = {
    "github": 0.30, "assessments": 0.25, "education": 0.20,
    "career_velocity": 0.15, "verification": 0.10,
}

# SKILL TRUST
SKILL_TRUST_WEIGHTS: dict[str, object] = {
    # Proficiency — higher differentiation for advanced vs expert
    "proficiency": {"beginner": 0.05, "intermediate": 0.15, "advanced": 0.30, "expert": 0.40},
    # Duration — key depth signal. 6mo ≠ 36mo. Capped at 48 months
    "duration_max_months": 48,
    "duration_weight": 0.20,
    # Endorsements — social proof, log scale
    "endorsement_log_base": 50,
    "endorsement_weight": 0.15,
    # Platform assessment — HARD evidence, increased weight
    "assessment_weight": 0.25,
    # Career corroboration — the USP signal
    "evidence_corroboration_bonus": 0.30,
    # Partial credit for high-value skills or platform-verified skills
    "high_value_no_corroboration": 0.15,
    # Minimum skill duration to count at all (filters out "used once" skills)
    "min_meaningful_duration_months": 6,
}

# EDUCATION
TIER_WEIGHTS: dict[str, float] = {"tier_1": 1.0, "tier_2": 0.8, "tier_3": 0.5, "tier_4": 0.3, "unknown": 0.4}
DEGREE_WEIGHTS: dict[str, float] = {
    "Ph.D": 1.0, "M.Tech": 0.9, "M.E.": 0.9, "M.S.": 0.85, "M.Sc": 0.7,
    "B.Tech": 0.7, "B.E.": 0.7, "B.Sc": 0.5, "MBA": 0.3,
}
