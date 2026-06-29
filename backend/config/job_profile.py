"""
Job profile, titles, location preferences, and runtime constants.

Defines the target role (Senior AI Engineer @ Redrob AI), title classifications,
city preferences, integrity gate thresholds, and filesystem paths.
"""

from datetime import date
import os

# PATHS
ENGINE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # backend/
PROJECT_ROOT: str = os.path.dirname(ENGINE_DIR)  # project root
DATA_DIR: str = os.path.join(PROJECT_ROOT, "datasets")
SAMPLE_CANDIDATES_PATH: str = os.path.join(DATA_DIR, "sample_candidates.json")

# RUNTIME CONSTANTS
TODAY: date = date(2026, 6, 15)
TOP_K: int = 100

# JD REQUIREMENTS — The role we are ranking against
JD: dict = {
    "title": "Senior AI Engineer",
    "experience_min": 5,
    "experience_max": 9,
    "experience_ideal": (6, 8),
    # Hard requirements — missing any = strong penalty
    "hard_requirements": [
        "production_ml_experience",
        "retrieval_or_ranking_systems",
        "python_proficiency",
    ],
    # Preferred requirements
    "preferred": [
        "vector_database_experience",
        "embeddings_or_semantic_search",
        "evaluation_framework_experience",
        "llm_or_finetuning_experience",
    ],
    # Explicit disqualifiers from JD text
    "disqualifiers": [
        "pure_consulting_no_ml",
        "non_tech_role_no_ml_evidence",
        "junior_under_3_years",
        "only_vision_speech_no_nlp_ir",
    ],
}

# INTEGRITY GATE (Stage 0)
INTEGRITY: dict[str, int] = {
    "timeline_tolerance_months": 24,
    "skill_inflation_threshold": 5,
    "high_prof_zero_endorse_threshold": 7,
    "min_duration_for_advanced": 6,
    "overlap_tolerance_months": 6,
}

# TITLES
AI_ML_TITLES: set[str] = {
    "ml engineer", "machine learning engineer", "ai engineer",
    "data scientist", "nlp engineer", "research engineer",
    "senior ml engineer", "senior ai engineer", "staff ml engineer",
    "applied scientist", "senior data scientist", "lead ml engineer",
    "junior ml engineer", "junior ai engineer", "recommendation systems engineer",
    "senior machine learning engineer", "lead ai engineer", "staff ai engineer",
    "applied ml engineer", "senior nlp engineer", "senior applied scientist",
    "search engineer", "senior search engineer", "ranking engineer",
    "retrieval engineer", "senior research engineer", "principal ml engineer",
    "staff machine learning engineer", "lead machine learning engineer",
    "ai/ml engineer", "deep learning engineer", "senior deep learning engineer",
    "ai research engineer", "senior ai research engineer", "ml research engineer",
    "machine learning researcher", "ai researcher", "research scientist",
    "applied ai engineer", "computer vision engineer", "cv engineer",
}

ADJACENT_TITLES: set[str] = {
    "software engineer", "backend engineer", "data engineer",
    "platform engineer", "senior software engineer", "tech lead",
    "senior backend engineer", "full stack engineer", "devops engineer",
    "senior data engineer", "lead engineer", "staff engineer", "cloud engineer",
    "senior platform engineer", "principal engineer", "staff software engineer",
    "mobile developer", "frontend engineer", "senior frontend engineer",
}

NON_TECH_TITLES: set[str] = {
    "marketing manager", "hr manager", "operations manager",
    "sales executive", "accountant", "content writer",
    "graphic designer", "business analyst", "customer support",
    "civil engineer", "mechanical engineer", "project manager",
    "product manager", "ux designer", "qa engineer", "manual tester",
}

# LOCATION PREFERENCES
PREFERRED_CITIES: list[str] = ["pune", "noida", "delhi", "ncr", "gurgaon", "gurugram"]
ACCEPTABLE_CITIES: list[str] = ["hyderabad", "bangalore", "bengaluru", "mumbai", "chennai"]
