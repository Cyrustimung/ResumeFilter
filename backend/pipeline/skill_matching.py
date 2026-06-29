"""
Fuzzy Skill Matching — Handles skill name variations.

Uses rapidfuzz to match candidate skills against known relevant skills,
catching variations like:
  - "scikit-learn" vs "sklearn"
  - "HuggingFace Transformers" vs "Hugging Face"
  - "Recommendation Systems" vs "recommender systems"
  - "TensorFlow" vs "tensorflow"

Also uses NLTK stemming to normalize skill descriptions in career text.

Runs fully offline, CPU-only, <10MB RAM.
"""

from rapidfuzz import fuzz, process
from functools import lru_cache

from backend.config import CORE_SKILLS, SECONDARY_SKILLS, HIGH_VALUE_AI_SKILLS

# Canonical skill names and their common aliases
_SKILL_ALIASES: dict[str, list[str]] = {
    "pytorch": ["torch", "py torch"],
    "tensorflow": ["tf", "tensor flow", "keras"],
    "scikit-learn": ["sklearn", "scikit learn", "sci-kit learn"],
    "hugging face transformers": ["huggingface", "hugging face", "hf transformers", "transformers"],
    "recommendation systems": ["recommender systems", "recommender", "recommendations", "recsys"],
    "embeddings": ["embedding", "word embeddings", "sentence embeddings", "dense embeddings"],
    "information retrieval": ["ir", "information retrieval systems"],
    "vector search": ["vector similarity", "ann search", "approximate nearest neighbor"],
    "natural language processing": ["nlp", "text processing", "language understanding"],
    "computer vision": ["cv", "image processing", "visual computing"],
    "deep learning": ["dl", "neural networks", "deep neural networks"],
    "machine learning": ["ml", "statistical learning"],
    "fine-tuning llms": ["fine tuning", "finetuning", "llm fine-tuning", "model fine-tuning"],
    "mlops": ["ml ops", "ml operations", "machine learning operations"],
    "faiss": ["facebook ai similarity search"],
    "pinecone": ["pinecone db", "pinecone vector"],
    "weaviate": ["weaviate db"],
    "qdrant": ["qdrant db"],
    "milvus": ["milvus db"],
    "elasticsearch": ["elastic search", "elastic"],
    "opensearch": ["open search", "aws opensearch"],
    "langchain": ["lang chain", "langchain framework"],
    "sentence transformers": ["sentence-transformers", "sbert"],
    "xgboost": ["xg boost", "extreme gradient boosting"],
    "object detection": ["object recognition", "yolo", "detection models"],
    "speech recognition": ["asr", "automatic speech recognition", "speech to text"],
    "gans": ["generative adversarial networks", "gan"],
    "prompt engineering": ["prompt design", "prompt crafting"],
    "weights & biases": ["wandb", "w&b"],
    "data pipelines": ["data pipeline", "etl pipeline", "data workflow"],
    "feature engineering": ["feature extraction", "feature design", "feature pipeline"],
    "model serving": ["model deployment", "serving infrastructure", "inference server"],
    "a/b testing": ["ab testing", "experimentation", "a/b test", "split testing"],
}

# Build a flat lookup: alias -> canonical name
_ALIAS_TO_CANONICAL: dict[str, str] = {}
for canonical, aliases in _SKILL_ALIASES.items():
    _ALIAS_TO_CANONICAL[canonical] = canonical
    for alias in aliases:
        _ALIAS_TO_CANONICAL[alias] = canonical

# All known skill names for fuzzy matching
_ALL_KNOWN_SKILLS: list[str] = list(
    set(list(CORE_SKILLS) + list(SECONDARY_SKILLS) + list(_ALIAS_TO_CANONICAL.keys()))
)


@lru_cache(maxsize=1000)
def normalize_skill_name(skill_name: str) -> str:
    """
    Normalize a skill name using aliases and fuzzy matching.
    Returns the canonical form if found, otherwise the original.
    """
    name = skill_name.lower().strip()
    
    # Exact match in aliases (fast path)
    if name in _ALIAS_TO_CANONICAL:
        return _ALIAS_TO_CANONICAL[name]
    
    # Check if it's already a known skill (fast path)
    if name in CORE_SKILLS or name in SECONDARY_SKILLS:
        return name
    
    # Fuzzy match against known aliases (threshold 85%)
    # Use WRatio for better multi-word matching
    best_match = process.extractOne(
        name,
        list(_ALIAS_TO_CANONICAL.keys()),
        scorer=fuzz.WRatio,
        score_cutoff=85,
    )
    if best_match:
        return _ALIAS_TO_CANONICAL[best_match[0]]
    
    # Fuzzy match against known skills (threshold 88% — higher to avoid false matches)
    best_skill = process.extractOne(
        name,
        list(CORE_SKILLS | SECONDARY_SKILLS),
        scorer=fuzz.WRatio,
        score_cutoff=88,
    )
    if best_skill:
        return best_skill[0]
    
    return name


def match_skills_to_jd(candidate_skills: list[dict]) -> dict:
    """
    Match candidate skills against JD requirements using fuzzy matching.
    
    Returns:
        Dict with matched core skills, secondary skills, match quality scores.
    """
    matched_core: list[dict] = []
    matched_secondary: list[dict] = []
    matched_high_value: list[dict] = []
    unmatched: list[str] = []
    
    for skill in candidate_skills:
        name = skill["name"]
        normalized = normalize_skill_name(name)
        
        skill_info = {
            "original": name,
            "normalized": normalized,
            "proficiency": skill.get("proficiency", "beginner"),
            "duration_months": skill.get("duration_months", 0),
            "endorsements": skill.get("endorsements", 0),
        }
        
        if normalized in CORE_SKILLS:
            matched_core.append(skill_info)
        elif normalized in SECONDARY_SKILLS:
            matched_secondary.append(skill_info)
        else:
            unmatched.append(name)
        
        # Also check high value (can be in both core and high_value)
        if normalized in {s.lower() for s in HIGH_VALUE_AI_SKILLS}:
            matched_high_value.append(skill_info)
    
    # Compute coverage scores
    core_coverage = len(matched_core) / max(len(CORE_SKILLS), 1)
    high_value_coverage = len(matched_high_value) / min(8, len(HIGH_VALUE_AI_SKILLS))
    
    # Depth score: weighted by duration and proficiency
    prof_weights = {"beginner": 0.2, "intermediate": 0.5, "advanced": 0.8, "expert": 1.0}
    
    def depth_score(skills_list: list[dict]) -> float:
        if not skills_list:
            return 0.0
        scores = []
        for s in skills_list:
            dur_factor = min(s["duration_months"] / 36.0, 1.0)
            prof_factor = prof_weights.get(s["proficiency"], 0.2)
            scores.append(dur_factor * 0.5 + prof_factor * 0.5)
        return sum(scores) / len(scores)
    
    return {
        "core_matches": len(matched_core),
        "secondary_matches": len(matched_secondary),
        "high_value_matches": len(matched_high_value),
        "core_coverage": round(core_coverage, 4),
        "high_value_coverage": round(high_value_coverage, 4),
        "core_depth_score": round(depth_score(matched_core), 4),
        "high_value_depth_score": round(depth_score(matched_high_value), 4),
        "matched_core_skills": [s["normalized"] for s in matched_core],
        "matched_high_value_skills": [s["normalized"] for s in matched_high_value],
        "unmatched_count": len(unmatched),
    }


def find_skill_in_text(text: str, skill_name: str) -> bool:
    """
    Check if a skill is mentioned in text, considering aliases.
    More robust than simple substring matching.
    """
    name_lower = skill_name.lower()
    text_lower = text.lower()
    
    # Direct match
    if name_lower in text_lower:
        return True
    
    # Check aliases
    canonical = normalize_skill_name(name_lower)
    if canonical in text_lower:
        return True
    
    # Check all aliases for this canonical name
    if canonical in _SKILL_ALIASES:
        for alias in _SKILL_ALIASES[canonical]:
            if alias in text_lower:
                return True
    
    # Check if any word longer than 4 chars from the skill name appears
    parts = name_lower.split()
    if len(parts) > 1:
        return any(p in text_lower for p in parts if len(p) > 4)
    
    return False
