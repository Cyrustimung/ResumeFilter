"""
Text Similarity Engine — TF-IDF based career-to-JD matching.

Uses scikit-learn TF-IDF vectorizer to compute cosine similarity between
candidate career text and the job description requirements. This gives
a semantic signal beyond simple keyword matching.

Runs fully offline, CPU-only, <50MB RAM.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# The JD requirements expressed as text corpus for TF-IDF comparison
_JD_CORPUS: list[str] = [
    # Core role description
    """senior ai engineer machine learning retrieval search ranking 
    recommendation systems embedding vector database semantic search
    production ml systems python pytorch tensorflow nlp information retrieval
    model serving deployment inference pipeline feature engineering
    evaluation metrics ndcg mrr precision recall a/b testing""",
    
    # Technical skills expected
    """python pytorch tensorflow scikit-learn xgboost lightgbm
    hugging face transformers sentence transformers faiss pinecone
    weaviate qdrant milvus elasticsearch opensearch mlflow mlops
    docker kubernetes aws gcp feature store model monitoring
    vector search embedding cosine similarity dense retrieval""",
    
    # Work style and seniority
    """senior engineer 5 to 9 years production systems scale
    team leadership cross functional product company tech startup
    ai ml data science research applied machine learning
    system design architecture distributed systems""",
]

# Pre-build vectorizer with the JD corpus
_vectorizer: TfidfVectorizer | None = None
_jd_vectors = None


def _get_vectorizer():
    """Lazy-initialize the TF-IDF vectorizer with JD corpus."""
    global _vectorizer, _jd_vectors
    if _vectorizer is None:
        _vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),  # unigrams + bigrams
            max_features=5000,
            stop_words="english",
            sublinear_tf=True,  # log-normalize term frequencies
            min_df=1,
        )
        # Fit on JD + a dummy doc to establish vocabulary
        all_docs = _JD_CORPUS + ["placeholder document for vocabulary"]
        _vectorizer.fit(all_docs)
        _jd_vectors = _vectorizer.transform(_JD_CORPUS)
    return _vectorizer, _jd_vectors


def compute_tfidf_similarity(career_text: str, skills_text: str = "") -> dict:
    """
    Compute TF-IDF cosine similarity between candidate and JD.
    
    Args:
        career_text: Combined career description text (lowercased)
        skills_text: Space-joined skill names
    
    Returns:
        Dict with similarity scores for career text, skills, and combined.
    """
    vectorizer, jd_vectors = _get_vectorizer()
    
    # Combine career + skills for a full candidate representation
    combined_text = f"{career_text} {skills_text}"
    
    # Compute similarities
    career_vec = vectorizer.transform([career_text])
    combined_vec = vectorizer.transform([combined_text])
    
    # Cosine similarity against each JD document
    career_sims = cosine_similarity(career_vec, jd_vectors)[0]
    combined_sims = cosine_similarity(combined_vec, jd_vectors)[0]
    
    # Weighted average across JD aspects
    # Role desc (most important) > Skills > Work style
    weights = np.array([0.5, 0.35, 0.15])
    
    career_score = float(np.dot(career_sims, weights))
    combined_score = float(np.dot(combined_sims, weights))
    
    return {
        "career_similarity": round(career_score, 4),
        "combined_similarity": round(combined_score, 4),
        "role_match": round(float(career_sims[0]), 4),
        "skill_match": round(float(combined_sims[1]), 4),
        "seniority_match": round(float(career_sims[2]), 4),
    }
