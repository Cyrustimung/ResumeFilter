"""
Keyword lists and skill sets for evidence matching.

Contains all keyword dictionaries, skill categorizations, and company lists
used to evaluate technical fit from career text and skill claims.
"""

# EVIDENCE KEYWORDS (Stage 1a) — Career text evidence
EVIDENCE_KEYWORDS: dict[str, list[str]] = {
    "retrieval_search": [
        "retrieval", "search", "ranking", "recommendation", "information retrieval",
        "bm25", "tf-idf", "inverted index", "query", "relevance", "reranking",
        "re-ranking", "search engine", "search quality", "candidate retrieval",
    ],
    "embeddings_vectors": [
        "embedding", "vector", "sentence-transformer", "semantic search",
        "cosine similarity", "dense retrieval", "faiss", "ann",
        "approximate nearest", "vector search", "semantic similarity",
        "representation learning",
    ],
    "vector_databases": [
        "pinecone", "weaviate", "qdrant", "milvus", "chroma", "faiss",
        "elasticsearch", "opensearch", "vector database", "vector store",
        "vector index", "annoy", "hnsw",
    ],
    "ml_production": [
        "model serving", "model deployment", "ml pipeline", "mlops",
        "inference server", "feature store", "model monitoring", "ml system",
        "recommendation engine", "ranking system", "ml model", "ai system",
        "machine learning pipeline", "trained model", "model inference",
        "prediction service", "serving infrastructure",
        "classification", "regression", "clustering", "xgboost",
        "sklearn", "scikit", "training data", "model training",
        "feature engineering", "prediction", "machine learning",
    ],
    "llm_nlp_work": [
        "llm", "fine-tuning", "fine tuning", "lora", "qlora", "peft",
        "prompt", "transformer", "gpt", "language model", "rlhf",
        "instruction tuning", "rag", "retrieval augmented", "nlp",
        "natural language processing", "text classification", "named entity",
        "sentiment analysis", "text generation",
    ],
    "evaluation_ml": [
        "ndcg", "mrr", "map", "precision@", "recall@", "a/b test",
        "offline evaluation", "online evaluation", "benchmark",
        "ranking quality", "relevance judgment", "model evaluation",
    ],
    "data_ml_infra": [
        "feature engineering", "feature pipeline", "training pipeline",
        "data pipeline for ml", "ml data", "model training",
        "experiment tracking", "hyperparameter", "model versioning",
    ],
}

CORE_EVIDENCE_CATEGORIES: list[str] = [
    "retrieval_search", "embeddings_vectors", "vector_databases",
    "ml_production", "evaluation_ml"
]
NICE_EVIDENCE_CATEGORIES: list[str] = ["llm_nlp_work", "data_ml_infra"]

# HARD REQUIREMENT CHECKS
HARD_REQUIREMENT_KEYWORDS: dict[str, list[str]] = {
    "has_ml_in_career": [
        "machine learning", "deep learning", "neural", "model", "training",
        "inference", "prediction", "classification", "regression", "embedding",
        "nlp", "computer vision", "recommendation", "ranking", "retrieval",
    ],
    "has_python": ["python"],
    "has_production_technical": [
        "deployed", "production", "shipped", "real users", "at scale",
        "serving", "monitoring", "model serving", "model deployment",
    ],
}

# SKILLS
CORE_SKILLS: set[str] = {
    # Retrieval & search (highest relevance)
    "embeddings", "sentence-transformers", "vector search", "pinecone",
    "weaviate", "qdrant", "milvus", "faiss", "elasticsearch", "opensearch",
    "information retrieval", "ranking", "search", "retrieval",
    "recommendation systems", "semantic search", "bm25",
    # ML core
    "python", "nlp", "machine learning", "deep learning", "pytorch", "tensorflow",
    "hugging face", "huggingface", "transformers", "hugging face transformers",
    # LLM & fine-tuning
    "fine-tuning llms", "fine tuning", "lora", "qlora", "peft",
    "language models", "llm",
    # MLOps
    "mlops", "model deployment", "model serving",
}

SECONDARY_SKILLS: set[str] = {
    "spark", "airflow", "kafka", "dbt", "snowflake", "databricks",
    "data pipelines", "feature engineering",
    "aws", "gcp", "azure", "docker", "kubernetes",
    "xgboost", "scikit-learn", "pandas", "numpy",
    "computer vision", "speech recognition", "gans", "image classification",
    "object detection", "kubeflow", "weights & biases", "mlflow",
    "sql", "postgresql",
}

# Rare/specialized AI skills — strong signal even without career text corroboration
HIGH_VALUE_AI_SKILLS: set[str] = {
    # Vector/retrieval tools
    "milvus", "pinecone", "weaviate", "qdrant", "faiss",
    # LLM/NLP
    "fine-tuning llms", "lora", "qlora", "nlp",
    "hugging face transformers", "transformers",
    # Retrieval/search
    "recommendation systems", "embeddings", "semantic search",
    "information retrieval", "bm25",
    # ML core that signals genuine expertise
    "gans", "object detection", "mlops", "kubeflow",
    "deep learning", "computer vision", "speech recognition",
    "image classification", "prompt engineering",
    "weights & biases", "tensorflow", "pytorch",
}

# COMPANIES
CONSULTING_FIRMS: set[str] = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "deloitte", "mckinsey", "bcg", "bain",
    "pwc", "ey", "kpmg", "ibm", "dxc", "ltimindtree",
    "mphasis", "hexaware", "persistent",
}

# EDUCATION FIELDS
RELEVANT_FIELDS: set[str] = {
    "computer science", "machine learning", "artificial intelligence",
    "data science", "information technology", "software engineering",
    "statistics", "mathematics", "electrical engineering", "electronics",
}
