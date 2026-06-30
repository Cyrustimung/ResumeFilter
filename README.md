# Intelligent Candidate Ranking

An evidence-based candidate ranking system that goes beyond keyword matching. Scores candidates by cross-verifying skill claims against career evidence, integrating behavioral signals, and applying semantic similarity — producing an explainable, ranked shortlist for any technical role.

## Quick Start

```bash
# Install dependencies
pip install -r backend/requirements.txt

# Run ranking (place candidates.jsonl in root, or use sample data)
python -m backend --candidates ./candidates.jsonl --out ./submission.csv

# Or test with included sample data (50 candidates)
python -m backend --candidates ./sample_candidates.json --out ./submission.csv

# Validate submission format
python validate_submission.py ./submission.csv
```

This scores 100,000 candidates and outputs a ranked top-100 CSV under ~5 minutes. No GPU, no network, no LLM APIs required.

## How It Works

The system runs a 4-stage scoring pipeline on each candidate:

```
Stage 1: Integrity Gate
  Detects fake profiles — timeline fabrication, skill inflation,
  impossible date overlaps. Honeypots get score = 0.

Stage 2: Technical Fit (65% of final score)
  - Extracts career evidence from job descriptions (keyword quality, not frequency)
  - TF-IDF semantic similarity between career text and the JD
  - Fuzzy skill matching with 40+ alias mappings (sklearn ↔ scikit-learn)
  - Skill trust scoring: duration + endorsements + platform assessments
  - Cross-verifies listed skills against career text (the core differentiator)
  - Role alignment: title category + experience band + ML progression

Stage 3: Hiring Feasibility (20% of final score)
  - Recruiter response rate, activity recency, notice period
  - Location match, open-to-work flag, interview completion rate

Stage 4: Credibility (15% of final score)
  - GitHub activity score, platform skill assessments
  - Education tier and relevance, career velocity
  - Profile verification status
```

**Final score** = integrity_gate × (0.65 × technical_fit + 0.20 × feasibility + 0.15 × credibility)

## What Makes This Different

**Evidence over keywords** — A listed skill only counts if it's corroborated by career descriptions, endorsed by peers, or backed by platform assessments. "Python" on a profile without Python work in career text gets discounted.

**Fraud detection at the scoring layer** — Timeline fabrication (claiming 8 years at a company founded 3 years ago), impossible skill durations, and inflated proficiency claims are caught before scoring begins.

**Semantic matching** — TF-IDF cosine similarity catches candidates who describe ML work with different vocabulary than the keyword lists. A candidate who writes "built a learning-to-rank model" matches even without the word "ranking" in their skills.

**Fuzzy skill normalization** — 40+ alias mappings ensure "scikit-learn" matches "sklearn", "HuggingFace" matches "Hugging Face Transformers", etc.

**Behavioral signal integration** — A perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% recruiter response rate is, for hiring purposes, not actually available. The system down-weights them appropriately.

**Every decision is explainable** — Each candidate gets a reasoning string referencing specific facts from their profile: keywords found in career text, verified skill durations, experience band fit, and honest concerns.

## Project Structure

```
├── backend/                   
│   ├── __main__.py
│   ├── ranker.py
│   ├── requirements.txt
│   ├── config/
│   │   ├── job_profile.py
│   │   ├── evidence_taxonomy.py
│   │   └── weights.py
│   ├── pipeline/
│   │   ├── feature_extraction.py
│   │   ├── integrity_gate.py
│   │   ├── technical_fit.py
│   │   ├── text_similarity.py
│   │   ├── skill_matching.py
│   │   ├── reliability.py
│   │   └── credibility.py
│   └── scoring/
│       ├── fusion.py
│       └── reasoning.py
├── sample_candidates.json      # Test data (50 candidates)
├── validate_submission.py      # Official format validator
├── submission.csv              # Generated ranked output
├── submission_metadata.yaml    # Hackathon metadata
├── .gitignore
└── README.md
```

## Requirements

- Python 3.10+
- ~2 GB RAM peak during scoring
- No GPU, no internet connection during ranking

**Dependencies:**
```
scikit-learn >= 1.4.0
rapidfuzz >= 3.6.0
nltk >= 3.8.0
numpy >= 1.26.0
fastapi >= 0.110.0
uvicorn >= 0.29.0
```

Install:
```bash
pip install -r backend/requirements.txt
python -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True)"
```

## Usage

```bash
# Rank the candidates → produces submission CSV
python -m backend --candidates ./candidates.jsonl --out ./submission.csv

# Validate submission format
python validate_submission.py ./submission.csv
```

The CLI shows live progress, a results table, and summary stats:

```
==================================================
  Intelligent Candidate Ranking
  Team Circle | Redrob Hackathon 2026
==================================================

  Loading candidates...
  ✓ Loaded 10,000 candidates

  Scoring 10,000 candidates (est. ~3s)...
  [########################] 100% · 10,000/10,000 ·

  ✓ Top 100 ranked and saved to submission.csv
```


```csv
candidate_id,rank,score,reasoning
CAND_0079387,1,0.930947,"AI Engineer with 6.9 years whose career demonstrates production ML work..."
CAND_0043228,2,0.921340,"Applied ML Engineer with 6.8 years..."
```

Each reasoning entry references verifiable facts from the candidate's profile — years, specific details found in career, verified skill durations, behavioral signals, and honest concerns where applicable.
