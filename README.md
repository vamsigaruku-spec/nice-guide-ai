# NICE Guide AI

## Evidence-Grounded Clinical Guidance Retrieval and Summarisation

NICE Guide AI is a production-oriented Retrieval-Augmented Generation (RAG) prototype for searching and summarising clinical guidance using retrieved evidence rather than relying on unconstrained language-model generation.

The system combines semantic retrieval, lexical BM25 retrieval, Reciprocal Rank Fusion (RRF), cross-encoder reranking, evidence sufficiency checks, citation validation, and NLI-based claim grounding before returning an answer.

> **Important:** This repository currently uses a synthetic clinical-guidance development corpus. It is a portfolio/development prototype and is NOT a medical device, clinical decision-support system, diagnostic system, or substitute for qualified professional clinical judgement.

---

## 1. Problem Statement

Clinical guidance can contain large amounts of structured information distributed across conditions, populations, sections, recommendations, and supporting evidence.

A basic LLM chatbot can produce fluent answers while potentially:

- retrieving irrelevant information
- relying on knowledge outside the intended source
- generating unsupported clinical claims
- providing incorrect or missing citations
- failing to distinguish insufficient evidence from an answerable question

NICE Guide AI is designed to address these retrieval and grounding problems through an explicit evidence-first RAG pipeline.

---

## 2. Project Objective

The objective is to build a clinical-guidance RAG system that:

1. Retrieves relevant guidance from a structured corpus.
2. Combines semantic and lexical retrieval.
3. Improves ranking using Reciprocal Rank Fusion.
4. Applies cross-encoder reranking to retrieved candidates.
5. Checks whether sufficient evidence exists before generation.
6. Generates answers using only retrieved evidence.
7. Requires citations to retrieved evidence chunks.
8. Validates citations against the retrieved evidence.
9. Uses NLI-based claim validation to check whether generated claims are supported.
10. Refuses to provide an answer when validation fails or evidence is insufficient.
11. Exposes the pipeline through a FastAPI backend.
12. Provides an interactive Streamlit interface.

---

## 3. System Architecture

```text
                    Clinical Guidance Source
                             |
                             v
                         Ingestion
                             |
                             v
                  Cleaning / Validation
                             |
                             v
                 Structure-Aware Chunks
                             |
                 +-----------+-----------+
                 |                       |
                 v                       v
          Text Embeddings             BM25
                 |                       |
                 +-----------+-----------+
                             |
                             v
                   Hybrid Retrieval
                         (RRF)
                             |
                             v
                  Cross-Encoder Reranker
                             |
                             v
                 Evidence Sufficiency Check
                             |
                             v
                       Groq LLM
                             |
                             v
                    Citation Validation
                             |
                             v
                   NLI Claim Validation
                             |
                    +--------+--------+
                    |                 |
                    v                 v
               Valid Answer        Refusal
```

---

## 4. Retrieval Pipeline

### 4.1 Query Normalisation

Incoming questions are normalised before retrieval so that equivalent terminology can be handled consistently.

### 4.2 Semantic Retrieval

The system uses Sentence Transformers with:

```text
all-MiniLM-L6-v2
```

The query and guidance chunks are represented as dense vectors and ranked using embedding similarity.

### 4.3 Lexical Retrieval

BM25 provides lexical matching and helps retrieve evidence where exact terminology is important.

### 4.4 Hybrid Retrieval

Semantic and BM25 rankings are combined using Reciprocal Rank Fusion (RRF).

This allows the system to benefit from both:

- semantic similarity
- exact/lexical term matching

### 4.5 Cross-Encoder Reranking

The hybrid candidates are passed through a cross-encoder reranker to obtain a more precise query-to-evidence ranking.

The application identifies this retrieval configuration as:

```text
hybrid_rrf_plus_cross_encoder
```

---

## 5. Grounded Generation

The generation layer is deliberately constrained.

The model is instructed to:

- use only retrieved evidence
- avoid outside medical knowledge
- avoid inventing recommendations
- avoid inventing diagnoses or treatments
- avoid unsupported thresholds or measurements
- cite retrieved evidence
- never invent citation identifiers
- refuse when validated evidence is insufficient

The generation layer uses the Groq API.

The API key is loaded through the environment variable:

```text
GROQ_API_KEY
```

API keys must never be committed to GitHub.

---

## 6. Evidence Validation

Generation is followed by multiple validation layers.

### Evidence Sufficiency

The system checks whether the retrieved evidence meets the configured reranker-score threshold.

### Citation Validation

Generated citation identifiers are compared against the actual retrieved chunk IDs.

Invalid or missing citations cause the answer to fail validation.

### Claim Grounding

Individual factual claims are checked against cited evidence using an NLI cross-encoder.

The current NLI model is:

```text
cross-encoder/nli-deberta-v3-base
```

The system uses evidence as the premise and the generated claim as the hypothesis.

### Refusal Mechanism

When the answer cannot be sufficiently validated, the system returns:

```text
The retrieved guidance does not provide enough validated evidence to answer this question.
```

This is preferable to generating an unsupported clinical response.

---

## 7. Current Development Dataset

The current repository contains a synthetic clinical-guidance development corpus.

It is intentionally used for development and testing rather than representing licensed NICE clinical content.

Current development artifacts include:

- 15 guidance chunks
- structured recommendation metadata
- provenance fields
- version information
- publication/update metadata
- active/inactive status
- 384-dimensional embeddings

Production NICE content should only be integrated through an authorised source and according to the applicable NICE licensing and usage requirements.

---

## 8. Evaluation

The project includes development evaluation covering retrieval and end-to-end RAG behaviour.

### Retrieval Evaluation

The evaluation compares:

- BM25
- semantic retrieval
- hybrid RRF
- hybrid RRF + cross-encoder reranking

Metrics include:

- Hit@1
- Hit@3
- Hit@5
- Recall@K
- MRR
- NDCG@K
- retrieval latency

### End-to-End Evaluation

The system also evaluates:

- answer validity
- evidence sufficiency
- citation validity
- claim grounding
- latency

> Current evaluation labels are development/silver labels, not independent human clinical validation.

---

## 9. Provenance and Freshness

The data model tracks metadata such as:

- guideline ID
- recommendation ID
- condition
- section
- source
- source type
- version
- publication date
- last updated date
- active/inactive status

This allows future ingestion workflows to detect source changes and support reproducible updates.

---

## 10. API

The application exposes a FastAPI backend.

### Health

```text
GET /health
```

### Ask

```text
POST /ask
```

Example request:

```json
{
  "query": "What education should be provided to adults with asthma?",
  "top_k": 3
}
```

The response includes:

- original query
- normalised query
- retrieval method
- generated answer
- retrieved evidence
- evidence validation
- citation validation
- claim validation
- final answer validity

### Retrieve

```text
POST /retrieve
```

This endpoint exposes the retrieved and reranked evidence without performing final answer generation.

---

## 11. Streamlit Interface

The project includes an interactive Streamlit application.

The interface allows a user to:

1. Enter a clinical-guidance question.
2. Select the number of evidence chunks.
3. Run the RAG pipeline.
4. View the grounded answer.
5. View evidence sufficiency status.
6. View citation validation status.
7. View answer validation status.
8. Inspect retrieved evidence.
9. Inspect claim-validation results.

---

## 12. Project Structure

```text
nice-guide-ai/
│
├── app/
│   ├── __init__.py
│   ├── rag.py
│   └── main.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── rag_chunks.csv
│   │   ├── guidance_dev.csv
│   │   └── embeddings.npy
│   └── evaluation/
│
├── reports/
│
├── tests/
│
├── streamlit_app.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 13. Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| Embeddings | Sentence Transformers |
| Embedding model | all-MiniLM-L6-v2 |
| Lexical retrieval | BM25 |
| Hybrid ranking | Reciprocal Rank Fusion |
| Reranking | Cross-Encoder |
| Claim validation | NLI Cross-Encoder |
| LLM | Groq API |
| API | FastAPI |
| UI | Streamlit |
| Testing | Pytest |
| Development | Google Colab |
| Version control | Git / GitHub |

---

## 14. Local Setup

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/nice-guide-ai.git
cd nice-guide-ai
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure the Groq API key:

```bash
export GROQ_API_KEY="your_api_key_here"
```

Run Streamlit:

```bash
streamlit run streamlit_app.py
```

Run FastAPI:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## 15. Testing

Run the test suite with:

```bash
pytest -q
```

The tests cover API contracts and core application behaviour.

---

## 16. Security

Secrets are intentionally excluded from version control.

The repository uses:

```text
.gitignore
.env.example
```

The real `GROQ_API_KEY` must be supplied through an environment variable or deployment platform secret.

Never commit:

- API keys
- passwords
- private credentials
- deployment secrets

---

## 17. Limitations

This project is a portfolio/development prototype and has important limitations.

- The current corpus is synthetic.
- It is not validated for clinical use.
- Retrieval quality depends on the indexed source material.
- NLI validation does not guarantee clinical correctness.
- Citation validation verifies citation identifiers and textual entailment, not the medical correctness of the underlying source.
- The current evaluation dataset contains development/silver labels rather than independent clinical expert annotations.
- Production deployment would require authorised source integration, stronger evaluation, monitoring, security controls, and appropriate governance.

---

## 18. Future Improvements

Planned improvements include:

- authorised NICE API/source integration
- production PostgreSQL + pgvector
- incremental document ingestion
- source change detection
- stronger metadata filtering
- larger expert-reviewed evaluation sets
- retrieval A/B testing
- automated regression evaluation
- production monitoring
- latency and cost tracking
- authentication and access control
- CI/CD with GitHub Actions
- improved deployment infrastructure

---

## 19. Project Status

Current status: **Development / Portfolio Prototype**

Implemented:

- Hybrid semantic + BM25 retrieval
- Reciprocal Rank Fusion
- Cross-encoder reranking
- Evidence sufficiency checks
- Grounded LLM generation
- Citation validation
- NLI claim validation
- Refusal mechanism
- Provenance metadata
- FastAPI backend
- Streamlit interface
- Automated testing
- Evaluation artifacts

---

## 20. Responsible Use

This project demonstrates engineering techniques for evidence-grounded retrieval and generation over clinical guidance.

It should not be interpreted as a system that replaces clinicians or provides autonomous medical advice.

Any future production implementation would require appropriate source licensing, clinical evaluation, governance, security, monitoring, and regulatory assessment.

---

## Author

Vamshi

