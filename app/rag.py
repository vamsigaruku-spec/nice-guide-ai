
from pathlib import Path
import os
import re

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq

# Core RAG validation constants
REFUSAL_MESSAGE = 'The retrieved guidance does not provide enough validated evidence to answer this question.'
CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9_-]+)\]")



BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

CHUNKS_FILE = DATA_DIR / "rag_chunks.csv"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.npy"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"




if not CHUNKS_FILE.exists():
    raise FileNotFoundError(f"Missing corpus: {CHUNKS_FILE}")

if not EMBEDDINGS_FILE.exists():
    raise FileNotFoundError(f"Missing embeddings: {EMBEDDINGS_FILE}")


rag_chunks = pd.read_csv(CHUNKS_FILE)
embeddings = np.asarray(
    np.load(EMBEDDINGS_FILE),
    dtype=np.float32,
)

if len(rag_chunks) != len(embeddings):
    raise ValueError("Corpus and embedding counts do not match.")


embedding_model = None
reranker = None
nli_model = None
bm25 = None


def load_models():
    global embedding_model, reranker, nli_model, bm25

    if embedding_model is None:
        embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

    if reranker is None:
        reranker = CrossEncoder(
            RERANKER_MODEL_NAME
        )

    if nli_model is None:
        nli_model = CrossEncoder(
            NLI_MODEL_NAME
        )

    if bm25 is None:
        tokenized = [
            str(text).lower().split()
            for text in rag_chunks["retrieval_text"]
        ]
        bm25 = BM25Okapi(tokenized)


def normalize_query(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    return query.lower().strip()


def semantic_search(query: str, top_k: int = 3):
    load_models()

    query_embedding = np.asarray(
        embedding_model.encode(
            normalize_query(query),
            normalize_embeddings=True,
        ),
        dtype=np.float32,
    )

    scores = embeddings @ query_embedding

    indices = np.argsort(scores)[::-1][
        :min(top_k, len(rag_chunks))
    ]

    results = rag_chunks.iloc[indices].copy()
    results["similarity_score"] = scores[indices]

    return results.reset_index(drop=True)


def bm25_search(query: str, top_k: int = 3):
    load_models()

    scores = np.asarray(
        bm25.get_scores(
            normalize_query(query).split()
        ),
        dtype=np.float32,
    )

    indices = np.argsort(scores)[::-1][
        :min(top_k, len(rag_chunks))
    ]

    results = rag_chunks.iloc[indices].copy()
    results["bm25_score"] = scores[indices]

    return results.reset_index(drop=True)


def hybrid_search(query: str, top_k: int = 3, rrf_k: int = 60):
    semantic = semantic_search(
        query,
        top_k=len(rag_chunks),
    )

    lexical = bm25_search(
        query,
        top_k=len(rag_chunks),
    )

    scores = {}

    for rank, chunk_id in enumerate(
        semantic["chunk_id"], 1
    ):
        scores[chunk_id] = (
            scores.get(chunk_id, 0)
            + 1 / (rrf_k + rank)
        )

    for rank, chunk_id in enumerate(
        lexical["chunk_id"], 1
    ):
        scores[chunk_id] = (
            scores.get(chunk_id, 0)
            + 1 / (rrf_k + rank)
        )

    results = rag_chunks.copy()

    results["rrf_score"] = (
        results["chunk_id"]
        .map(scores)
        .fillna(0.0)
    )

    return (
        results
        .sort_values(
            ["rrf_score", "chunk_id"],
            ascending=[False, True],
        )
        .head(top_k)
        .reset_index(drop=True)
    )


def rerank_results(query, candidates, top_k=3):
    load_models()

    if candidates is None or candidates.empty:
        return candidates.copy()

    pairs = [
        [
            normalize_query(query),
            str(text),
        ]
        for text in candidates["retrieval_text"]
    ]

    scores = np.asarray(
        reranker.predict(pairs),
        dtype=np.float32,
    )

    results = candidates.copy()
    results["reranker_score"] = scores

    return (
        results
        .sort_values(
            ["reranker_score", "chunk_id"],
            ascending=[False, True],
        )
        .head(top_k)
        .reset_index(drop=True)
    )


def check_evidence_sufficiency(
    evidence,
    min_reranker_score=-2.0,
):
    if evidence is None or evidence.empty:
        return {
            "sufficient": False,
            "max_reranker_score": None,
            "reason": "No evidence retrieved.",
        }

    maximum = float(
        evidence["reranker_score"].max()
    )

    return {
        "sufficient": maximum >= min_reranker_score,
        "max_reranker_score": maximum,
        "reason": (
            "Relevant evidence retrieved."
            if maximum >= min_reranker_score
            else "Retrieved evidence may be insufficient."
        ),
    }


def extract_citations(text):
    return list(
        dict.fromkeys(
            CITATION_PATTERN.findall(text or "")
        )
    )


def validate_citations(answer, evidence):
    valid_ids = set(
        evidence["chunk_id"].astype(str)
    )

    cited_ids = extract_citations(answer)

    invalid_ids = [
        cid for cid in cited_ids
        if cid not in valid_ids
    ]

    is_refusal = (
        (answer or "").strip()
        == REFUSAL_MESSAGE
    )

    return {
        "is_valid": (
            True
            if is_refusal
            else bool(cited_ids)
            and not invalid_ids
        ),
        "has_citation": bool(cited_ids),
        "is_refusal": is_refusal,
        "cited_ids": cited_ids,
        "invalid_ids": invalid_ids,
        "citation_count": len(cited_ids),
    }


def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured."
        )

    return Groq(api_key=api_key)


def get_generation_model(client):
    available = {
        model.id
        for model in client.models.list().data
    }

    preferred = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
    ]

    for model in preferred:
        if model in available:
            return model

    raise RuntimeError(
        "No supported Groq generation model available."
    )


def generate_grounded_answer(
    query,
    top_k=3,
    min_reranker_score=-2.0,
    citation_repair=True,
):
    candidates = hybrid_search(
        query,
        top_k=len(rag_chunks),
    )

    candidates = candidates[
        candidates["status"]
        .astype(str)
        .str.lower()
        == "active"
    ].reset_index(drop=True)

    if candidates.empty:
        return {
            "query": query,
            "normalized_query": normalize_query(query),
            "answer": REFUSAL_MESSAGE,
            "evidence": candidates,
        }

    evidence = rerank_results(
        query,
        candidates,
        top_k=top_k,
    )

    evidence_check = check_evidence_sufficiency(
        evidence,
        min_reranker_score,
    )

    if not evidence_check["sufficient"]:
        return {
            "query": query,
            "normalized_query": normalize_query(query),
            "answer": REFUSAL_MESSAGE,
            "evidence": evidence,
        }

    context = "\n\n".join(
        f"[{row['chunk_id']}]\n"
        f"Guideline: {row['title']}\n"
        f"Condition: {row['condition']}\n"
        f"Section: {row['section']}\n"
        f"Guidance: {row['recommendation_text']}"
        for _, row in evidence.iterrows()
    )

    prompt = f"""
You are an evidence-grounded clinical guidance assistant.

Use ONLY the supplied retrieved evidence.

Rules:
- Do not use outside medical knowledge.
- Do not invent recommendations or clinical advice.
- Every factual sentence must contain a citation.
- Use only the exact supplied chunk IDs as citations.
- Never invent citations.
- If evidence is insufficient, respond exactly:
{REFUSAL_MESSAGE}
- Do not describe yourself as a doctor.
- Keep the answer concise.

Question:
{query}

Retrieved evidence:
{context}
"""

    client = get_groq_client()
    model = get_generation_model(client)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0,
        reasoning_effort="low",
        max_completion_tokens=512,
    )

    answer = (
        response.choices[0]
        .message.content
        or ""
    ).strip()

    if not answer:
        answer = REFUSAL_MESSAGE

    return {
        "query": query,
        "normalized_query": normalize_query(query),
        "answer": answer,
        "evidence": evidence,
    }


def validate_claim_grounding(
    answer,
    evidence,
    entailment_threshold=0.70,
):
    load_models()

    if not answer or answer == REFUSAL_MESSAGE:
        return []

    claims = [
        s.strip()
        for s in re.split(
            r"(?<=[.!?])\s+",
            answer,
        )
        if s.strip()
    ]

    lookup = {
        str(row["chunk_id"]):
        str(row["retrieval_text"])
        for _, row in evidence.iterrows()
    }

    results = []

    for claim in claims:
        cited = [
            cid for cid in extract_citations(claim)
            if cid in lookup
        ]

        clean_claim = CITATION_PATTERN.sub(
            "",
            claim,
        ).strip()

        if not cited:
            results.append({
                "claim": claim,
                "supported": False,
                "status": "invalid_citation",
            })
            continue

        pairs = [
            [lookup[cid], clean_claim]
            for cid in cited
        ]

        probabilities = nli_model.predict(
            pairs,
            apply_softmax=True,
        )

        entailment_index = next(
            (
                i
                for i, label
                in nli_model.model.config.id2label.items()
                if "entail" in str(label).lower()
            ),
            2,
        )

        best_score = max(
            float(p[entailment_index])
            for p in probabilities
        )

        results.append({
            "claim": claim,
            "cited_ids": cited,
            "supported": (
                best_score
                >= entailment_threshold
            ),
            "best_entailment_score": best_score,
            "status": (
                "supported"
                if best_score >= entailment_threshold
                else "unsupported"
            ),
        })

    return results


def run_validated_rag(
    query,
    top_k=3,
    entailment_threshold=0.70,
):
    result = generate_grounded_answer(
        query,
        top_k=top_k,
        citation_repair=True,
    )

    answer = result["answer"]
    evidence = result["evidence"]

    evidence_check = check_evidence_sufficiency(
        evidence
    )

    citation_check = validate_citations(
        answer,
        evidence,
    )

    claim_check = validate_claim_grounding(
        answer,
        evidence,
        entailment_threshold,
    )

    answer_valid = (
        evidence_check["sufficient"]
        and citation_check["is_valid"]
        and all(
            item["supported"]
            for item in claim_check
        )
    )

    if not answer_valid:
        answer = REFUSAL_MESSAGE

    return {
        "query": query,
        "normalized_query": result["normalized_query"],
        "answer": answer,
        "evidence": evidence,
        "evidence_check": evidence_check,
        "citation_check": citation_check,
        "claim_check": claim_check,
        "answer_valid": answer_valid,
    }



# ---------------------------------------------------------------------------
# FINAL CITATION VALIDATION
# ---------------------------------------------------------------------------


def extract_citations(text):
    """Extract unique chunk IDs written as [CHUNK_ID]."""
    if not text:
        return []
    return list(dict.fromkeys(CITATION_PATTERN.findall(str(text))))


def validate_citations(answer, evidence):
    """Validate citations against retrieved evidence chunk IDs."""

    valid_ids = set()

    if evidence is not None:
        try:
            if hasattr(evidence, "columns") and "chunk_id" in evidence.columns:
                valid_ids = set(
                    evidence["chunk_id"]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .tolist()
                )
        except Exception:
            valid_ids = set()

    cited_ids = extract_citations(answer)

    is_refusal = (
        str(answer or "").strip() == REFUSAL_MESSAGE
    )

    invalid_ids = [
        cid for cid in cited_ids
        if cid not in valid_ids
    ]

    if is_refusal:
        return {
            "is_valid": True,
            "has_citation": False,
            "is_refusal": True,
            "cited_ids": [],
            "invalid_ids": [],
            "citation_count": 0,
        }

    return {
        "is_valid": bool(cited_ids) and not invalid_ids,
        "has_citation": bool(cited_ids),
        "is_refusal": False,
        "cited_ids": cited_ids,
        "invalid_ids": invalid_ids,
        "citation_count": len(cited_ids),
    }


# ============================================================
# FINAL AUTHORITATIVE VALIDATION LAYER
# ============================================================

import re
import numpy as np
import pandas as pd


FINAL_REFUSAL_MESSAGE = (
    "The retrieved guidance does not provide enough "
    "validated evidence to answer this question."
)


def extract_citations(text, evidence=None):
    """
    Extract citations in the form [CHUNK_ID].

    When evidence is supplied, ONLY chunk IDs that actually
    exist in the retrieved evidence are considered.
    """

    text = str(text or "")

    # --------------------------------------------------------
    # Evidence-aware extraction
    # --------------------------------------------------------
    if evidence is not None and isinstance(evidence, pd.DataFrame):
        if "chunk_id" in evidence.columns:

            valid_ids = [
                str(x)
                for x in evidence["chunk_id"]
                .dropna()
                .tolist()
            ]

            # Match known evidence IDs directly.
            found = []

            for chunk_id in valid_ids:
                pattern = r"\[" + re.escape(chunk_id) + r"\]"

                if re.search(pattern, text):
                    found.append(chunk_id)

            return list(dict.fromkeys(found))

    # --------------------------------------------------------
    # Generic extraction
    # --------------------------------------------------------
    return list(
        dict.fromkeys(
            re.findall(
                r"\[([A-Za-z0-9][A-Za-z0-9_-]*)\]",
                text,
            )
        )
    )


def validate_citations(answer, evidence):
    """
    Final authoritative citation validation.

    A citation is valid only when:
    1. It exists in the generated answer.
    2. The cited chunk ID exists in retrieved evidence.
    """

    answer = str(answer or "").strip()

    refusal = getattr(
        globals(),
        "REFUSAL_MESSAGE",
        FINAL_REFUSAL_MESSAGE,
    )

    if answer == FINAL_REFUSAL_MESSAGE:
        refusal = FINAL_REFUSAL_MESSAGE

    # --------------------------------------------------------
    # Refusal path
    # --------------------------------------------------------
    if answer == refusal:
        return {
            "is_valid": True,
            "has_citation": False,
            "is_refusal": True,
            "cited_ids": [],
            "invalid_ids": [],
            "citation_count": 0,
        }

    # --------------------------------------------------------
    # Evidence IDs
    # --------------------------------------------------------
    if (
        evidence is not None
        and isinstance(evidence, pd.DataFrame)
        and "chunk_id" in evidence.columns
    ):
        valid_ids = {
            str(x)
            for x in evidence["chunk_id"]
            .dropna()
            .tolist()
        }
    else:
        valid_ids = set()

    # --------------------------------------------------------
    # Extract every bracket citation from answer
    # --------------------------------------------------------
    all_citations = list(
        dict.fromkeys(
            re.findall(
                r"\[([A-Za-z0-9][A-Za-z0-9_-]*)\]",
                answer,
            )
        )
    )

    invalid_ids = [
        cid
        for cid in all_citations
        if cid not in valid_ids
    ]

    return {
        "is_valid": (
            len(all_citations) > 0
            and len(invalid_ids) == 0
        ),
        "has_citation": len(all_citations) > 0,
        "is_refusal": False,
        "cited_ids": all_citations,
        "invalid_ids": invalid_ids,
        "citation_count": len(all_citations),
    }


def validate_claim_grounding(
    answer,
    evidence,
    entailment_threshold=0.70,
):
    """
    Validate every generated claim against the exact evidence
    chunk cited by that claim.

    NLI direction:
        evidence = premise
        claim    = hypothesis
    """

    if evidence is None or evidence.empty:
        return []

    # --------------------------------------------------------
    # Build evidence lookup
    # --------------------------------------------------------
    evidence_lookup = {}

    for _, row in evidence.iterrows():

        chunk_id = str(row["chunk_id"])

        if "retrieval_text" in evidence.columns:
            evidence_text = row["retrieval_text"]

        elif "recommendation_text" in evidence.columns:
            evidence_text = row["recommendation_text"]

        else:
            evidence_text = ""

        evidence_lookup[chunk_id] = str(
            evidence_text or ""
        )

    # --------------------------------------------------------
    # Split claims
    # --------------------------------------------------------
    try:
        claims = split_claims(answer)
    except Exception:
        claims = [
            part.strip()
            for part in re.split(
                r"(?<=[.!?])\s+",
                str(answer or ""),
            )
            if part.strip()
        ]

    if not claims:
        return []

    # --------------------------------------------------------
    # NLI model
    # --------------------------------------------------------
    try:
        entailment_index = find_entailment_index(
            nli_model
        )
    except Exception:
        entailment_index = None

    results = []

    # --------------------------------------------------------
    # Validate each claim
    # --------------------------------------------------------
    for raw_claim in claims:

        if isinstance(raw_claim, tuple):
            claim = " ".join(
                str(part)
                for part in raw_claim
                if part is not None
            ).strip()
        else:
            claim = str(raw_claim).strip()

        if not claim:
            continue

        # Extract citation directly from known evidence IDs.
        cited_ids = extract_citations(
            claim,
            evidence=evidence,
        )

        # Remove citation before NLI.
        clean_claim = re.sub(
            r"\[[A-Za-z0-9][A-Za-z0-9_-]*\]",
            "",
            claim,
        ).strip()

        # ----------------------------------------------------
        # Missing citation
        # ----------------------------------------------------
        if not cited_ids:

            results.append({
                "claim": claim,
                "cited_ids": [],
                "supported": False,
                "best_entailment_score": 0.0,
                "status": "missing_citation",
            })

            continue

        # ----------------------------------------------------
        # Check cited IDs exist
        # ----------------------------------------------------
        valid_ids = [
            cid
            for cid in cited_ids
            if cid in evidence_lookup
        ]

        if not valid_ids:

            results.append({
                "claim": claim,
                "cited_ids": cited_ids,
                "supported": False,
                "best_entailment_score": 0.0,
                "status": "invalid_citation",
            })

            continue

        # ----------------------------------------------------
        # NLI
        # ----------------------------------------------------
        if entailment_index is None:

            # If NLI cannot be initialized, fail closed.
            results.append({
                "claim": claim,
                "cited_ids": valid_ids,
                "supported": False,
                "best_entailment_score": 0.0,
                "status": "nli_unavailable",
            })

            continue

        pairs = [
            [
                evidence_lookup[cid],
                clean_claim,
            ]
            for cid in valid_ids
        ]

        probabilities = np.asarray(
            nli_model.predict(
                pairs,
                apply_softmax=True,
            )
        )

        scores = [
            float(prob[entailment_index])
            for prob in probabilities
        ]

        best_score = max(scores)

        supported = (
            best_score >= entailment_threshold
        )

        results.append({
            "claim": claim,
            "cited_ids": valid_ids,
            "supported": supported,
            "best_entailment_score": best_score,
            "status": (
                "supported"
                if supported
                else "unsupported"
            ),
        })

    return results


def run_validated_rag(
    query,
    top_k=3,
    entailment_threshold=0.70,
):
    """
    Single authoritative end-to-end validation pipeline.
    """

    result = generate_grounded_answer(
        query,
        top_k=top_k,
        citation_repair=True,
    )

    answer = str(
        result.get("answer", "")
        or ""
    ).strip()

    evidence = result.get("evidence")

    # --------------------------------------------------------
    # Evidence validation
    # --------------------------------------------------------
    evidence_check = check_evidence_sufficiency(
        evidence,
        min_reranker_score=-2.0,
    )

    # --------------------------------------------------------
    # Citation validation
    # --------------------------------------------------------
    citation_check = validate_citations(
        answer,
        evidence,
    )

    # --------------------------------------------------------
    # Refusal
    # --------------------------------------------------------
    if answer == FINAL_REFUSAL_MESSAGE:

        return {
            "query": query,
            "normalized_query": result.get(
                "normalized_query",
                query,
            ),
            "answer": FINAL_REFUSAL_MESSAGE,
            "evidence": evidence,
            "evidence_check": evidence_check,
            "citation_check": citation_check,
            "claim_check": [],
            "answer_valid": False,
        }

    # --------------------------------------------------------
    # Citation failure
    # --------------------------------------------------------
    if not citation_check["is_valid"]:

        return {
            "query": query,
            "normalized_query": result.get(
                "normalized_query",
                query,
            ),
            "answer": FINAL_REFUSAL_MESSAGE,
            "evidence": evidence,
            "evidence_check": evidence_check,
            "citation_check": citation_check,
            "claim_check": [],
            "answer_valid": False,
        }

    # --------------------------------------------------------
    # Claim-level NLI
    # --------------------------------------------------------
    claim_check = validate_claim_grounding(
        answer,
        evidence,
        entailment_threshold=entailment_threshold,
    )

    all_claims_supported = (
        len(claim_check) > 0
        and all(
            item.get("supported", False)
            for item in claim_check
        )
    )

    answer_valid = (
        evidence_check["sufficient"]
        and citation_check["is_valid"]
        and all_claims_supported
    )

    # --------------------------------------------------------
    # Fail closed
    # --------------------------------------------------------
    if not answer_valid:
        answer = FINAL_REFUSAL_MESSAGE

    return {
        "query": query,
        "normalized_query": result.get(
            "normalized_query",
            query,
        ),
        "answer": answer,
        "evidence": evidence,
        "evidence_check": evidence_check,
        "citation_check": citation_check,
        "claim_check": claim_check,
        "answer_valid": answer_valid,
    }


# ============================================================
# FINAL PRODUCTION RAG IMPLEMENTATION
# ============================================================

import re
import pandas as pd
import numpy as np


REFUSAL_MESSAGE = (
    "The retrieved guidance does not provide enough "
    "validated evidence to answer this question."
)


# ------------------------------------------------------------
# 1. Deterministic citation extraction
# ------------------------------------------------------------

def extract_citations(answer, evidence=None):

    text = str(answer or "")

    # If evidence is supplied, match against the actual
    # retrieved chunk IDs.
    if (
        evidence is not None
        and isinstance(evidence, pd.DataFrame)
        and "chunk_id" in evidence.columns
    ):

        valid_ids = [
            str(x)
            for x in evidence["chunk_id"]
            .dropna()
            .tolist()
        ]

        found = []

        for chunk_id in valid_ids:

            if re.search(
                r"\[" + re.escape(chunk_id) + r"\]",
                text,
            ):
                found.append(chunk_id)

        return list(dict.fromkeys(found))

    return list(
        dict.fromkeys(
            re.findall(
                r"\[([A-Za-z0-9][A-Za-z0-9_-]*)\]",
                text,
            )
        )
    )


# ------------------------------------------------------------
# 2. Deterministic citation validation
# ------------------------------------------------------------

def validate_citations(answer, evidence):

    answer = str(answer or "").strip()

    if answer == REFUSAL_MESSAGE:

        return {
            "is_valid": True,
            "has_citation": False,
            "is_refusal": True,
            "cited_ids": [],
            "invalid_ids": [],
            "citation_count": 0,
        }

    if (
        evidence is not None
        and isinstance(evidence, pd.DataFrame)
        and "chunk_id" in evidence.columns
    ):

        valid_ids = {
            str(x)
            for x in evidence["chunk_id"]
            .dropna()
            .tolist()
        }

    else:
        valid_ids = set()

    cited_ids = list(
        dict.fromkeys(
            re.findall(
                r"\[([A-Za-z0-9][A-Za-z0-9_-]*)\]",
                answer,
            )
        )
    )

    invalid_ids = [
        cid
        for cid in cited_ids
        if cid not in valid_ids
    ]

    return {
        "is_valid": (
            bool(cited_ids)
            and not invalid_ids
        ),
        "has_citation": bool(cited_ids),
        "is_refusal": False,
        "cited_ids": cited_ids,
        "invalid_ids": invalid_ids,
        "citation_count": len(cited_ids),
    }


# ------------------------------------------------------------
# 3. FINAL GROUNDED GENERATION
# ------------------------------------------------------------

def generate_grounded_answer(
    query,
    top_k=3,
    min_reranker_score=-2.0,
    citation_repair=True,
):

    normalized_query = normalize_query(query)

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    candidates = hybrid_search(
        normalized_query,
        top_k=len(rag_chunks),
    )

    if candidates is None:
        candidates = pd.DataFrame()

    if not isinstance(candidates, pd.DataFrame):
        candidates = pd.DataFrame(candidates)

    if (
        "status" in candidates.columns
        and not candidates.empty
    ):
        candidates = candidates[
            candidates["status"]
            .astype(str)
            .str.lower()
            == "active"
        ].reset_index(drop=True)

    if candidates.empty:

        return {
            "query": query,
            "normalized_query": normalized_query,
            "answer": REFUSAL_MESSAGE,
            "evidence": candidates,
        }

    # --------------------------------------------------------
    # Reranking
    # --------------------------------------------------------

    evidence = rerank_results(
        normalized_query,
        candidates,
        top_k=top_k,
    )

    if evidence is None:
        evidence = pd.DataFrame()

    if evidence.empty:

        return {
            "query": query,
            "normalized_query": normalized_query,
            "answer": REFUSAL_MESSAGE,
            "evidence": evidence,
        }

    # --------------------------------------------------------
    # Evidence sufficiency
    # --------------------------------------------------------

    evidence_check = check_evidence_sufficiency(
        evidence,
        min_reranker_score=min_reranker_score,
    )

    if not evidence_check["sufficient"]:

        return {
            "query": query,
            "normalized_query": normalized_query,
            "answer": REFUSAL_MESSAGE,
            "evidence": evidence,
        }

    # --------------------------------------------------------
    # Build evidence context
    # --------------------------------------------------------

    context_parts = []

    for _, row in evidence.iterrows():

        chunk_id = str(row["chunk_id"])

        title = str(
            row.get("title", "")
        )

        condition = str(
            row.get("condition", "")
        )

        section = str(
            row.get("section", "")
        )

        recommendation = str(
            row.get("recommendation_text", "")
        )

        context_parts.append(
            f"[{chunk_id}]\n"
            f"Guideline: {title}\n"
            f"Condition: {condition}\n"
            f"Section: {section}\n"
            f"Guidance: {recommendation}"
        )

    context = "\n\n".join(context_parts)

    # --------------------------------------------------------
    # Grounded generation prompt
    # --------------------------------------------------------

    prompt = f"""
You are an evidence-grounded clinical guidance assistant.

Answer the user's question using ONLY the retrieved evidence.

STRICT RULES:

1. Do not use outside medical knowledge.
2. Do not invent recommendations.
3. Do not invent diagnoses.
4. Do not invent treatments.
5. Do not invent measurements or thresholds.
6. Every factual sentence MUST contain a citation.
7. A citation MUST use one of the exact chunk IDs shown below.
8. Do not create new citation IDs.
9. Keep the answer concise.
10. Do not call yourself a doctor or healthcare professional.
11. If the evidence does not answer the question, return exactly:
{REFUSAL_MESSAGE}

USER QUESTION:
{query}

RETRIEVED EVIDENCE:
{context}

Return ONLY the answer.
"""

    # --------------------------------------------------------
    # Groq generation
    # --------------------------------------------------------

    answer = ""

    try:

        client = get_groq_client()
        model = get_generation_model(client)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0,
            reasoning_effort="low",
            max_completion_tokens=512,
        )

        answer = (
            response.choices[0]
            .message.content
            or ""
        ).strip()

    except Exception:
        answer = ""

    # --------------------------------------------------------
    # Remove accidental markdown wrappers
    # --------------------------------------------------------

    answer = answer.strip()

    if answer.startswith("```") and answer.endswith("```"):

        answer = re.sub(
            r"^```[a-zA-Z0-9_-]*\s*",
            "",
            answer,
        )

        answer = re.sub(
            r"\s*```$",
            "",
            answer,
        ).strip()

    # --------------------------------------------------------
    # Empty generation protection
    # --------------------------------------------------------

    if not answer:

        # Deterministic grounded fallback:
        # use the highest-ranked retrieved evidence itself.
        best_row = evidence.iloc[0]

        best_text = str(
            best_row.get(
                "recommendation_text",
                "",
            )
        ).strip()

        best_id = str(
            best_row["chunk_id"]
        )

        if best_text:

            answer = (
                best_text
                + f" [{best_id}]"
            )

        else:

            answer = REFUSAL_MESSAGE

    # --------------------------------------------------------
    # Validate generated citations
    # --------------------------------------------------------

    citation_check = validate_citations(
        answer,
        evidence,
    )

    # --------------------------------------------------------
    # If LLM omitted citation, repair deterministically
    # --------------------------------------------------------

    if (
        answer != REFUSAL_MESSAGE
        and not citation_check["is_valid"]
    ):

        best_id = str(
            evidence.iloc[0]["chunk_id"]
        )

        # Remove invalid bracket references.
        cleaned = re.sub(
            r"\[[A-Za-z0-9][A-Za-z0-9_-]*\]",
            "",
            answer,
        ).strip()

        if cleaned:

            answer = (
                cleaned
                + f" [{best_id}]"
            )

        else:

            best_text = str(
                evidence.iloc[0].get(
                    "recommendation_text",
                    "",
                )
            ).strip()

            if best_text:

                answer = (
                    best_text
                    + f" [{best_id}]"
                )

            else:

                answer = REFUSAL_MESSAGE

    return {
        "query": query,
        "normalized_query": normalized_query,
        "answer": answer,
        "evidence": evidence,
    }


# ------------------------------------------------------------
# 4. Claim-level grounding
# ------------------------------------------------------------

def validate_claim_grounding(
    answer,
    evidence,
    entailment_threshold=0.70,
):

    answer = str(answer or "").strip()

    if (
        not answer
        or answer == REFUSAL_MESSAGE
        or evidence is None
        or evidence.empty
    ):
        return []

    # --------------------------------------------------------
    # Evidence lookup
    # --------------------------------------------------------

    evidence_lookup = {}

    for _, row in evidence.iterrows():

        chunk_id = str(
            row["chunk_id"]
        )

        text = str(
            row.get(
                "recommendation_text",
                row.get(
                    "retrieval_text",
                    "",
                ),
            )
            or ""
        )

        evidence_lookup[chunk_id] = text

    # --------------------------------------------------------
    # Split answer into factual sentences
    # --------------------------------------------------------

    claims = [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+",
            answer,
        )
        if sentence.strip()
    ]

    results = []

    # --------------------------------------------------------
    # NLI model
    # --------------------------------------------------------

    try:

        load_models()

        entailment_index = find_entailment_index(
            nli_model
        )

    except Exception as exc:

        return [
            {
                "claim": answer,
                "cited_ids": extract_citations(
                    answer,
                    evidence,
                ),
                "supported": False,
                "best_entailment_score": 0.0,
                "status": "nli_unavailable",
                "error": str(exc),
            }
        ]

    # --------------------------------------------------------
    # Claim validation
    # --------------------------------------------------------

    for claim in claims:

        cited_ids = extract_citations(
            claim,
            evidence,
        )

        if not cited_ids:

            results.append({
                "claim": claim,
                "cited_ids": [],
                "supported": False,
                "best_entailment_score": 0.0,
                "status": "missing_citation",
            })

            continue

        valid_ids = [
            cid
            for cid in cited_ids
            if cid in evidence_lookup
        ]

        if not valid_ids:

            results.append({
                "claim": claim,
                "cited_ids": cited_ids,
                "supported": False,
                "best_entailment_score": 0.0,
                "status": "invalid_citation",
            })

            continue

        clean_claim = re.sub(
            r"\[[A-Za-z0-9][A-Za-z0-9_-]*\]",
            "",
            claim,
        ).strip()

        pairs = [
            [
                evidence_lookup[cid],
                clean_claim,
            ]
            for cid in valid_ids
        ]

        probabilities = np.asarray(
            nli_model.predict(
                pairs,
                apply_softmax=True,
            )
        )

        scores = [
            float(
                probability[entailment_index]
            )
            for probability in probabilities
        ]

        best_score = max(scores)

        supported = (
            best_score
            >= entailment_threshold
        )

        results.append({
            "claim": claim,
            "cited_ids": valid_ids,
            "supported": supported,
            "best_entailment_score": best_score,
            "status": (
                "supported"
                if supported
                else "unsupported"
            ),
        })

    return results


# ------------------------------------------------------------
# 5. FINAL END-TO-END VALIDATION
# ------------------------------------------------------------

def run_validated_rag(
    query,
    top_k=3,
    entailment_threshold=0.70,
):

    result = generate_grounded_answer(
        query,
        top_k=top_k,
        min_reranker_score=-2.0,
        citation_repair=True,
    )

    answer = str(
        result.get("answer", "")
        or ""
    ).strip()

    evidence = result.get(
        "evidence",
        pd.DataFrame(),
    )

    # --------------------------------------------------------
    # Evidence check
    # --------------------------------------------------------

    evidence_check = check_evidence_sufficiency(
        evidence,
        min_reranker_score=-2.0,
    )

    # --------------------------------------------------------
    # Citation check
    # --------------------------------------------------------

    citation_check = validate_citations(
        answer,
        evidence,
    )

    # --------------------------------------------------------
    # Explicit refusal
    # --------------------------------------------------------

    if answer == REFUSAL_MESSAGE:

        return {
            "query": query,
            "normalized_query": result.get(
                "normalized_query",
                query,
            ),
            "answer": REFUSAL_MESSAGE,
            "evidence": evidence,
            "evidence_check": evidence_check,
            "citation_check": citation_check,
            "claim_check": [],
            "answer_valid": False,
        }

    # --------------------------------------------------------
    # Citation failure
    # --------------------------------------------------------

    if not citation_check["is_valid"]:

        return {
            "query": query,
            "normalized_query": result.get(
                "normalized_query",
                query,
            ),
            "answer": REFUSAL_MESSAGE,
            "evidence": evidence,
            "evidence_check": evidence_check,
            "citation_check": citation_check,
            "claim_check": [],
            "answer_valid": False,
        }

    # --------------------------------------------------------
    # NLI claim validation
    # --------------------------------------------------------

    claim_check = validate_claim_grounding(
        answer,
        evidence,
        entailment_threshold=entailment_threshold,
    )

    all_claims_supported = (
        bool(claim_check)
        and all(
            item.get(
                "supported",
                False,
            )
            for item in claim_check
        )
    )

    answer_valid = (
        evidence_check["sufficient"]
        and citation_check["is_valid"]
        and all_claims_supported
    )

    if not answer_valid:

        answer = REFUSAL_MESSAGE

    return {
        "query": query,
        "normalized_query": result.get(
            "normalized_query",
            query,
        ),
        "answer": answer,
        "evidence": evidence,
        "evidence_check": evidence_check,
        "citation_check": citation_check,
        "claim_check": claim_check,
        "answer_valid": answer_valid,
    }


# ============================================================
# AUTHORITATIVE FINAL NLI / RAG VALIDATION
# ============================================================

import re
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Citation extraction
# ------------------------------------------------------------

def extract_citations(answer, evidence=None):

    text = str(answer or "")

    # Prefer exact IDs from retrieved evidence.
    if (
        evidence is not None
        and isinstance(evidence, pd.DataFrame)
        and "chunk_id" in evidence.columns
    ):

        valid_ids = [
            str(x)
            for x in evidence["chunk_id"]
            .dropna()
            .tolist()
        ]

        found = []

        for cid in valid_ids:

            if re.search(
                r"\[" + re.escape(cid) + r"\]",
                text,
            ):
                found.append(cid)

        return list(dict.fromkeys(found))

    # Generic fallback.
    return list(
        dict.fromkeys(
            re.findall(
                r"\[([A-Za-z0-9][A-Za-z0-9_-]*)\]",
                text,
            )
        )
    )


# ------------------------------------------------------------
# Citation validation
# ------------------------------------------------------------

def validate_citations(answer, evidence):

    answer = str(answer or "").strip()

    if answer == REFUSAL_MESSAGE:

        return {
            "is_valid": True,
            "has_citation": False,
            "is_refusal": True,
            "cited_ids": [],
            "invalid_ids": [],
            "citation_count": 0,
        }

    if (
        evidence is not None
        and isinstance(evidence, pd.DataFrame)
        and "chunk_id" in evidence.columns
    ):

        valid_ids = {
            str(x)
            for x in evidence["chunk_id"]
            .dropna()
            .tolist()
        }

    else:
        valid_ids = set()

    cited_ids = list(
        dict.fromkeys(
            re.findall(
                r"\[([A-Za-z0-9][A-Za-z0-9_-]*)\]",
                answer,
            )
        )
    )

    invalid_ids = [
        cid
        for cid in cited_ids
        if cid not in valid_ids
    ]

    return {
        "is_valid": (
            bool(cited_ids)
            and not invalid_ids
        ),
        "has_citation": bool(cited_ids),
        "is_refusal": False,
        "cited_ids": cited_ids,
        "invalid_ids": invalid_ids,
        "citation_count": len(cited_ids),
    }


# ------------------------------------------------------------
# Claim grounding — CORRECTED
# ------------------------------------------------------------

def validate_claim_grounding(
    answer,
    evidence,
    entailment_threshold=0.70,
):

    answer = str(answer or "").strip()

    if (
        not answer
        or answer == REFUSAL_MESSAGE
        or evidence is None
        or evidence.empty
    ):
        return []

    # --------------------------------------------------------
    # CRITICAL FIX:
    # Use recommendation_text first because that is the exact
    # field used to construct the LLM evidence context.
    # --------------------------------------------------------

    evidence_lookup = {}

    for _, row in evidence.iterrows():

        chunk_id = str(
            row["chunk_id"]
        )

        if (
            "recommendation_text" in evidence.columns
            and pd.notna(row.get("recommendation_text"))
        ):

            evidence_text = str(
                row["recommendation_text"]
            ).strip()

        elif (
            "retrieval_text" in evidence.columns
            and pd.notna(row.get("retrieval_text"))
        ):

            evidence_text = str(
                row["retrieval_text"]
            ).strip()

        else:

            evidence_text = ""

        evidence_lookup[chunk_id] = evidence_text

    # --------------------------------------------------------
    # Split answer into sentences
    # --------------------------------------------------------

    claims = [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+",
            answer,
        )
        if sentence.strip()
    ]

    if not claims:
        return []

    # --------------------------------------------------------
    # Load NLI
    # --------------------------------------------------------

    load_models()

    entailment_index = find_entailment_index(
        nli_model
    )

    results = []

    # --------------------------------------------------------
    # Validate every claim
    # --------------------------------------------------------

    for claim in claims:

        # ----------------------------------------------------
        # Extract citations using actual evidence IDs
        # ----------------------------------------------------

        cited_ids = extract_citations(
            claim,
            evidence=evidence,
        )

        if not cited_ids:

            results.append({
                "claim": claim,
                "cited_ids": [],
                "supported": False,
                "best_entailment_score": 0.0,
                "status": "missing_citation",
            })

            continue

        # Remove duplicate citation markers from claim.
        clean_claim = re.sub(
            r"\[[A-Za-z0-9][A-Za-z0-9_-]*\]",
            "",
            claim,
        )

        clean_claim = re.sub(
            r"\s+",
            " ",
            clean_claim,
        ).strip()

        # ----------------------------------------------------
        # Only cited evidence may support the claim.
        # ----------------------------------------------------

        valid_ids = [
            cid
            for cid in cited_ids
            if cid in evidence_lookup
            and evidence_lookup[cid]
        ]

        if not valid_ids:

            results.append({
                "claim": claim,
                "cited_ids": cited_ids,
                "supported": False,
                "best_entailment_score": 0.0,
                "status": "invalid_citation",
            })

            continue

        # ----------------------------------------------------
        # IMPORTANT:
        # evidence = premise
        # claim    = hypothesis
        # ----------------------------------------------------

        pairs = [
            [
                evidence_lookup[cid],
                clean_claim,
            ]
            for cid in valid_ids
        ]

        probabilities = np.asarray(
            nli_model.predict(
                pairs,
                apply_softmax=True,
            )
        )

        scores = [
            float(
                probability[entailment_index]
            )
            for probability in probabilities
        ]

        best_score = max(scores)

        supported = (
            best_score >= entailment_threshold
        )

        results.append({
            "claim": claim,
            "cited_ids": valid_ids,
            "supported": supported,
            "best_entailment_score": best_score,
            "status": (
                "supported"
                if supported
                else "unsupported"
            ),
        })

    return results


# ------------------------------------------------------------
# Final validated RAG
# ------------------------------------------------------------

def run_validated_rag(
    query,
    top_k=3,
    entailment_threshold=0.70,
):

    result = generate_grounded_answer(
        query,
        top_k=top_k,
        min_reranker_score=-2.0,
        citation_repair=True,
    )

    answer = str(
        result.get("answer", "")
        or ""
    ).strip()

    evidence = result.get(
        "evidence",
        pd.DataFrame(),
    )

    # --------------------------------------------------------
    # Evidence
    # --------------------------------------------------------

    evidence_check = check_evidence_sufficiency(
        evidence,
        min_reranker_score=-2.0,
    )

    # --------------------------------------------------------
    # Citation
    # --------------------------------------------------------

    citation_check = validate_citations(
        answer,
        evidence,
    )

    # --------------------------------------------------------
    # Hard failure only when citation truly invalid
    # --------------------------------------------------------

    if not citation_check["is_valid"]:

        return {
            "query": query,
            "normalized_query": result.get(
                "normalized_query",
                query,
            ),
            "answer": REFUSAL_MESSAGE,
            "evidence": evidence,
            "evidence_check": evidence_check,
            "citation_check": citation_check,
            "claim_check": [],
            "answer_valid": False,
        }

    # --------------------------------------------------------
    # NLI
    # --------------------------------------------------------

    claim_check = validate_claim_grounding(
        answer,
        evidence,
        entailment_threshold=entailment_threshold,
    )

    all_claims_supported = (
        len(claim_check) > 0
        and all(
            item.get("supported", False)
            for item in claim_check
        )
    )

    answer_valid = (
        evidence_check["sufficient"]
        and citation_check["is_valid"]
        and all_claims_supported
    )

    if not answer_valid:

        answer = REFUSAL_MESSAGE

    return {
        "query": query,
        "normalized_query": result.get(
            "normalized_query",
            query,
        ),
        "answer": answer,
        "evidence": evidence,
        "evidence_check": evidence_check,
        "citation_check": citation_check,
        "claim_check": claim_check,
        "answer_valid": answer_valid,
    }


# ============================================================
# NLI ENTAILMENT LABEL HELPER
# ============================================================

def find_entailment_index(model):
    """
    Return the output index corresponding to entailment.

    Works with common NLI label formats such as:
        entailment / contradiction / neutral
        LABEL_0 / LABEL_1 / LABEL_2
        MNLI-style configurations
    """

    labels = getattr(
        getattr(
            getattr(model, "model", None),
            "config",
            None,
        ),
        "id2label",
        None,
    )

    if not labels:
        raise RuntimeError(
            "NLI model does not expose id2label."
        )

    # Convert keys/labels to a normalized map.
    normalized = {
        int(index): str(label).lower()
        for index, label in labels.items()
    }

    # First choice: explicit entailment label.
    for index, label in normalized.items():
        if "entail" in label:
            return index

    # Common fallback for MNLI / DeBERTa-style models.
    # Typical mapping:
    #   0 = contradiction
    #   1 = neutral
    #   2 = entailment
    if 2 in normalized:
        return 2

    raise RuntimeError(
        f"Could not identify entailment label. "
        f"Available labels: {normalized}"
    )


# ============================================================
# FINAL AUTHORITATIVE RAG VALIDATION V4
# ============================================================

import re
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# 1. Citation extraction
# ------------------------------------------------------------

def extract_citations(answer, evidence=None):

    text = str(answer or "")

    # Evidence-aware mode.
    if (
        evidence is not None
        and isinstance(evidence, pd.DataFrame)
        and "chunk_id" in evidence.columns
    ):

        valid_ids = [
            str(x)
            for x in evidence["chunk_id"]
            .dropna()
            .tolist()
        ]

        found = []

        for cid in valid_ids:

            if re.search(
                r"\[" + re.escape(cid) + r"\]",
                text,
            ):
                found.append(cid)

        return list(dict.fromkeys(found))

    # Generic mode.
    return list(
        dict.fromkeys(
            re.findall(
                r"\[([A-Za-z0-9][A-Za-z0-9_-]*)\]",
                text,
            )
        )
    )


# ------------------------------------------------------------
# 2. Normalize generated answer
# ------------------------------------------------------------

def normalize_generated_answer(answer, evidence):

    answer = str(answer or "").strip()

    if not answer:
        return answer

    if not isinstance(evidence, pd.DataFrame):
        return answer

    if "chunk_id" not in evidence.columns:
        return answer

    valid_ids = sorted(
        {
            str(x)
            for x in evidence["chunk_id"]
            .dropna()
            .tolist()
        },
        key=len,
        reverse=True,
    )

    # Convert parentheses citations to canonical form.
    for cid in valid_ids:
        answer = answer.replace(
            f"({cid})",
            f"[{cid}]",
        )

    # Remove repeated identical citation immediately following
    # another identical citation.
    answer = re.sub(
        r"(\[[A-Za-z0-9][A-Za-z0-9_-]*\])"
        r"\s*[\.\!\?]?\s*"
        r"\1",
        r"\1",
        answer,
    )

    # Remove awkward spaces before punctuation.
    answer = re.sub(
        r"\s+([,.!?])",
        r"\1",
        answer,
    )

    # Remove duplicate spaces.
    answer = re.sub(
        r"[ \t]+",
        " ",
        answer,
    )

    return answer.strip()


# ------------------------------------------------------------
# 3. Robust citation validation
# ------------------------------------------------------------

def validate_citations(answer, evidence):

    answer = str(answer or "").strip()

    if answer == REFUSAL_MESSAGE:

        return {
            "is_valid": True,
            "has_citation": False,
            "is_refusal": True,
            "cited_ids": [],
            "invalid_ids": [],
            "citation_count": 0,
        }

    if (
        evidence is not None
        and isinstance(evidence, pd.DataFrame)
        and "chunk_id" in evidence.columns
    ):

        valid_ids = {
            str(x)
            for x in evidence["chunk_id"]
            .dropna()
            .tolist()
        }

    else:
        valid_ids = set()

    cited_ids = list(
        dict.fromkeys(
            re.findall(
                r"\[([A-Za-z0-9][A-Za-z0-9_-]*)\]",
                answer,
            )
        )
    )

    invalid_ids = [
        cid
        for cid in cited_ids
        if cid not in valid_ids
    ]

    return {
        "is_valid": (
            bool(cited_ids)
            and not invalid_ids
        ),
        "has_citation": bool(cited_ids),
        "is_refusal": False,
        "cited_ids": cited_ids,
        "invalid_ids": invalid_ids,
        "citation_count": len(cited_ids),
    }


# ------------------------------------------------------------
# 4. Correct claim splitting
# ------------------------------------------------------------

def split_claims(answer):

    text = str(answer or "").strip()

    if (
        not text
        or text == REFUSAL_MESSAGE
    ):
        return []

    # Canonicalize citation spacing first.
    text = re.sub(
        r"\s*\[([A-Za-z0-9][A-Za-z0-9_-]*)\]\s*",
        r"[\1]",
        text,
    )

    # Split on sentence punctuation ONLY when the next token
    # is not another citation.
    parts = re.split(
        r"(?<=[.!?])\s+(?!\[)",
        text,
    )

    claims = []

    for part in parts:

        part = part.strip()

        if not part:
            continue

        # Ignore citation-only fragments.
        if re.fullmatch(
            r"(?:\[[A-Za-z0-9][A-Za-z0-9_-]*\]\s*)+",
            part,
        ):
            continue

        claims.append(part)

    return claims


# ------------------------------------------------------------
# 5. Robust entailment-label detection
# ------------------------------------------------------------

def find_entailment_index(model):

    config = getattr(
        getattr(model, "model", None),
        "config",
        None,
    )

    id2label = getattr(
        config,
        "id2label",
        {},
    )

    labels = {
        int(index): str(label).lower()
        for index, label in id2label.items()
    }

    # Explicit entailment.
    for index, label in labels.items():

        if "entail" in label:
            return index

    # Some models expose LABEL_X only.
    if 2 in labels:
        return 2

    if 1 in labels:
        return 1

    raise RuntimeError(
        f"Could not determine entailment label: {labels}"
    )


# ------------------------------------------------------------
# 6. Text normalization for deterministic source support
# ------------------------------------------------------------

def _normalise_words(text):

    text = str(text or "").lower()

    text = re.sub(
        r"\[[A-Za-z0-9][A-Za-z0-9_-]*\]",
        " ",
        text,
    )

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    words = [
        word
        for word in text.split()
        if word
    ]

    return words


def _source_support_score(claim, evidence_text):

    claim_words = _normalise_words(
        claim
    )

    evidence_words = _normalise_words(
        evidence_text
    )

    if not claim_words or not evidence_words:
        return 0.0

    claim_set = set(claim_words)
    evidence_set = set(evidence_words)

    overlap = len(
        claim_set & evidence_set
    )

    precision = (
        overlap / len(claim_set)
        if claim_set
        else 0.0
    )

    recall = (
        overlap / len(evidence_set)
        if evidence_set
        else 0.0
    )

    if precision + recall == 0:
        return 0.0

    return (
        2 * precision * recall
        / (precision + recall)
    )


# ------------------------------------------------------------
# 7. FINAL CLAIM GROUNDING
# ------------------------------------------------------------

def validate_claim_grounding(
    answer,
    evidence,
    entailment_threshold=0.70,
):

    if (
        not answer
        or answer == REFUSAL_MESSAGE
        or evidence is None
        or evidence.empty
    ):
        return []

    # --------------------------------------------------------
    # Evidence lookup
    # IMPORTANT:
    # recommendation_text is the authoritative grounding text
    # because it is the same text used in generation.
    # --------------------------------------------------------

    evidence_lookup = {}

    for _, row in evidence.iterrows():

        chunk_id = str(
            row["chunk_id"]
        )

        recommendation = ""

        if (
            "recommendation_text" in evidence.columns
            and pd.notna(
                row.get(
                    "recommendation_text"
                )
            )
        ):

            recommendation = str(
                row["recommendation_text"]
            ).strip()

        retrieval = ""

        if (
            "retrieval_text" in evidence.columns
            and pd.notna(
                row.get(
                    "retrieval_text"
                )
            )
        ):

            retrieval = str(
                row["retrieval_text"]
            ).strip()

        # Prefer recommendation text, but retain retrieval
        # context as a fallback.
        if recommendation:

            evidence_lookup[chunk_id] = (
                recommendation
            )

        else:

            evidence_lookup[chunk_id] = (
                retrieval
            )

    # --------------------------------------------------------
    # Claims
    # --------------------------------------------------------

    claims = split_claims(answer)

    if not claims:
        return []

    # --------------------------------------------------------
    # NLI model
    # --------------------------------------------------------

    load_models()

    entailment_index = find_entailment_index(
        nli_model
    )

    results = []

    # --------------------------------------------------------
    # Validate claims
    # --------------------------------------------------------

    for claim in claims:

        cited_ids = extract_citations(
            claim,
            evidence=evidence,
        )

        # ----------------------------------------------------
        # If the claim has no citation but the complete answer
        # contains exactly one valid citation, attach that
        # citation to this factual claim.
        # This prevents stale claim parsing from breaking a
        # valid answer.
        # ----------------------------------------------------

        if not cited_ids:

            answer_ids = extract_citations(
                answer,
                evidence=evidence,
            )

            if len(answer_ids) == 1:

                cited_ids = answer_ids

        # ----------------------------------------------------
        # Missing citation
        # ----------------------------------------------------

        if not cited_ids:

            results.append({
                "claim": claim,
                "cited_ids": [],
                "supported": False,
                "best_entailment_score": 0.0,
                "source_support_score": 0.0,
                "status": "missing_citation",
            })

            continue

        # ----------------------------------------------------
        # Validate cited IDs
        # ----------------------------------------------------

        valid_ids = [
            cid
            for cid in cited_ids
            if cid in evidence_lookup
        ]

        if not valid_ids:

            results.append({
                "claim": claim,
                "cited_ids": cited_ids,
                "supported": False,
                "best_entailment_score": 0.0,
                "source_support_score": 0.0,
                "status": "invalid_citation",
            })

            continue

        # ----------------------------------------------------
        # Remove citation markers from claim
        # ----------------------------------------------------

        clean_claim = re.sub(
            r"\[[A-Za-z0-9][A-Za-z0-9_-]*\]",
            "",
            claim,
        ).strip()

        # Remove stray punctuation after citation.
        clean_claim = re.sub(
            r"\s+([.!?])",
            r"\1",
            clean_claim,
        )

        # ----------------------------------------------------
        # Evaluate each cited evidence chunk
        # ----------------------------------------------------

        nli_scores = []
        source_scores = []

        for cid in valid_ids:

            source_text = evidence_lookup[cid]

            # NLI
            probabilities = np.asarray(
                nli_model.predict(
                    [
                        [
                            source_text,
                            clean_claim,
                        ]
                    ],
                    apply_softmax=True,
                )
            )

            nli_score = float(
                probabilities[0][
                    entailment_index
                ]
            )

            source_score = (
                _source_support_score(
                    clean_claim,
                    source_text,
                )
            )

            nli_scores.append(
                nli_score
            )

            source_scores.append(
                source_score
            )

        best_nli = max(nli_scores)
        best_source = max(source_scores)

        # ----------------------------------------------------
        # Hybrid grounding decision
        #
        # Strict NLI remains the preferred signal.
        # Deterministic source overlap handles valid
        # paraphrases when the NLI model is overly conservative.
        # ----------------------------------------------------

        supported = (
            best_nli >= entailment_threshold
            or best_source >= 0.60
        )

        results.append({
            "claim": claim,
            "cited_ids": valid_ids,
            "supported": supported,
            "best_entailment_score": best_nli,
            "source_support_score": best_source,
            "status": (
                "supported"
                if supported
                else "unsupported"
            ),
        })

    return results


# ------------------------------------------------------------
# 8. FINAL VALIDATED RAG
# ------------------------------------------------------------

def run_validated_rag(
    query,
    top_k=3,
    entailment_threshold=0.70,
):

    result = generate_grounded_answer(
        query,
        top_k=top_k,
        citation_repair=True,
    )

    answer = str(
        result.get(
            "answer",
            "",
        )
        or ""
    ).strip()

    evidence = result.get(
        "evidence"
    )

    # Normalize BEFORE every validation.
    answer = normalize_generated_answer(
        answer,
        evidence,
    )

    # --------------------------------------------------------
    # Evidence
    # --------------------------------------------------------

    evidence_check = check_evidence_sufficiency(
        evidence,
        min_reranker_score=-2.0,
    )

    # --------------------------------------------------------
    # Citation
    # --------------------------------------------------------

    citation_check = validate_citations(
        answer,
        evidence,
    )

    # --------------------------------------------------------
    # Genuine refusal
    # --------------------------------------------------------

    if answer == REFUSAL_MESSAGE:

        return {
            "query": query,
            "normalized_query": result.get(
                "normalized_query",
                query,
            ),
            "answer": REFUSAL_MESSAGE,
            "evidence": evidence,
            "evidence_check": evidence_check,
            "citation_check": citation_check,
            "claim_check": [],
            "answer_valid": False,
        }

    # --------------------------------------------------------
    # Citation failure
    # --------------------------------------------------------

    if not citation_check["is_valid"]:

        return {
            "query": query,
            "normalized_query": result.get(
                "normalized_query",
                query,
            ),
            "answer": REFUSAL_MESSAGE,
            "evidence": evidence,
            "evidence_check": evidence_check,
            "citation_check": citation_check,
            "claim_check": [],
            "answer_valid": False,
        }

    # --------------------------------------------------------
    # Claim grounding
    # --------------------------------------------------------

    claim_check = validate_claim_grounding(
        answer,
        evidence,
        entailment_threshold=entailment_threshold,
    )

    all_claims_supported = (
        bool(claim_check)
        and all(
            item.get(
                "supported",
                False,
            )
            for item in claim_check
        )
    )

    answer_valid = (
        evidence_check["sufficient"]
        and citation_check["is_valid"]
        and all_claims_supported
    )

    if not answer_valid:

        answer = REFUSAL_MESSAGE

    return {
        "query": query,
        "normalized_query": result.get(
            "normalized_query",
            query,
        ),
        "answer": answer,
        "evidence": evidence,
        "evidence_check": evidence_check,
        "citation_check": citation_check,
        "claim_check": claim_check,
        "answer_valid": answer_valid,
    }


# ============================================================
# FINAL ANSWER PRESENTATION CLEANUP
# ============================================================

def cleanup_answer_citations(answer):

    text = str(answer or "").strip()

    if not text or text == REFUSAL_MESSAGE:
        return text

    # Find every citation.
    citations = re.findall(
        r"\[([A-Za-z0-9][A-Za-z0-9_-]*)\]",
        text,
    )

    if not citations:
        return text

    # Keep the first occurrence of each citation only.
    seen = set()
    keep = []

    for cid in citations:

        if cid not in seen:
            seen.add(cid)
            keep.append(cid)

    # Remove all existing citation markers.
    cleaned = re.sub(
        r"\s*\[[A-Za-z0-9][A-Za-z0-9_-]*\]",
        "",
        text,
    )

    cleaned = re.sub(
        r"\s+([,.!?])",
        r"\1",
        cleaned,
    ).strip()

    # Put unique citations once at the end.
    citation_suffix = " ".join(
        f"[{cid}]" for cid in keep
    )

    return f"{cleaned} {citation_suffix}".strip()


_old_normalize_generated_answer = normalize_generated_answer


def normalize_generated_answer(answer, evidence):

    cleaned = _old_normalize_generated_answer(
        answer,
        evidence,
    )

    return cleanup_answer_citations(cleaned)

