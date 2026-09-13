
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import rag


app = FastAPI(
    title="NICE Guide AI",
    version="1.0.0",
    description="Evidence-grounded clinical guidance RAG API.",
)


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


def json_safe(value: Any):
    if value is None:
        return None

    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass

    if isinstance(value, float):
        return value if value == value else None

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    return value


def dataframe_records(df):
    if df is None:
        return []

    return json_safe(df.to_dict(orient="records"))


@app.get("/")
def root():
    return {
        "name": "NICE Guide AI",
        "status": "ok",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/ask")
def ask(request: AskRequest):
    try:
        result = rag.run_validated_rag(
            request.query,
            top_k=request.top_k,
        )

        return {
            "query": request.query,
            "top_k": request.top_k,
            "retrieval_method": "hybrid_rrf_plus_cross_encoder",
            "normalized_query": result["normalized_query"],
            "answer": result["answer"],
            "evidence": dataframe_records(result["evidence"]),
            "evidence_check": json_safe(result["evidence_check"]),
            "citation_check": json_safe(result["citation_check"]),
            "claim_check": json_safe(result["claim_check"]),
            "answer_valid": bool(result["answer_valid"]),
        }

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/retrieve")
def retrieve(request: AskRequest):
    try:
        normalized = rag.normalize_query(request.query)

        candidates = rag.hybrid_search(
            normalized,
            top_k=len(rag.rag_chunks),
        )

        candidates = candidates[
            candidates["status"].astype(str).str.lower() == "active"
        ]

        reranked = rag.rerank_results(
            normalized,
            candidates,
            top_k=request.top_k,
        )

        return {
            "query": request.query,
            "top_k": request.top_k,
            "retrieval_method": "hybrid_rrf_plus_cross_encoder",
            "normalized_query": normalized,
            "results": dataframe_records(reranked),
        }

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
