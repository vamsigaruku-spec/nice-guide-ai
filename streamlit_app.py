
import os
import sys
import time
from pathlib import Path

import streamlit as st


# ------------------------------------------------------------
# App path
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


# ------------------------------------------------------------
# Streamlit secrets -> environment
# rag.py reads GROQ_API_KEY from os.environ
# ------------------------------------------------------------

if "GROQ_API_KEY" not in os.environ:

    try:
        if "GROQ_API_KEY" in st.secrets:
            os.environ["GROQ_API_KEY"] = str(
                st.secrets["GROQ_API_KEY"]
            ).strip()
    except Exception:
        pass


# ------------------------------------------------------------
# Import backend AFTER secret initialization
# ------------------------------------------------------------

from app.rag import run_validated_rag


# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="NICE Guide AI",
    page_icon="🩺",
    layout="wide",
)


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

st.title("🩺 NICE Guide AI")

st.caption(
    "Evidence-grounded clinical guidance retrieval "
    "and summarisation"
)

st.warning(
    "Development / portfolio prototype only. "
    "Not a medical device, clinical decision-support system, "
    "or substitute for professional clinical judgement."
)


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

with st.sidebar:

    st.header("RAG Configuration")

    top_k = st.slider(
        "Retrieved evidence",
        min_value=1,
        max_value=5,
        value=3,
    )

    entailment_threshold = st.slider(
        "NLI threshold",
        min_value=0.50,
        max_value=0.95,
        value=0.70,
        step=0.05,
    )

    st.divider()

    st.subheader("Pipeline")

    st.write("✓ Query normalisation")
    st.write("✓ Semantic retrieval")
    st.write("✓ BM25 retrieval")
    st.write("✓ Reciprocal Rank Fusion")
    st.write("✓ Cross-encoder reranking")
    st.write("✓ Evidence sufficiency")
    st.write("✓ Groq grounded generation")
    st.write("✓ Citation validation")
    st.write("✓ NLI claim grounding")
    st.write("✓ Fail-closed validation")


# ------------------------------------------------------------
# Main question interface
# ------------------------------------------------------------

st.subheader("Ask a clinical-guidance question")

query = st.text_area(
    "Question",
    placeholder=(
        "Example: How should suspected hypertension be assessed?"
    ),
    height=110,
)


# ------------------------------------------------------------
# Execute RAG
# ------------------------------------------------------------

if st.button(
    "Generate grounded answer",
    type="primary",
):

    if not isinstance(query, str) or not query.strip():

        st.error(
            "Please enter a clinical-guidance question."
        )

    elif not os.getenv("GROQ_API_KEY", "").strip():

        st.error(
            "GROQ_API_KEY is not configured. "
            "Add it to Streamlit Secrets."
        )

    else:

        start = time.perf_counter()

        with st.spinner(
            "Retrieving evidence and validating answer..."
        ):

            try:

                result = run_validated_rag(
                    query.strip(),
                    top_k=top_k,
                    entailment_threshold=(
                        entailment_threshold
                    ),
                )

            except Exception as exc:

                st.error(
                    f"RAG execution failed: "
                    f"{type(exc).__name__}: {exc}"
                )

                st.stop()

        latency_ms = (
            time.perf_counter() - start
        ) * 1000


        # ----------------------------------------------------
        # Answer
        # ----------------------------------------------------

        st.subheader("Answer")

        st.write(
            result["answer"]
        )


        # ----------------------------------------------------
        # Status metrics
        # ----------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Evidence",
                (
                    "Sufficient"
                    if result[
                        "evidence_check"
                    ]["sufficient"]
                    else "Insufficient"
                ),
            )

        with col2:

            st.metric(
                "Citations",
                (
                    "Valid"
                    if result[
                        "citation_check"
                    ]["is_valid"]
                    else "Invalid"
                ),
            )

        with col3:

            st.metric(
                "Answer",
                (
                    "Validated"
                    if result["answer_valid"]
                    else "Refused"
                ),
            )

        with col4:

            st.metric(
                "Latency",
                f"{latency_ms:.0f} ms",
            )


        # ----------------------------------------------------
        # Citation validation
        # ----------------------------------------------------

        with st.expander(
            "Citation validation",
            expanded=True,
        ):

            st.json(
                result["citation_check"]
            )


        # ----------------------------------------------------
        # Claim validation
        # ----------------------------------------------------

        with st.expander(
            "Claim-level validation"
        ):

            if result["claim_check"]:

                st.json(
                    result["claim_check"]
                )

            else:

                st.info(
                    "No claims were generated."
                )


        # ----------------------------------------------------
        # Retrieved evidence
        # ----------------------------------------------------

        with st.expander(
            "Retrieved evidence"
        ):

            evidence = result["evidence"]

            if evidence.empty:

                st.info(
                    "No evidence retrieved."
                )

            else:

                for _, row in evidence.iterrows():

                    st.markdown(
                        f"### [{row['chunk_id']}] "
                        f"{row['section']}"
                    )

                    st.write(
                        row["recommendation_text"]
                    )

                    if (
                        "reranker_score"
                        in evidence.columns
                    ):

                        st.caption(
                            "Reranker score: "
                            f"{row['reranker_score']}"
                        )


        # ----------------------------------------------------
        # Raw diagnostics
        # ----------------------------------------------------

        with st.expander(
            "Evidence sufficiency"
        ):

            st.json(
                result["evidence_check"]
            )


st.divider()

st.caption(
    "NICE Guide AI — development / portfolio prototype "
    "using a synthetic development corpus."
)
