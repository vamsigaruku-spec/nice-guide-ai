import os
import time

import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NICE Guide AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STREAMLIT SECRETS
# Must be loaded BEFORE importing app.rag
# ============================================================

try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass


# ============================================================
# BACKEND IMPORT
# ============================================================

from app.rag import run_validated_rag


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.7rem;
        font-weight: 800;
        margin-bottom: 0.1rem;
    }

    .subtitle {
        color: #9aa3b2;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }

    .answer-card {
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.035);
        margin-top: 0.5rem;
    }

    .evidence-title {
        font-size: 1.15rem;
        font-weight: 750;
        margin-bottom: 0.25rem;
    }

    .evidence-meta {
        color: #9aa3b2;
        font-size: 0.88rem;
        margin-bottom: 0.7rem;
    }

    .section-label {
        font-size: 1.05rem;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🩺 NICE Guide AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Evidence-grounded clinical guidance retrieval and summarisation'
    '</div>',
    unsafe_allow_html=True,
)

st.warning(
    "Development / portfolio prototype only. "
    "Uses a synthetic clinical-guidance development corpus. "
    "Not a medical device, clinical decision-support system, "
    "or substitute for professional clinical judgement."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("RAG Configuration")

    top_k = st.slider(
        "Evidence chunks",
        min_value=1,
        max_value=5,
        value=3,
        step=1,
        help="Number of evidence chunks retrieved for the answer.",
    )

    entailment_threshold = st.slider(
        "NLI validation threshold",
        min_value=0.50,
        max_value=0.95,
        value=0.70,
        step=0.05,
        help="Threshold used for claim-level entailment validation.",
    )

    st.divider()

    st.subheader("Pipeline")

    pipeline = [
        "Semantic retrieval",
        "BM25 retrieval",
        "Reciprocal Rank Fusion",
        "Cross-encoder reranking",
        "Evidence sufficiency",
        "Grounded generation",
        "Citation validation",
        "NLI claim validation",
    ]

    for step in pipeline:
        st.write(f"✓ {step}")

    st.divider()

    st.subheader("Example questions")

    st.caption(
        "How should suspected hypertension be assessed?"
    )

    st.caption(
        "What education should be provided to adults with asthma?"
    )

    st.caption(
        "What lifestyle information is relevant for adults with type 2 diabetes?"
    )


# ============================================================
# QUESTION INPUT
# ============================================================

st.subheader("Ask a clinical-guidance question")

query = st.text_area(
    "Question",
    placeholder=(
        "Example: What education should be provided "
        "to adults with asthma?"
    ),
    height=115,
)


# ============================================================
# RUN BUTTON
# ============================================================

generate = st.button(
    "Generate grounded answer",
    type="primary",
)


# ============================================================
# EXECUTE RAG PIPELINE
# ============================================================

if generate:

    if not isinstance(query, str) or not query.strip():

        st.error(
            "Please enter a clinical-guidance question."
        )

    else:

        start_time = time.perf_counter()

        with st.spinner(
            "Retrieving evidence and validating the answer..."
        ):

            try:

                result = run_validated_rag(
                    query.strip(),
                    top_k=top_k,
                    entailment_threshold=entailment_threshold,
                )

                latency_ms = (
                    time.perf_counter() - start_time
                ) * 1000

                st.session_state["result"] = result
                st.session_state["latency_ms"] = latency_ms

            except Exception as exc:

                st.error(
                    "The application encountered an error."
                )

                with st.expander("Technical details"):

                    st.exception(exc)

                st.stop()


# ============================================================
# DISPLAY STORED RESULT
# ============================================================

if "result" in st.session_state:

    result = st.session_state["result"]

    latency_ms = st.session_state.get(
        "latency_ms",
        0,
    )

    answer = str(
        result.get(
            "answer",
            "",
        )
    )

    evidence = result.get(
        "evidence",
        pd.DataFrame(),
    )

    evidence_check = result.get(
        "evidence_check",
        {},
    )

    citation_check = result.get(
        "citation_check",
        {},
    )

    claim_check = result.get(
        "claim_check",
        [],
    )

    answer_valid = bool(
        result.get(
            "answer_valid",
            False,
        )
    )

    evidence_sufficient = bool(
        evidence_check.get(
            "sufficient",
            False,
        )
    )

    citation_valid = bool(
        citation_check.get(
            "is_valid",
            False,
        )
    )

    is_refusal = bool(
        citation_check.get(
            "is_refusal",
            False,
        )
    )

    cited_ids = [
        str(value)
        for value in citation_check.get(
            "cited_ids",
            [],
        )
    ]


    # ========================================================
    # ANSWER
    # ========================================================

    st.divider()

    st.subheader("Answer")

    if answer_valid:

        st.success(answer)

    elif is_refusal:

        st.warning(answer)

    else:

        st.error(answer)


    # ========================================================
    # STATUS METRICS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Evidence",
            (
                "Sufficient"
                if evidence_sufficient
                else "Insufficient"
            ),
        )

    with col2:

        st.metric(
            "Citations",
            (
                "Valid"
                if citation_valid
                else "Invalid"
            ),
        )

    with col3:

        st.metric(
            "Answer",
            (
                "Validated"
                if answer_valid
                else "Refused"
            ),
        )

    with col4:

        st.metric(
            "Latency",
            f"{latency_ms / 1000:.1f}s",
        )


    # ========================================================
    # CITED EVIDENCE
    # ========================================================

    st.divider()

    st.subheader("Evidence supporting the answer")

    if not isinstance(
        evidence,
        pd.DataFrame,
    ):
        evidence = pd.DataFrame(evidence)


    if evidence.empty:

        st.info(
            "No evidence was retrieved."
        )

    else:

        evidence = evidence.copy()

        if "chunk_id" in evidence.columns:

            evidence["__cited"] = (
                evidence["chunk_id"]
                .astype(str)
                .isin(cited_ids)
            )

            evidence = (
                evidence
                .sort_values(
                    "__cited",
                    ascending=False,
                )
                .drop(
                    columns=["__cited"]
                )
                .reset_index(drop=True)
            )


        cited_evidence = []

        additional_evidence = []

        for _, row in evidence.iterrows():

            row_data = row.to_dict()

            chunk_id = str(
                row_data.get(
                    "chunk_id",
                    "",
                )
            )

            if chunk_id in cited_ids:

                cited_evidence.append(
                    row_data
                )

            else:

                additional_evidence.append(
                    row_data
                )


        # ----------------------------------------------------
        # CITED CHUNKS
        # ----------------------------------------------------

        if cited_evidence:

            for row in cited_evidence:

                chunk_id = str(
                    row.get(
                        "chunk_id",
                        "UNKNOWN",
                    )
                )

                section = str(
                    row.get(
                        "section",
                        "",
                    )
                )

                condition = str(
                    row.get(
                        "condition",
                        "",
                    )
                )

                recommendation = str(
                    row.get(
                        "recommendation_text",
                        "",
                    )
                )

                st.markdown(
                    f'<div class="evidence-title">'
                    f'[{chunk_id}] {section}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                if condition:

                    st.markdown(
                        f'<div class="evidence-meta">'
                        f'Condition: {condition}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                st.write(
                    recommendation
                )

                st.divider()

        else:

            if is_refusal:

                st.info(
                    "No citation was required because "
                    "the system refused the answer."
                )

            else:

                st.warning(
                    "No cited evidence was identified."
                )


        # ----------------------------------------------------
        # ADDITIONAL RETRIEVED EVIDENCE
        # ----------------------------------------------------

        if additional_evidence:

            with st.expander(
                f"Additional retrieved evidence "
                f"({len(additional_evidence)})"
            ):

                for row in additional_evidence:

                    chunk_id = str(
                        row.get(
                            "chunk_id",
                            "UNKNOWN",
                        )
                    )

                    section = str(
                        row.get(
                            "section",
                            "",
                        )
                    )

                    condition = str(
                        row.get(
                            "condition",
                            "",
                        )
                    )

                    recommendation = str(
                        row.get(
                            "recommendation_text",
                            "",
                        )
                    )

                    st.markdown(
                        f"**[{chunk_id}] {section}**"
                    )

                    if condition:

                        st.caption(
                            f"Condition: {condition}"
                        )

                    st.write(
                        recommendation
                    )

                    st.divider()


    # ========================================================
    # VALIDATION DETAILS
    # ========================================================

    st.divider()

    with st.expander(
        "Citation validation"
    ):

        st.json(
            citation_check
        )


    with st.expander(
        "Evidence sufficiency"
    ):

        st.json(
            evidence_check
        )


    with st.expander(
        "Claim-level NLI validation"
    ):

        if claim_check:

            claim_rows = []

            for item in claim_check:

                claim_rows.append(
                    {
                        "Status": item.get(
                            "status",
                            "",
                        ),
                        "Supported": item.get(
                            "supported",
                            False,
                        ),
                        "Entailment score": item.get(
                            "best_entailment_score",
                            None,
                        ),
                        "Citation IDs": ", ".join(
                            map(
                                str,
                                item.get(
                                    "cited_ids",
                                    [],
                                ),
                            )
                        ),
                    }
                )

            st.dataframe(
                pd.DataFrame(claim_rows),
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.info(
                "No claims were generated."
            )


    # ========================================================
    # QUERY DETAILS
    # ========================================================

    with st.expander(
        "Query details"
    ):

        st.write(
            "**Original question**"
        )

        st.write(
            result.get(
                "query",
                query,
            )
        )

        st.write(
            "**Normalised question**"
        )

        st.write(
            result.get(
                "normalized_query",
                "",
            )
        )

        if cited_ids:

            st.write(
                "**Cited evidence IDs**"
            )

            st.code(
                "\n".join(cited_ids)
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "NICE Guide AI • Production-oriented Hybrid RAG • "
    "Synthetic development corpus • Streamlit"
)
