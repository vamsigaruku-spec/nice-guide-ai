import os
import time

import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NICE Guide AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# LOAD STREAMLIT SECRET
# ============================================================

try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass


# ============================================================
# BACKEND
# ============================================================

from app.rag import run_validated_rag


# ============================================================
# CUSTOM STYLE
# ============================================================

st.markdown(
    """
    <style>

    .title {
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #9aa3b2;
        margin-top: 0.2rem;
        margin-bottom: 1.2rem;
    }

    .evidence-heading {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .evidence-meta {
        color: #9aa3b2;
        font-size: 0.85rem;
        margin-bottom: 0.7rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">🩺 NICE Guide AI</div>',
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
    "This application uses a synthetic clinical-guidance development corpus. "
    "It is not a medical device, clinical decision-support system, "
    "or substitute for professional clinical judgement."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Configuration")

    top_k = st.slider(
        "Evidence chunks",
        min_value=1,
        max_value=5,
        value=3,
        step=1,
        help="Number of evidence chunks used by the RAG pipeline.",
    )

    entailment_threshold = st.slider(
        "NLI validation threshold",
        min_value=0.50,
        max_value=0.95,
        value=0.70,
        step=0.05,
        help="Threshold used for claim-level grounding validation.",
    )

    st.divider()

    st.subheader("RAG pipeline")

    pipeline_steps = [
        "Semantic retrieval",
        "BM25 retrieval",
        "Reciprocal Rank Fusion",
        "Cross-encoder reranking",
        "Evidence sufficiency",
        "Grounded generation",
        "Citation validation",
        "NLI claim validation",
    ]

    for step in pipeline_steps:
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
# INPUT
# ============================================================

st.subheader("Ask a clinical-guidance question")

query = st.text_area(
    "Question",
    placeholder=(
        "Example: What education should be provided "
        "to adults with asthma?"
    ),
    height=120,
)


# ============================================================
# GENERATE
# ============================================================

generate_button = st.button(
    "Generate grounded answer",
    type="primary",
)


# ============================================================
# RUN PIPELINE
# ============================================================

if generate_button:

    if not query or not query.strip():

        st.error(
            "Please enter a clinical-guidance question."
        )

    else:

        start_time = time.perf_counter()

        with st.spinner(
            "Retrieving evidence, generating answer and validating claims..."
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

                st.session_state["rag_result"] = result
                st.session_state["rag_latency_ms"] = latency_ms

            except Exception as exc:

                st.error(
                    "The RAG pipeline could not complete the request."
                )

                with st.expander("Technical details"):

                    st.exception(exc)

                st.stop()


# ============================================================
# RESULT
# ============================================================

if "rag_result" in st.session_state:

    result = st.session_state["rag_result"]

    latency_ms = st.session_state.get(
        "rag_latency_ms",
        0.0,
    )

    answer = str(
        result.get(
            "answer",
            "",
        )
    ).strip()

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
        str(cid)
        for cid in citation_check.get(
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
    # STATUS
    # ========================================================

    status_1, status_2, status_3, status_4 = st.columns(4)

    with status_1:

        st.metric(
            "Evidence",
            "Sufficient"
            if evidence_sufficient
            else "Insufficient",
        )

    with status_2:

        st.metric(
            "Citations",
            "Valid"
            if citation_valid
            else "Invalid",
        )

    with status_3:

        st.metric(
            "Answer",
            "Validated"
            if answer_valid
            else "Refused",
        )

    with status_4:

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
                cited_evidence.append(row_data)
            else:
                additional_evidence.append(row_data)


        # ----------------------------------------------------
        # CITED EVIDENCE
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
                    f'<div class="evidence-heading">'
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
                    "No citation is shown because the system "
                    "refused the answer due to insufficient evidence."
                )

            else:

                st.info(
                    "No cited evidence was identified."
                )


        # ----------------------------------------------------
        # ADDITIONAL EVIDENCE
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
    # VALIDATION SUMMARY
    # ========================================================

    st.divider()

    st.subheader("Validation summary")

    val_1, val_2, val_3 = st.columns(3)

    with val_1:

        if evidence_sufficient:
            st.success("✓ Evidence sufficient")
        else:
            st.error("✗ Evidence insufficient")

    with val_2:

        if citation_valid:
            st.success("✓ Citation valid")
        else:
            st.error("✗ Citation invalid")

    with val_3:

        if answer_valid:
            st.success("✓ Answer validated")
        elif is_refusal:
            st.warning("⚠ Answer refused")
        else:
            st.error("✗ Answer failed validation")


    # ========================================================
    # CLAIM-LEVEL VALIDATION
    # ========================================================

    if claim_check:

        st.markdown("### Claim-level grounding")

        supported_count = sum(
            bool(
                item.get(
                    "supported",
                    False,
                )
            )
            for item in claim_check
        )

        total_count = len(claim_check)

        st.write(
            f"{supported_count}/{total_count} claims supported"
        )

        claim_rows = []

        for item in claim_check:

            claim_rows.append(
                {
                    "Status": (
                        "Supported"
                        if item.get(
                            "supported",
                            False,
                        )
                        else "Not supported"
                    ),
                    "Entailment score": round(
                        float(
                            item.get(
                                "best_entailment_score",
                                0,
                            )
                        ),
                        3,
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

        st.caption(
            "No generated claims were available for validation."
        )


    # ========================================================
    # QUERY DETAILS
    # ========================================================

    with st.expander("Query details"):

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

            for cid in cited_ids:

                st.code(
                    cid,
                    language=None,
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "NICE Guide AI • Production-oriented Hybrid RAG • "
    "Synthetic development corpus • Streamlit"
)
