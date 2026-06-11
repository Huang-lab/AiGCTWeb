"""Streamlit chat app: ask natural-language questions about VEP performance.

Claude routes each question to one of two aigct query methods and the app
renders the returned table, ranked by AUC.
"""

from __future__ import annotations

import streamlit as st
from aigct.container import VEBenchmarkContainer

import llm

st.set_page_config(page_title="AIGCT — VEP Performance Chat", page_icon="🧬")

CONFIG_PATH = "aigct.yaml"

EXAMPLES = [
    "Top performing VEPs for predicting pathogenicity of variants in the PTEN gene for cancer.",
    "Top performing VEPs for predicting pathogenicity for Alzheimer's disease.",
]


@st.cache_resource
def get_query_mgr():
    """Build the aigct container once per process (SQLite-backed, reused)."""
    container = VEBenchmarkContainer(CONFIG_PATH)
    return container.query_mgr


@st.cache_resource
def get_client():
    return llm.make_client(st.secrets["GROQ_API_KEY"])


def render_history():
    for entry in st.session_state.history:
        with st.chat_message(entry["role"]):
            if entry.get("text"):
                st.markdown(entry["text"])
            for title, df in entry.get("tables", []):
                st.caption(title)
                st.dataframe(df, use_container_width=True, hide_index=True)


def main():
    st.title("🧬 AIGCT — VEP Performance Chat")
    st.caption(
        "Query AIGCT platform data to find out which variant effect predictors perform best for a disease or gene. "
    )

    # `messages` = raw LLM conversation; `history` = renderable record.
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history" not in st.session_state:
        st.session_state.history = []
    if "show_about" not in st.session_state:
        st.session_state.show_about = False
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None

    with st.sidebar:
        if st.button("About"):
            st.session_state.show_about = not st.session_state.show_about

        if st.session_state.show_about:
            st.markdown(
                """
                **AIGCT Chat** is a front end to summary data available in
                AIGCT, a platform for systematically evaluating ML/AI
                models of variant effects across the spectrum of genomics-based
                precision medicine.

                It allows users to ask questions in natural English about the
                performance of publicly available variant effect predictors
                (VEPs) in predicting pathogenicity in various disease areas.

                More detailed variant effect prediction data can be obtained
                directly in the AIGCT platform. The AIGCT platform also allows
                users to benchmark the results of their own variant effect
                predictors against the VEP data in the platform.

                Documentation and code for AIGCT are located at https://aigct.readthedocs.io/en/latest/
                and https://github.com/Huang-lab/AiGCT, respectively.
                """
            )
            st.divider()

        st.subheader("Example questions")
        for ex in EXAMPLES:
            if st.button(ex, use_container_width=True):
                st.session_state.pending_prompt = ex
        st.divider()
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.session_state.history = []
            st.rerun()

    render_history()

    prompt = st.chat_input("Ask about VEP performance…")
    if not prompt and st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None
    if not prompt:
        return

    st.session_state.history.append({"role": "user", "text": prompt, "tables": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        try:
            with st.spinner("Querying the benchmark…"):
                client = get_client()
                query_mgr = get_query_mgr()
                text, tables = llm.run_turn(
                    client, st.session_state.messages, query_mgr
                )
        except Exception as exc:
            # surface errors as a chat message, don't crash
            err = f"⚠️ Something went wrong: {exc}"
            st.error(err)
            st.session_state.history.append(
                {"role": "assistant", "text": err, "tables": []}
            )
            return

        if text:
            st.markdown(text)
        for title, df in tables:
            st.caption(title)
            st.dataframe(df, use_container_width=True, hide_index=True)

    st.session_state.history.append(
        {"role": "assistant", "text": text, "tables": tables}
    )


if __name__ == "__main__":
    main()
