"""Streamlit chat app: ask natural-language questions about VEP performance.

Claude routes each question to one of two aigct query methods and the app
renders the returned table, ranked by AUC.
"""

from __future__ import annotations

import streamlit as st
from aigct.container import VEBenchmarkContainer

import llm

st.set_page_config(
    page_title="AIGCT — VEP Performance Chat",
    page_icon="🧬",
    layout="centered",
)

CONFIG_PATH = "aigct.yaml"

EXAMPLES = [
    {
        "label": "Cancer — overall",
        "prompt": "Which variant effect predictors perform best for cancer pathogenicity?",
    },
    {
        "label": "Alzheimer's disease",
        "prompt": "Top performing VEPs for predicting pathogenicity for Alzheimer's disease.",
    },
    {
        "label": "Gene-specific (PTEN / Cancer)",
        "prompt": "Top performing VEPs for predicting pathogenicity of variants in the PTEN gene for cancer.",
    },
    {
        "label": "Congenital heart disease",
        "prompt": "Which VEPs are best at predicting pathogenicity for congenital heart disease?",
    },
]

_CSS = """
<style>
/* ── Header ─────────────────────────────────────────────────────── */
.aigct-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.25rem;
}
.aigct-header h1 {
    font-size: 1.75rem;
    font-weight: 700;
    color: #0F172A;
    margin: 0;
    line-height: 1.2;
}
.aigct-badge {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #1D4ED8;
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 999px;
    padding: 2px 10px;
    white-space: nowrap;
}
.aigct-subtitle {
    color: #64748B;
    font-size: 0.92rem;
    margin-bottom: 1.5rem;
}

/* ── Example cards ───────────────────────────────────────────────── */
.example-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
    margin-top: 2rem;
}
div[data-testid="stVerticalBlock"] .example-card-btn button {
    border: 1.5px solid #E2E8F0;
    border-radius: 10px;
    background: #F8FAFC;
    color: #1E293B;
    font-size: 0.85rem;
    text-align: left;
    padding: 0.75rem 1rem;
    transition: border-color 0.15s, background 0.15s;
    height: auto;
    white-space: normal;
}
div[data-testid="stVerticalBlock"] .example-card-btn button:hover {
    border-color: #1D4ED8;
    background: #EFF6FF;
}

/* ── Sidebar tweaks ──────────────────────────────────────────────── */
[data-testid="stSidebarContent"] {
    padding-top: 1.5rem;
}
.sidebar-section-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94A3B8;
    margin-bottom: 0.4rem;
}

/* ── Table caption ───────────────────────────────────────────────── */
.table-title {
    font-size: 0.82rem;
    font-weight: 600;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin: 1rem 0 0.3rem;
}
</style>
"""


@st.cache_resource
def get_query_mgr():
    """Build the aigct container once per process (SQLite-backed, reused)."""
    container = VEBenchmarkContainer(CONFIG_PATH)
    return container.query_mgr


@st.cache_resource
def get_client():
    return llm.make_client(st.secrets["OPENROUTER_API_KEY"])


def _auc_column_config(df):
    """Return column_config for st.dataframe, adding a progress bar for ROC AUC."""
    cfg = {}
    if "ROC AUC" in df.columns:
        cfg["ROC AUC"] = st.column_config.ProgressColumn(
            "ROC AUC",
            help="Area under the ROC curve (0.5 = random, 1.0 = perfect)",
            min_value=0.5,
            max_value=1.0,
            format="%.4f",
        )
    return cfg


def render_table(title: str, df):
    st.markdown(f'<p class="table-title">{title}</p>', unsafe_allow_html=True)
    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config=_auc_column_config(df),
    )


def render_history():
    for entry in st.session_state.history:
        with st.chat_message(entry["role"]):
            if entry.get("text"):
                st.markdown(entry["text"])
            for title, df in entry.get("tables", []):
                render_table(title, df)


def render_welcome():
    st.markdown(
        "<p style='color:#64748B;font-size:0.95rem;margin-bottom:0.5rem'>"
        "Try one of these example questions, or type your own below:"
        "</p>",
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        with cols[i % 2]:
            with st.container():
                st.markdown(f"<div class='example-card-btn'>", unsafe_allow_html=True)
                if st.button(
                    f"**{ex['label']}**\n\n{ex['prompt']}",
                    key=f"ex_{i}",
                    width="stretch",
                ):
                    st.session_state.pending_prompt = ex["prompt"]
                st.markdown("</div>", unsafe_allow_html=True)


def main():
    st.markdown(_CSS, unsafe_allow_html=True)

    # Header
    st.markdown(
        '<div class="aigct-header">'
        "<h1>🧬 AIGCT</h1>"
        '<span class="aigct-badge">VEP Performance Chat</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="aigct-subtitle">'
        "Ask in plain English which variant effect predictors perform best "
        "for a disease area or gene."
        "</p>",
        unsafe_allow_html=True,
    )

    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history" not in st.session_state:
        st.session_state.history = []
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None

    # Sidebar
    with st.sidebar:
        st.markdown(
            '<p class="sidebar-section-label">About</p>', unsafe_allow_html=True
        )
        with st.expander("What is AIGCT Chat?", expanded=False):
            st.markdown(
                """
**AIGCT Chat** is a front end to summary data available in AIGCT, a platform
for systematically evaluating ML/AI models of variant effects across the
spectrum of genomics-based precision medicine.

It lets you ask in natural English about the performance of publicly available
variant effect predictors (VEPs) for various disease areas and genes.

**Disease areas covered:** Cancer, Alzheimer's & related dementias, ClinVar,
Autism spectrum disorder, Congenital heart disease, Developmental disorders.

More detailed data and benchmarking of your own VEPs are available directly
in the AIGCT platform — see the
[documentation](https://aigct.readthedocs.io/en/latest/) and
[GitHub](https://github.com/Huang-lab/AiGCT).
                """
            )

        st.divider()
        st.markdown(
            '<p class="sidebar-section-label">Quick examples</p>',
            unsafe_allow_html=True,
        )
        for ex in EXAMPLES:
            if st.button(ex["label"], width="stretch", key=f"sb_{ex['label']}"):
                st.session_state.pending_prompt = ex["prompt"]

        st.divider()
        if st.button("Clear conversation", width="stretch"):
            st.session_state.messages = []
            st.session_state.history = []
            st.rerun()

    # Main content
    if st.session_state.history:
        render_history()
    else:
        render_welcome()

    # Chat input
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
            err = f"Something went wrong: {exc}"
            st.error(err)
            st.session_state.history.append(
                {"role": "assistant", "text": err, "tables": []}
            )
            return

        if text:
            st.markdown(text)
        for title, df in tables:
            render_table(title, df)

    st.session_state.history.append(
        {"role": "assistant", "text": text, "tables": tables}
    )


if __name__ == "__main__":
    main()
