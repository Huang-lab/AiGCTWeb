# AIGCT — VEP Performance Chat

A Streamlit chat app where you ask natural-language questions about the benchmark
performance of **variant effect predictors (VEPs)**, and an LLM answers by
calling the AIGCT package's query methods and rendering a ranked results table.
AIGCT is a platform for systematically evaluating ML/AI models of variant effects across the spectrum of genomics-based precision medicine. It consists of python based API and a database of variant effect data organized into categories based on the source of the data. AIGCT code and documentation can be found here:
https://github.com/Huang-lab/AiGCT

This app only supports querying summary data. To access the complete set of available data use the AIGCT
platform directly.

The LLM (served free via [Groq](https://groq.com)) doesn't know the numbers — it
translates each question into a call to one of two `aigct` query methods, then
summarizes the returned table (ranked by ROC AUC, descending).

**Example questions:**
- *"Top performing VEPs for predicting pathogenicity of variants in the PTEN gene for cancer."* → per-gene method
- *"Top performing VEPs for predicting pathogenicity for Alzheimer's disease."* → per-task method (ADRD)

## How it works

```
Browser ──► app.py (Streamlit) ──► LLM via Groq (tool-use loop)
                       │                    │
                       │            picks a tool + args
                       ▼                    ▼
            aigct VEBenchmarkContainer ──► query_mgr ──► db/aigct.db (SQLite)
```

One Streamlit process holds the Groq client and the aigct container in memory
(`@st.cache_resource`). Each question runs a tool-use loop: the model picks
`get_top_veps_for_task` or `get_top_veps_for_task_gene`, the app runs it against
the bundled SQLite benchmark, returns a compact table to the model, and the model
writes a short summary. The app renders the full sorted table with `st.dataframe`.

## Benchmark tasks

| TASK_CODE | Meaning |
|-----------|---------|
| `CANCER`  | Cancer |
| `ADRD`    | Alzheimer's disease and related dementias |
| `CLINVAR` | ClinVar (general clinical pathogenicity) |
| `ASD`     | Autism spectrum disorder |
| `CHD`     | Congenital heart disease |
| `DDD`     | Developmental disorders (Deciphering Developmental Disorders) |

## Project layout

```
aigctweb/
├── app.py                  # Streamlit UI + tool-use loop wiring
├── llm.py                  # Groq client, system prompt, run_turn() agent loop
├── aigct_tools.py          # 2 query fns + tool schemas + result serialization
├── aigct.yaml              # aigct config → db/aigct.db
├── db/aigct.db             # bundled ~6 MB SQLite benchmark (committed)
├── vendor/                 # vendored aigct wheel (committed; installed via requirements.txt)
├── requirements.txt
└── .streamlit/
    ├── config.toml
    └── secrets.toml        # local only, gitignored — your GROQ_API_KEY
```

## Getting a Groq API key (free)

The app calls a Llama model hosted on Groq's free tier, so you need a Groq API
key (no credit card required for the free tier):

1. Go to <https://console.groq.com> and sign in (Google/GitHub or email).
2. Open **API Keys** → **Create API Key**, give it a name.
3. Copy the key (`gsk_...`) immediately — it's shown only once.

The free tier has generous rate limits, easily enough for interactive use. The
default model is `llama-3.3-70b-versatile`; if Groq retires it, pick a current
one from <https://console.groq.com/docs/models> and update `MODEL` in `llm.py`.

> Not to be confused with **Grok** (xAI's chatbot). **Groq** is a separate AI
> inference company; the two are unrelated.

## Run locally

Dependencies are already installed in `.venv_aigweb` (a `uv`-managed venv). To
recreate elsewhere: `uv pip install -r requirements.txt`.

```bash
# 1. Add your key
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
#    then edit secrets.toml and paste your GROQ_API_KEY

# 2. Start the app
.venv_aigweb/bin/python -m streamlit run app.py
```

The app opens at <http://localhost:8501>.

## Deploy (Streamlit Community Cloud)

1. Push this repo to GitHub (the `db/aigct.db` and `vendor/*.whl` files must be
   committed — they're needed at build/runtime).
2. On <https://share.streamlit.io>, **New app** → point at this repo and
   `app.py`.
3. In the app's **Settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "gsk_..."
   ```
4. Deploy. The build installs `requirements.txt`, including the vendored aigct
   wheel by relative path (no PyPI / git auth needed). aigct's own dependencies
   (pandas, sqlalchemy, scikit-learn, matplotlib, …) resolve from PyPI, so the
   first build is slow; subsequent cold starts are cheap since the SQLite db is
   local and small.
