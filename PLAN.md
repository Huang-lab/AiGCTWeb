# Plan: AIGCT Chat Web App (Streamlit + Claude)

## Goal
A chat web app where researchers ask natural-language questions about variant
effect predictor (VEP) performance. Claude does **not** know answers — it
translates each question into a call to one of two `aigct` query methods, then
the app renders the returned DataFrame as a table sorted by AUC descending.

Example questions:
- "Top performing VEPs for predicting pathogenicity of variants in the PTEN gene for cancer." → gene method
- "Top performing VEPs for predicting pathogenicity for Alzheimer's disease." → task method (ADRD)

## Verified facts (smoke test passed 2026-06-03)
- **Method 1** — `query_mgr.get_score_source_roc_auc_by_task(task_code)` → cols
  **lowercase** `score_source, auc, num_positive, num_negative` (38 rows for CANCER).
- **Method 2** — `query_mgr.get_score_source_gene_roc_auc_by_task_gene(task_code, gene_symbol)`
  → cols `score_source, gene_symbol, auc, num_positive, num_negative`
  (correctly filters by gene now).
- Columns are **lowercase** (plan originally assumed uppercase — code/UI uses lowercase).
- `TASK_CODE` vocab in DB: `CANCER`, `ADRD` (Alzheimer's), `CLINVAR`,
  `ASD` (autism), `CHD` (congenital heart disease), `DDD`.
- Only the **6.2 MB SQLite DB** is needed (caches are lazy; no Zenodo CSV repo).
  Now copied to `db/aigct.db`.
- Container build requires a config with a **full `plot` block** (plotter reads
  `roc_pr_line`, `mwu_bar`, `calibration_line` at init), `db.url`,
  `repository.root_dir` (any path string), `log.dir`, `output_dir`.
- Wheel now bundles `sqlalchemy`. Source wheel:
  `~/gitrepo/aigct_dev/dist/aigct-1.0.1-py3-none-any.whl`.
- venv is `.venv_aigweb` (uv-managed, no pip → `VIRTUAL_ENV=.venv_aigweb uv pip install ...`).

## Architecture
One Streamlit process holds everything in memory:
```
Browser (Streamlit React client over WebSocket)
        │
        ▼
app.py  (server-side Python, re-runs per message)
  ├─ anthropic.Anthropic client            ← @st.cache_resource
  ├─ VEBenchmarkContainer("aigct.yaml")    ← @st.cache_resource → query_mgr → db/aigct.db (6.2MB)
  └─ tool functions (aigct_tools.py)
        get_top_veps_for_task(task_code)
        get_top_veps_for_task_gene(task_code, gene_symbol)
```

**Request flow (Claude tool-use loop):**
1. Send conversation + tool schemas to Claude.
2. Claude returns prose or a `tool_use` block (method + args).
3. App executes the matching Python fn → `query_mgr` → SQLite → DataFrame.
4. App sorts by `auc` desc, returns a compact representation to Claude as `tool_result`.
5. Claude writes a short NL summary.
6. App renders `st.dataframe(sorted_df)` + `st.markdown(summary)`.

## Project structure
```
aigctweb/
├── app.py                  # Streamlit UI + Claude tool-use loop
├── aigct_tools.py          # 2 tool fns + JSON tool schemas + DataFrame formatting
├── llm.py                  # Anthropic client factory + system prompt + agent loop
├── aigct.yaml              # aigct config pointing at db/aigct.db (relative path)
├── db/
│   └── aigct.db            # bundled 6.2 MB SQLite repo (committed)  ✅ already copied
├── vendor/
│   └── aigct-1.0.1-py3-none-any.whl   # vendored aigct wheel (committed)  ✅ already copied
├── requirements.txt
├── .streamlit/
│   ├── config.toml         # theme/server options
│   └── secrets.toml        # local only (gitignored): ANTHROPIC_API_KEY
└── README.md
```

## Implementation steps

### 1. Dependencies (`requirements.txt`)
```
streamlit
anthropic
pandas
tabulate            # if df.to_markdown() is used for tool_result
./vendor/aigct-1.0.1-py3-none-any.whl   # vendored wheel (LOCKED decision)
```
- aigct isn't on PyPI. **Decision (locked):** vendor the wheel in `vendor/` and
  reference it by relative path in `requirements.txt`. Community Cloud installs
  it from the committed file — no git auth needed. Revisit only if a second
  consumer or a public release appears.

### 2. aigct config (`aigct.yaml`)
- `db.url: "sqlite:///db/aigct.db"` (relative to app working dir).
- `repository.root_dir`: any path (unused by our methods) — point at `db/` or a placeholder.
- `log.dir` / `output_dir`: a writable temp path (e.g. `/tmp/aigct` — hosts have ephemeral disk).
- Full `plot` block copied from the verified sample config.

### 3. Tool layer (`aigct_tools.py`)
- `get_container(config_path)` — build `VEBenchmarkContainer(config_path)` once
  (wrapped by `@st.cache_resource` in app.py).
- `get_top_veps_for_task(query_mgr, task_code) -> pd.DataFrame` — calls method 1,
  returns `[score_source, auc, num_positive, num_negative]` sorted by `auc` desc.
- `get_top_veps_for_task_gene(query_mgr, task_code, gene_symbol) -> pd.DataFrame`
  — calls method 2, same cols **+ `gene_symbol`**, sorted by `auc` desc.
- `TOOL_SCHEMAS` — Anthropic tool-use JSON schema for both fns. Param
  descriptions enumerate valid `task_code` values; `gene_symbol` is an official
  HGNC symbol (e.g. `PTEN`).
- `df_to_tool_result(df)` — compact, token-bounded serialization (e.g.
  `df.head(n).to_markdown()` or records JSON) sent to Claude. Full DataFrame kept
  in Python for display.

### 4. LLM agent loop (`llm.py`)
- `make_client()` → `anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])`.
- `SYSTEM_PROMPT` — role; TASK_CODE↔disease mapping; rule: use the gene method
  **iff** a specific gene is named, else the task method; always present results
  as a table sorted by descending `auc`; if disease/gene unrecognized, ask the
  user to clarify rather than guess.
- `run_turn(client, messages, query_mgr)` — tool-use loop: call Claude → if
  `stop_reason == "tool_use"`, dispatch to the matching fn, append `tool_result`,
  loop; else return final text. Returns both the assistant text and any
  DataFrame, so the UI can render the table.
- Default model: **`claude-opus-4-8`** (LOCKED) — a module constant.

### 5. Streamlit UI (`app.py`)
- `@st.cache_resource` wrappers for the Anthropic client and the aigct container.
- `st.session_state["messages"]` holds chat history (Anthropic message format).
- Render history with `st.chat_message`; input via `st.chat_input`.
- On submit: append user msg → `run_turn(...)` → render returned DataFrame with
  `st.dataframe(df)` and summary with `st.markdown`. Persist a serializable
  record of each table so reruns redraw it.
- Spinner during the round-trip; surface tool/query errors as a chat message
  instead of crashing.

### 6. Config & secrets
- `.streamlit/secrets.toml` (gitignored) for local `ANTHROPIC_API_KEY`; same key
  set in Community Cloud Secrets UI.
- Add `secrets.toml` to `.gitignore`.

### 7. Deploy (Streamlit Community Cloud)
- Push repo to GitHub; "New app" → point at `app.py`; add `ANTHROPIC_API_KEY`
  secret. Build installs `requirements.txt`.
- Document cold-start: app sleeps after idle; cache rebuild on wake is cheap
  (local 6.2 MB SQLite).

## Build status (2026-06-03)
All app files written and compile-clean:
- ✅ `requirements.txt` — streamlit, groq, pandas, sqlalchemy, tabulate,
  `./vendor/aigct-1.0.1-py3-none-any.whl`
- ✅ `aigct.yaml` — relative `db.url: sqlite:///db/aigct.db`; full plot block;
  log/output under `/tmp/aigct`
- ✅ `aigct_tools.py` — 2 query fns, `TOOL_SCHEMAS` (OpenAI/Groq function format),
  `dispatch()`, `df_to_tool_result()`; lowercase db cols renamed to friendly
  labels (VEP, AUC…)
- ✅ `llm.py` — Groq SDK, `MODEL=llama-3.3-70b-versatile`, system prompt
  prepended per call, manual tool-use loop in `run_turn()` (OpenAI message format)
- ✅ `app.py` — Streamlit chat UI, `@st.cache_resource` for client + container,
  reads `st.secrets["GROQ_API_KEY"]`, example sidebar, error-as-chat-message
- ✅ `.streamlit/config.toml`, `.streamlit/secrets.toml.example`, `.gitignore`
  (ignores `.venv_aigweb/` + `.streamlit/secrets.toml`)
- ✅ `README.md` — overview, task table, layout, API-key steps, run + deploy

## Verification
1. ✅ **aigct smoke test, no LLM** — both methods against `db/aigct.db`, lowercase
   columns confirmed.
2. ✅ **Tool functions** — `dispatch()` for task + gene returns sorted-desc AUC,
   correct columns (gene col only on the second), unknown gene → clean
   clarification message, markdown serialization clean. Container builds from the
   relative db path at repo root.
3. ⏳ **End-to-end local** — `streamlit run app.py`; ask both example questions.
   BLOCKED: needs `GROQ_API_KEY` (free, from console.groq.com; no local
   secrets.toml yet). User to run.
4. ⏳ **Error paths** — empty-result path unit-verified; full no-crash UI path
   pending the local run.
5. ⏳ **Deploy check** — push, deploy on Community Cloud, set secret, re-run.

## How to run locally
1. `cp .streamlit/secrets.toml.example .streamlit/secrets.toml` and put your free
   `GROQ_API_KEY` (from <https://console.groq.com>) in it.
2. `.venv_aigweb/bin/python -m streamlit run app.py` (deps already installed in
   `.venv_aigweb`).

## Decisions (locked)
1. **LLM backend**: **Groq** (free tier, OpenAI-compatible function calling),
   model `llama-3.3-70b-versatile`. Switched from Anthropic/Claude on 2026-06-03
   to avoid paid API cost. Secret: `GROQ_API_KEY`.
2. **aigct install for deploy**: vendor the wheel at
   `vendor/aigct-1.0.1-py3-none-any.whl` and reference it by relative path in
   `requirements.txt`. (Locally we use the already-installed `.venv_aigweb`.)

_No open questions — ready to implement._
