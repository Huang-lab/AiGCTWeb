# aigctweb — Claude Code Instructions

## Project overview

Streamlit chat app where users ask natural-language questions about variant effect
predictor (VEP) benchmark performance. An LLM (OpenRouter) answers by tool-calling the
`aigct` package's query methods and rendering a ranked ROC-AUC table.

## Key files

```
app.py              — Streamlit UI + tool-use loop wiring
llm.py              — OpenRouter client, system prompt, run_turn() agent loop
aigct_tools.py      — query functions, tool schemas, result serialization
aigct.yaml          — aigct config pointing to db/aigct.db
db/aigct.db         — bundled ~6 MB SQLite benchmark (committed)
vendor/             — vendored aigct wheel (committed; installed via requirements.txt)
requirements.txt
.streamlit/
  config.toml
  secrets.toml      — local only, gitignored — OPENROUTER_API_KEY goes here
```

## Architecture

```
Browser -> app.py (Streamlit) -> llm.py (OpenRouter tool-use loop)
                                        |
                          aigct_tools.py dispatch()
                                        |
                          aigct VEBenchmarkContainer -> db/aigct.db
```

- LLM: OpenRouter free tier, model `llama-3.3-70b-versatile`, OpenAI-compatible function calling
- Secret: `OPENROUTER_API_KEY` (from `.streamlit/secrets.toml` locally; Streamlit Cloud Secrets in prod)
- Tool calling is in-process (no MCP server)
- The `aigct` package is NOT on PyPI — it is vendored as a `.whl` in `vendor/` and installed
  via `requirements.txt` by relative path

## aigct query API

The `query_mgr` (from `VEBenchmarkContainer`) exposes:

- `get_score_source_roc_auc_by_task(task_code)` → DataFrame cols **lowercase**:
  `score_source, auc, num_positive, num_negative`
- `get_score_source_gene_roc_auc_by_task_gene(task_code, gene_symbol)` → cols **lowercase**:
  `score_source, gene_symbol, auc, num_positive, num_negative`

Columns are LOWERCASE. Display labels (VEP, ROC AUC, etc.) are applied in `aigct_tools.py`.

## TASK_CODE vocabulary

| Code | Meaning |
|------|---------|
| CANCER | Cancer |
| ADRD | Alzheimer's disease and related dementias |
| CLINVAR | ClinVar (general clinical pathogenicity) |
| ASD | Autism spectrum disorder |
| CHD | Congenital heart disease |
| DDD | Developmental disorders (Deciphering Developmental Disorders) |

## Development environment

- Python 3.10, venv: `.venv_aigweb` (uv-managed)
- Install: `VIRTUAL_ENV=.venv_aigweb uv pip install -r requirements.txt`
- Run: `.venv_aigweb/bin/python -m streamlit run app.py`
- Do NOT use `pip` directly — use `uv pip` with `VIRTUAL_ENV` set

## aigct container initialization

`VEBenchmarkContainer(config_file)` requires `aigct.yaml` to include:
- `db.url` (sqlite path to `db/aigct.db`)
- `repository.root_dir` (any string; unused by our query methods)
- `log.dir`, `output_dir`
- Full `plot` block (`roc_pr_line`, `mwu_bar`, `calibration_line`) — plotter reads at init

## Deployment (Streamlit Community Cloud)

- `db/aigct.db` and `vendor/*.whl` must be committed — needed at build/runtime
- In app Secrets UI set: `OPENROUTER_API_KEY = "gsk_..."`
- `requirements.txt` installs the vendored wheel by relative path

## Do not

- Add the aigct wheel to PyPI or try to install it from git — use the vendored wheel
- Commit `.streamlit/secrets.toml` — it is gitignored
- Use `pip` directly in this venv — use `uv pip`
