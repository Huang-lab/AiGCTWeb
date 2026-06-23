# Plan: AIGCT Chat Web App (FastAPI + Ollama + Oracle Cloud)

## Goal
A chat web app where researchers ask natural-language questions about variant
effect predictor (VEP) performance. An LLM answers by tool-calling the `aigct`
package's `query_mgr` methods and rendering a ranked AUC table.

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

### Compute
Oracle Cloud Free Tier — **Ampere A1 ARM** instance: 4 OCPUs + 24 GB RAM.
Enough to run a 7B quantized model on CPU (~5 GB) alongside the app.

### Stack
```
Browser (N users)
    |  HTTP + Server-Sent Events (streaming)
    v
FastAPI  (Python, Uvicorn)
  +-- per-session chat history (in-memory dict, keyed by session cookie)
  +-- aigct_tools.py          <- unchanged query/tool layer
  +-- VEBenchmarkContainer    <- loaded once at startup (read-only, shared)
  +-- Ollama client (httpx to localhost:11434)
        |
        v
Ollama  (localhost:11434, systemd service)
  +-- model: qwen2.5:7b  (best tool-calling at this size, ~5 GB RAM)
        |
        v
db/aigct.db  (6.2 MB SQLite, read-only)
```

### Request flow
1. User message → FastAPI handler appends to session history.
2. POST to Ollama `/v1/chat/completions` with tool schemas (OpenAI format).
3. Ollama returns prose or a `tool_use` block (function + args).
4. App dispatches to matching `aigct_tools.py` fn → `query_mgr` → SQLite → DataFrame.
5. DataFrame sorted by `auc` desc; compact markdown representation returned as `tool` message.
6. Ollama writes a short NL summary.
7. Response streamed back to browser via SSE; table rendered client-side from JSON payload.

### Why these choices

| Concern | Choice | Reason |
|---|---|---|
| Multi-user | FastAPI | Async; handles concurrent requests. Streamlit re-runs per user per message — not suited for multi-user. |
| Streaming | Server-Sent Events | Simple, unidirectional, no WebSocket overhead. |
| Frontend | htmx + Jinja2 | Zero build step; SSE support built-in; no JS framework needed. |
| Local LLM | Ollama | One-command ARM install; OpenAI-compatible API (drop-in for Groq client). |
| Model | qwen2.5:7b | Best tool/function-calling of small models; ~5 GB RAM leaving 19 GB for OS + app. |
| Session state | In-memory dict | Sufficient for low traffic; can swap to SQLite later if persistence needed. |

### Concurrency note
Ollama processes one generation at a time on CPU. At ~10–20 tok/s on 4 ARM cores,
a queue of 2–3 simultaneous users is fine for a research tool with modest traffic.

## Process management (systemd)

Ollama installs its own systemd service automatically:
```bash
curl -fsSL https://ollama.com/install.sh | sh
systemctl enable ollama
systemctl start ollama
ollama pull qwen2.5:7b
```

FastAPI app service (`/etc/systemd/system/aigctweb.service`):
```ini
[Unit]
Description=aigctweb FastAPI app
After=network.target ollama.service
Requires=ollama.service

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/aigctweb
ExecStart=/home/ubuntu/aigctweb/.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

App startup health-checks Ollama before accepting requests:
```python
@app.on_event("startup")
async def wait_for_ollama():
    import httpx, asyncio
    for _ in range(30):
        try:
            async with httpx.AsyncClient() as c:
                await c.get("http://localhost:11434/api/tags", timeout=2)
            return
        except Exception:
            await asyncio.sleep(1)
    raise RuntimeError("Ollama not ready after 30s")
```

## Project structure
```
aigctweb/
+-- app.py                  # FastAPI app: routes, SSE, session management
+-- llm.py                  # Ollama client (openai SDK, base_url=localhost:11434), system prompt, run_turn()
+-- aigct_tools.py          # 2 tool fns + OpenAI tool schemas + DataFrame formatting  [reuse as-is]
+-- aigct.yaml              # aigct config pointing at db/aigct.db
+-- templates/
|   +-- index.html          # Jinja2 chat UI with htmx + SSE
+-- db/
|   +-- aigct.db            # bundled 6.2 MB SQLite repo (committed)
+-- vendor/
|   +-- aigct-1.0.1-py3-none-any.whl
+-- requirements.txt
+-- aigctweb.service        # systemd unit file (reference copy)
+-- README.md
```

## Implementation steps

### 1. Dependencies (`requirements.txt`)
```
fastapi
uvicorn[standard]
httpx
openai              # used as Ollama client (base_url override)
jinja2
python-multipart    # for form parsing
pandas
tabulate
./vendor/aigct-1.0.1-py3-none-any.whl
```

### 2. aigct config (`aigct.yaml`)
Unchanged from current build — relative `db.url: sqlite:///db/aigct.db`,
full plot block, log/output under `/tmp/aigct`.

### 3. Tool layer (`aigct_tools.py`)
Reuse as-is. `TOOL_SCHEMAS` is already in OpenAI function-calling format
(written for Groq, which is the same format Ollama expects).

### 4. LLM agent loop (`llm.py`)
- `make_client()` → `openai.OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")`.
- `MODEL = "qwen2.5:7b"` — module constant.
- `SYSTEM_PROMPT` — unchanged role + TASK_CODE mapping.
- `run_turn(client, messages, query_mgr)` — same tool-use loop as Groq version
  (OpenAI message format, already implemented).

### 5. FastAPI app (`app.py`)
- Load `VEBenchmarkContainer` once at startup (module-level, not per-request).
- Session store: `dict[session_id, list[messages]]` in module scope.
- `GET /` → render `templates/index.html`.
- `POST /chat` → append user message, call `run_turn()`, return JSON
  `{text, table_markdown}`.
- SSE endpoint `GET /chat/stream` for streaming responses (optional — can start
  without streaming and add later).
- Session cookie set on first visit; 24h TTL; prune stale sessions on each request.

### 6. Frontend (`templates/index.html`)
- htmx `hx-post="/chat"` on form submit.
- Response swapped into chat history div.
- Table rendered as `<pre>` or a simple HTML table from the markdown payload.
- Minimal CSS — no framework required.

### 7. Deploy (Oracle Cloud)
1. Provision Ampere A1 free instance (Ubuntu 22.04).
2. Install Ollama, pull `qwen2.5:7b`.
3. Clone repo, create venv, `VIRTUAL_ENV=.venv uv pip install -r requirements.txt`.
4. Install `aigctweb.service`, `systemctl enable --now aigctweb`.
5. Open port 8000 in OCI Security List (or put nginx in front on port 80/443).
6. Optional: Certbot for HTTPS.

## Build status (2026-06-22 — FastAPI/Ollama rewrite)
All files written and dependencies installed in `.venv_aigweb`.

- [x] `aigct_tools.py` — verified, reused as-is
- [x] `aigct.yaml` — verified, reused as-is
- [x] `db/aigct.db` — committed, 6.2 MB
- [x] `vendor/aigct-1.0.1-py3-none-any.whl` — committed
- [x] `llm.py` — Ollama client (`openai` SDK, `base_url=http://localhost:11434/v1`), `MODEL=qwen2.5:7b`
- [x] `app.py` — FastAPI: startup Ollama health-check, in-memory sessions, GET `/`, POST `/chat`, POST `/clear`
- [x] `templates/index.html` — htmx chat UI: sidebar examples, user/assistant bubbles, result tables
- [x] `requirements.txt` — fastapi, uvicorn[standard], openai, jinja2, python-multipart, markdown; streamlit/groq removed
- [x] `aigctweb.service` — systemd unit with `Requires=ollama.service`
- [ ] End-to-end test on OCI instance (needs Ollama + `qwen2.5:7b` running)

## Decisions (locked)
1. **LLM backend**: **Ollama** (local, free), model `qwen2.5:7b`. No external API key needed.
2. **Web framework**: **FastAPI** + htmx/Jinja2 for multi-user support.
3. **Hosting**: **Oracle Cloud Free Tier** Ampere A1 (4 OCPUs, 24 GB RAM).
4. **Process management**: **systemd** — `ollama.service` (auto-installed) +
   `aigctweb.service` with `Requires=ollama.service`.
5. **aigct install**: vendor the wheel at `vendor/aigct-1.0.1-py3-none-any.whl`,
   reference by relative path in `requirements.txt`.
