# AIGCT Web Architecture

## Overview

`aigctweb` is a Streamlit-based chat application that lets users ask natural language questions about benchmark performance of variant effect predictors (VEPs). The app routes each question through a OpenRouter-hosted LLM using function/tool calls to query a local `aigct` benchmark database, then renders both a short assistant summary and a ranked results table.

## Architectural Components

- `Browser` / User Interface
  - Sends user questions via HTTP to the Streamlit app.
  - Displays chat history, assistant summary, and rendered tables.

- `Streamlit App` (`app.py`)
  - Serves the chat UI.
  - Holds cached resources: OpenRouter client and `aigct` query manager.
  - Collects user input and session state.
  - Orchestrates the tool-use loop by calling `llm.run_turn()`.

- `OpenRouter Client` (`llm.py`)
  - Wraps the OpenRouter API client.
  - Maintains the system prompt and model configuration.
  - Sends conversation history and tool schemas to the model.
  - Receives assistant replies and tool call requests.

- `Tool Layer` (`aigct_tools.py`)
  - Defines tool schemas for function calling.
  - Dispatches two query tools:
    - `get_top_veps_for_task(task_code)`
    - `get_top_veps_for_task_gene(task_code, gene_symbol)`
  - Formats query results into pandas DataFrames and Markdown summaries.

- `aigct Benchmark Container` / `query_mgr`
  - Created from `VEBenchmarkContainer(CONFIG_PATH)` in `app.py`.
  - Provides database-backed query methods for benchmark performance.
  - Uses bundled metadata from `aigct.yaml` and `db/aigct.db`.

- `SQLite Benchmark Database` (`db/aigct.db`)
  - Stores the benchmark data used by the `aigct` query manager.
  - Read locally by the `VEBenchmarkContainer`.

## Message and Data Flow

1. User submits a question in the browser.
2. `Streamlit` app appends the question to session-state messages.
3. `app.py` calls `llm.run_turn(client, messages, query_mgr)`.
4. `llm.run_turn()` sends the full conversation to OpenRouter with tool schemas:
   - system prompt
   - user turn history
   - candidate tool definitions
5. OpenRouter returns one of three outcomes:
   - direct text reply (no tool call) → proceed to step 8
   - tool call request for `get_top_veps_for_task`
   - tool call request for `get_top_veps_for_task_gene`
6. If a tool call is returned:
   - `llm.run_turn()` records the assistant tool call in `messages`
   - `llm.run_turn()` dispatches the tool via `aigct_tools.dispatch()`
   - `dispatch()` executes the query against `query_mgr`
   - query result becomes a pandas DataFrame and markdown text
   - `messages` appends a tool response entry (in OpenAI message format)
   - **The loop continues: `llm.run_turn()` sends the updated messages (now including the tool result) back to OpenRouter**
7. OpenRouter processes the tool result and generates a final text summary:
   - OpenRouter reads the markdown result and ranked table
   - OpenRouter returns a final text response (no more tool calls)
   - `llm.run_turn()` records this final response in `messages`
8. `llm.run_turn()` returns the final assistant text and collected tables to `app.py`:
   - `app.py` renders the text summary to the chat
   - `app.py` renders the result DataFrame(s) as Streamlit dataframe widgets

## Object Interaction Diagram

![AIGCT architecture diagram](interaction_diagram.png)

The diagram above is a true PNG image showing the component flow and message exchanges between the browser, Streamlit app, OpenRouter LLM, tool layer, query manager, and SQLite benchmark.

## Component Responsibilities

- `app.py`
  - UI layout and session state management.
  - Caching long-lived resources to avoid rebuilding the OpenRouter client or database container repeatedly.
  - Error handling and final presentation of assistant replies and tables.

- `llm.py`
  - LLM orchestration and tool-call loop.
  - Converts OpenRouter function-calling responses into structured assistant/tool messages.
  - Keeps the responsibilities of the model separate from the query logic.

- `aigct_tools.py`
  - Encodes the domain-specific tool API.
  - Maintains the official `TASK_CODE` vocabulary.
  - Converts raw query results into human-readable outputs.
  - Supports future extraction into a standalone MCP server.

- `db/aigct.db`
  - Provides the only persistent data source in the application.
  - Stored locally and committed for reproducible deployment.

## Deployment Instructions

### Prerequisites

- Python 3.10
- `uv` package manager
- An OpenRouter API key (free tier is sufficient)

### Local development

1. **Create and populate the virtual environment:**

   ```bash
   VIRTUAL_ENV=.venv_aigweb uv pip install -r requirements.txt
   ```

   This installs Streamlit, the OpenAI client, pandas, and the vendored `aigct` wheel
   from `vendor/aigct-1.0.1-py3-none-any.whl`. Do not use `pip` directly.

2. **Set the API key** by creating `.streamlit/secrets.toml` (gitignored):

   ```toml
   OPENROUTER_API_KEY = "sk-or-..."
   ```

3. **Run the app:**

   ```bash
   .venv_aigweb/bin/python -m streamlit run app.py
   ```

   The app is served at `http://localhost:8501`.

### Streamlit Community Cloud

1. **Commit required assets** — the following must be present in the repository:
   - `db/aigct.db` — the bundled SQLite benchmark database (~6 MB)
   - `vendor/aigct-1.0.1-py3-none-any.whl` — the vendored `aigct` wheel
   - `requirements.txt` — references the wheel by relative path (`./vendor/...`)
   - `aigct.yaml` — `aigct` configuration (db path, log dirs, plot block)

2. **Configure secrets** in the Streamlit Cloud app settings (Secrets UI):

   ```toml
   OPENROUTER_API_KEY = "sk-or-..."
   ```

3. **Set the entry point** to `app.py` in the Streamlit Cloud dashboard.

4. Streamlit Cloud will install dependencies from `requirements.txt` at build time,
   including the vendored wheel, and the app will be available at
   `https://<your-app>.streamlit.app/`.

## Key Architectural Patterns

- Tool-oriented LLM interaction: the model is asked to call tools rather than answer directly.
- Cache resource reuse: `@st.cache_resource` keeps the OpenRouter client and query manager in memory.
- Separation of concerns:
  - UI layer in `app.py`
  - LLM orchestration in `llm.py`
  - domain/tool layer in `aigct_tools.py`
  - data persistence in `db/aigct.db`

## Summary

This architecture centers on a Streamlit UI that mediates between user input and an LLM tool-use loop. The LLM decides which benchmark query to execute, the tool layer runs the query against a local SQLite-backed `aigct` container, and the app presents the results as both a concise assistant summary and a ranked DataFrame.
