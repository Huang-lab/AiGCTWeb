"""FastAPI chat app: ask natural-language questions about VEP performance.

The LLM routes each question to one of the aigct query methods via tool calling
and the app renders the returned table ranked by AUC.
"""

from __future__ import annotations

import asyncio
import html
import logging
import uuid
from typing import Annotated

import httpx
import markdown as md
from fastapi import Cookie, FastAPI, Form, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from aigct.container import VEBenchmarkContainer

import llm

CONFIG_PATH = "aigct.yaml"
OLLAMA_HEALTH_URL = "http://localhost:11434/api/tags"

logger = logging.getLogger(__name__)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Shared resources — initialised once at startup.
_client = None
_query_mgr = None
_ollama_ready = False

# Per-session message history: session_id -> list of OpenAI-format messages.
_sessions: dict[str, list] = {}

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


@app.on_event("startup")
async def startup() -> None:
    global _client, _query_mgr, _ollama_ready

    # Check Ollama — non-fatal so the app starts in dev without it.
    try:
        async with httpx.AsyncClient() as c:
            await c.get(OLLAMA_HEALTH_URL, timeout=3.0)
        _ollama_ready = True
    except Exception:
        logger.warning(
            "Ollama not reachable at %s — chat will return an error until "
            "Ollama is started.",
            OLLAMA_HEALTH_URL,
        )

    _client = llm.make_client()
    container = VEBenchmarkContainer(CONFIG_PATH)
    _query_mgr = container.query_mgr

    # Pre-warm: load the model into Ollama so the first user query isn't slow.
    if _ollama_ready:
        try:
            await run_in_threadpool(
                lambda: _client.chat.completions.create(
                    model=llm.MODEL,
                    max_tokens=1,
                    messages=[{"role": "user", "content": "hi"}],
                    extra_body={"options": {"num_ctx": llm.NUM_CTX}},
                )
            )
            logger.info("Ollama model pre-warmed.")
        except Exception as e:
            logger.warning("Pre-warm failed: %s", e)


def _get_or_create_session(session_id: str | None) -> tuple[str, list]:
    if not session_id or session_id not in _sessions:
        session_id = uuid.uuid4().hex
        _sessions[session_id] = []
    return session_id, _sessions[session_id]


@app.get("/", response_class=HTMLResponse)
async def index(
    request: Request,
    response: Response,
    session_id: Annotated[str | None, Cookie()] = None,
) -> HTMLResponse:
    session_id, _ = _get_or_create_session(session_id)
    resp = templates.TemplateResponse(
        request,
        "index.html",
        {"examples": EXAMPLES, "session_id": session_id},
    )
    resp.set_cookie("session_id", session_id, max_age=86400, httponly=True)
    return resp


@app.post("/chat", response_class=HTMLResponse)
async def chat(
    request: Request,
    response: Response,
    message: Annotated[str, Form()],
    session_id: Annotated[str | None, Cookie()] = None,
) -> HTMLResponse:
    session_id, messages = _get_or_create_session(session_id)

    if not _ollama_ready:
        # Re-check in case Ollama started after the app did.
        try:
            async with httpx.AsyncClient() as c:
                await c.get(OLLAMA_HEALTH_URL, timeout=2.0)
            globals()["_ollama_ready"] = True
        except Exception:
            pass

    messages.append({"role": "user", "content": message})

    if not _ollama_ready:
        text = (
            "**Ollama is not running.** "
            "Start Ollama (`ollama serve`) and ensure `qwen2.5:7b` is pulled, "
            "then try again."
        )
        tables = []
    else:
        try:
            text, tables = await run_in_threadpool(
                llm.run_turn, _client, messages, _query_mgr
            )
        except Exception as exc:
            text = f"**Error:** {exc}"
            tables = []

    # Build the HTML fragment: user bubble + assistant bubble.
    user_html = html.escape(message)

    assistant_body = md.markdown(text) if text else ""
    for title, df in tables:
        assistant_body += f'<p class="table-title">{html.escape(title)}</p>'
        assistant_body += df.to_html(
            index=False,
            classes="result-table",
            float_format=lambda x: f"{x:.4f}",
            border=0,
        )

    fragment = f"""
<div class="msg msg-user">
  <div class="bubble">{user_html}</div>
</div>
<div class="msg msg-assistant">
  <div class="bubble">
    {assistant_body}
  </div>
</div>
"""
    resp = HTMLResponse(fragment)
    resp.set_cookie("session_id", session_id, max_age=86400, httponly=True)
    return resp


@app.post("/clear", response_class=HTMLResponse)
async def clear(
    session_id: Annotated[str | None, Cookie()] = None,
) -> HTMLResponse:
    if session_id and session_id in _sessions:
        _sessions[session_id] = []
    return HTMLResponse("")
