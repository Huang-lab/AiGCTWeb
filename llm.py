"""Groq client factory, system prompt, and the tool-use agent loop.

Groq exposes an OpenAI-compatible chat-completions API with function calling.
The conversation in `messages` is kept in OpenAI message format (the system
prompt is prepended at call time, not stored).
"""

from __future__ import annotations

import json
import re
import uuid

from groq import BadRequestError, Groq

import aigct_tools

# Matches legacy function-call formats that some Groq model generations produce
# instead of proper JSON tool calls, e.g.:
#   <function=name({"arg": "val"})></function>
#   <function=name[]{"arg": "val"}</function>
# Strategy: capture the function name, then find the first {...} in the tag.
_LEGACY_FN_RE = re.compile(r"<function=(\w+)[^{]*(\{.+?\})[^<]*</function>", re.DOTALL)


def _parse_legacy_tool_call(failed_generation: str) -> dict | None:
    """Extract a tool call from Groq's legacy function-call format, or None."""
    m = _LEGACY_FN_RE.search(failed_generation)
    if not m:
        return None
    name, args_str = m.group(1), m.group(2)
    try:
        json.loads(args_str)  # validate JSON before using it
    except json.JSONDecodeError:
        return None
    return {"id": f"call_{uuid.uuid4().hex[:8]}", "name": name, "arguments": args_str}


MODEL = "llama-3.1-8b-instant"
MAX_TOKENS = 512          # tool-call turn; tool JSON is short
MAX_TOKENS_SUMMARY = 256  # summary turn; system prompt says 1-3 sentences

SYSTEM_PROMPT = """\
You are an assistant that answers questions about the benchmark performance of \
variant effect predictors (VEPs, also called "score sources") at predicting the \
pathogenicity of genetic variants. You do not know the performance numbers \
yourself — you must call a tool to retrieve them, then summarize the result.

Benchmark tasks (TASK_CODE = meaning):
- CANCER = Cancer
- ADRD = Alzheimer's disease and related dementias
- CLINVAR = ClinVar (general clinical pathogenicity)
- ASD = Autism spectrum disorder
- CHD = Congenital heart disease
- DDD = Developmental disorders (Deciphering Developmental Disorders)

Rules:
- Map the disease or condition the user mentions to one of the TASK_CODEs above.
- If the user names a specific gene (e.g. PTEN, BRCA1), call \
get_top_veps_for_task_gene with that gene as an uppercased official HGNC symbol.
- Otherwise call get_top_veps_for_task for overall task performance.
- Results are ranked by ROC AUC descending; a higher AUC means better \
discrimination of pathogenic from benign variants.
- If you cannot confidently map the disease to a TASK_CODE, or a tool returns no \
results (e.g. an unrecognized gene), ask the user to clarify rather than guessing.
- The full results table is rendered to the user separately, so do NOT repeat the \
whole table in your reply. Give a brief (1-3 sentence) summary naming the top few \
VEPs and their AUCs, and note the task/gene you used.
- IMPORTANT: when calling a tool, arguments MUST be valid JSON enclosed in curly \
braces. Never use any other format for tool arguments.
"""


def make_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


def run_turn(client: Groq, messages: list, query_mgr):
    """Run one user turn through the tool-use loop.

    Mutates `messages` (OpenAI format, no system message) in place with the
    assistant and tool turns. Returns (assistant_text, tables) where `tables` is
    a list of (title, DataFrame) produced during the turn.
    """
    tables = []
    has_tool_results = False

    while True:
        try:
            kwargs = dict(
                model=MODEL,
                max_tokens=MAX_TOKENS_SUMMARY if has_tool_results else MAX_TOKENS,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            )
            if not has_tool_results:
                kwargs["tools"] = aigct_tools.TOOL_SCHEMAS
                kwargs["tool_choice"] = "auto"
            response = client.chat.completions.create(**kwargs)
        except BadRequestError as exc:
            # Groq rejects its own model output when the model emits the legacy
            # <function=name(args)></function> format instead of JSON tool calls.
            # Parse the failed generation and synthesise a proper tool call so
            # the loop can continue normally.
            body = exc.body or {}
            err = body.get("error", {}) if isinstance(body, dict) else {}
            if err.get("code") == "tool_use_failed":
                legacy = _parse_legacy_tool_call(err.get("failed_generation", ""))
                if legacy:
                    fake_tc = {
                        "id": legacy["id"],
                        "type": "function",
                        "function": {
                            "name": legacy["name"],
                            "arguments": legacy["arguments"],
                        },
                    }
                    messages.append(
                        {"role": "assistant", "content": "", "tool_calls": [fake_tc]}
                    )
                    for tc_dict in [fake_tc]:
                        try:
                            args = json.loads(tc_dict["function"]["arguments"])
                        except json.JSONDecodeError:
                            args = {}
                        title, df, result_text = aigct_tools.dispatch(
                            tc_dict["function"]["name"], args, query_mgr
                        )
                        if df is not None:
                            tables.append((title, df))
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc_dict["id"],
                                "content": result_text,
                            }
                        )
                    has_tool_results = True
                    continue  # re-enter loop so the model can summarise
            raise  # re-raise if we can't handle it

        msg = response.choices[0].message

        if not msg.tool_calls:
            text = msg.content or ""
            messages.append({"role": "assistant", "content": text})
            return text, tables

        # Append the assistant turn carrying the tool calls (serializable form).
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }
        )

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            title, df, result_text = aigct_tools.dispatch(
                tc.function.name, args, query_mgr
            )
            if df is not None:
                tables.append((title, df))
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_text,
                }
            )
        has_tool_results = True
