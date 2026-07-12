"""OpenRouter client factory, system prompt, and the tool-use agent loop.

OpenRouter exposes an OpenAI-compatible chat-completions API with function calling.
The conversation in `messages` is kept in OpenAI message format (the system
prompt is prepended at call time, not stored).
"""

from __future__ import annotations

import json

from openai import OpenAI

import aigct_tools

# MODEL = "meta-llama/llama-3.1-8b-instruct"
MODEL = "google/gemini-2.5-flash-lite"
MAX_TOKENS = 512  # tool-call turn; tool JSON is short
MAX_TOKENS_SUMMARY = 1024  # summary turn; system prompt says 1-3 sentences

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
- Results contain ROC AUC and negative log10 mann-whitney u p-value for each VEP.
- Results are ranked by ROC AUC descending; a higher AUC means better \
discrimination of pathogenic from benign variants.
- If you cannot confidently map the disease to a TASK_CODE, or a tool returns no \
results (e.g. an unrecognized gene), ask the user to clarify rather than guessing.
- The full results table is rendered to the user separately, so do NOT repeat the \
whole table in your reply. Give a brief (1-3 sentence) summary naming the top few \
VEPs and their ROC AUCs and negative log10 mann-whitney u p-values, \
and note the task/gene you used.
- When user mentions only a disease without mentioning a gene provide the results
for the overall task, not for any specific gene.
- IMPORTANT: when calling a tool, arguments MUST be valid JSON enclosed in curly \
braces. Never use any other format for tool arguments.
- If the user asks for a disease area not in the list above, say you do not \
have data for that disease and ask them to choose one of the supported \
diseases. Suggest that you can query the clinvar task because it covers \
many different disease areas. Use the exact phrase, do not have data, somewhere in your response. \
Do NOT include any benchmark results in your reply.
"""


_NO_DATA_PHRASES = (
    "don't have data",
    "do not have data",
    "no data",
    "not in the list",
    "not supported",
    "not one of the supported",
    "not available for",
    "cannot find data",
    "no benchmark",
)


def _is_declining(text: str) -> bool:
    """Return True if the summary text is refusing/declining the request."""
    t = text.lower()
    return any(phrase in t for phrase in _NO_DATA_PHRASES)


def make_client(api_key: str) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        timeout=60.0,
    )


def _user_facing_messages(messages: list) -> list:
    """Strip raw tool-call and tool-result messages from history.

    Used when sending the first API call of a new turn so the model cannot
    answer from cached tool data — it must call a tool again.
    """
    out = []
    for msg in messages:
        role = msg.get("role")
        if role == "tool":
            continue
        if role == "assistant" and msg.get("tool_calls"):
            continue
        # Skip synthetic "Tool result for …" user messages from the
        # content-parsing fallback path.
        if role == "user" and msg.get("content", "").startswith("Tool result for "):
            continue
        out.append(msg)
    return out


def _parse_content_tool_calls(content: str | None) -> list[tuple[str, dict]] | None:
    """Try to parse JSON tool calls embedded in message content.

    Returns a list of (function_name, args_dict) tuples, or None if the content
    doesn't look like a tool-call payload.
    """
    if not content:
        return None
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return None
    calls = []
    for item in data:
        if isinstance(item, dict) and "name" in item:
            args = item.get("parameters") or item.get("arguments") or {}
            calls.append((item["name"], args))
    return calls or None


def run_turn(client: OpenAI, messages: list, query_mgr):
    """Run one user turn through the tool-use loop.

    Mutates `messages` (OpenAI format, no system message) in place with the
    assistant and tool turns. Returns (assistant_text, tables) where `tables` is
    a list of (title, DataFrame) produced during the turn.
    """
    tables = []
    has_tool_results = False
    max_iterations = 6
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        history = messages if has_tool_results else _user_facing_messages(messages)
        kwargs = dict(
            model=MODEL,
            max_tokens=MAX_TOKENS_SUMMARY if has_tool_results else MAX_TOKENS,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
        )
        if not has_tool_results:
            kwargs["tools"] = aigct_tools.TOOL_SCHEMAS
            kwargs["tool_choice"] = "required"
        response = client.chat.completions.create(**kwargs)

        msg = response.choices[0].message

        if not msg.tool_calls:
            # Some models return tool calls as JSON text in content instead of
            # using the structured tool_calls field.  Try to parse and dispatch.
            parsed_calls = _parse_content_tool_calls(msg.content)
            if parsed_calls and not has_tool_results:
                messages.append({"role": "assistant", "content": msg.content or ""})
                for name, args in parsed_calls:
                    title, df, result_text = aigct_tools.dispatch(name, args, query_mgr)
                    if df is not None:
                        tables.append((title, df))
                    messages.append(
                        {
                            "role": "user",
                            "content": f"Tool result for {name}: {result_text}",
                        }
                    )
                has_tool_results = True
                continue

            text = msg.content or ""
            messages.append({"role": "assistant", "content": text})
            return text, [] if _is_declining(text) else tables

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

    raise RuntimeError("Tool-use loop exceeded maximum iterations without a final answer.")
