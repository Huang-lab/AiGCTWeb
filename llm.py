"""OpenRouter client factory, system prompt, and the tool-use agent loop.

OpenRouter exposes an OpenAI-compatible chat-completions API with function calling.
The conversation in `messages` is kept in OpenAI message format (the system
prompt is prepended at call time, not stored).
"""

from __future__ import annotations

import json

from openai import OpenAI

import aigct_tools

MODEL = "meta-llama/llama-3.1-8b-instruct"
MAX_TOKENS = 512  # tool-call turn; tool JSON is short
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


def make_client(api_key: str) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )


def run_turn(client: OpenAI, messages: list, query_mgr):
    """Run one user turn through the tool-use loop.

    Mutates `messages` (OpenAI format, no system message) in place with the
    assistant and tool turns. Returns (assistant_text, tables) where `tables` is
    a list of (title, DataFrame) produced during the turn.
    """
    tables = []
    has_tool_results = False

    while True:
        kwargs = dict(
            model=MODEL,
            max_tokens=MAX_TOKENS_SUMMARY if has_tool_results else MAX_TOKENS,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        )
        if not has_tool_results:
            kwargs["tools"] = aigct_tools.TOOL_SCHEMAS
            kwargs["tool_choice"] = "auto"
        response = client.chat.completions.create(**kwargs)

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
