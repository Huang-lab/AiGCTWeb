"""Tool layer over the aigct query_mgr.

Two pure functions ((query_mgr, args) -> DataFrame) plus the Anthropic tool
schemas and a serializer for tool results. Written so they could be lifted into
a standalone MCP server later with no change to the query logic.

The aigct query methods return lowercase columns:
  - get_score_source_roc_auc_by_task(task_code)
        -> score_source, auc, num_positive, num_negative
  - get_score_source_gene_roc_auc_by_task_gene(task_code, gene_symbol)
        -> score_source, gene_symbol, auc, num_positive, num_negative
"""

from __future__ import annotations

import pandas as pd

# TASK_CODE vocabulary present in the bundled SQLite db (db/aigct.db).
TASK_CODES = {
    "CANCER": "Cancer",
    "ADRD": "Alzheimer's disease and related dementias",
    "CLINVAR": "ClinVar (general clinical pathogenicity)",
    "ASD": "Autism spectrum disorder",
    "CHD": "Congenital heart disease",
    "DDD": "Developmental disorders (Deciphering Developmental Disorders)",
}

# Friendlier display labels for the lowercase db columns.
_COLUMN_LABELS = {
    "score_source": "VEP",
    "gene_symbol": "Gene",
    "auc": "ROC AUC",
    "num_positive": "Num Positive",
    "num_negative": "Num Negative",
}


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by AUC descending and apply display labels."""
    if df is None or df.empty:
        return df
    df = df.sort_values("auc", ascending=False).reset_index(drop=True)
    return df.rename(columns=_COLUMN_LABELS)


def get_top_veps_for_task(query_mgr, task_code: str) -> pd.DataFrame:
    """Top VEPs for a task, ranked by AUC descending."""
    df = query_mgr.get_score_source_roc_auc_by_task(task_code)
    return _finalize(df)


def get_top_veps_for_task_gene(
    query_mgr, task_code: str, gene_symbol: str
) -> pd.DataFrame:
    """Top VEPs for a specific gene within a task, ranked by AUC descending."""
    df = query_mgr.get_score_source_gene_roc_auc_by_task_gene(task_code, gene_symbol)
    return _finalize(df)


def _vep_list_finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase, sort by VEP name and apply labels."""
    if df is None or df.empty:
        return df
    df = df.rename(columns={c: c.lower() for c in df.columns})
    df["score_source"] = df["code"]
    if "score_source" in df.columns:
        df = df.sort_values("score_source").reset_index(drop=True)
    cols = [c for c in ["score_source", "description"] if c in df.columns]
    return df[cols].rename(columns=_COLUMN_LABELS)


def get_all_variant_effect_source(query_mgr) -> pd.DataFrame:
    """Return all variant effect sources (VEPs) available in the DB."""
    try:
        if hasattr(query_mgr, "get_all_variant_effect_source"):
            df = query_mgr.get_all_variant_effect_source()
        elif hasattr(query_mgr, "get_variant_effect_sources"):
            # fallback to the generic method present in older/newer query mgrs
            df = query_mgr.get_variant_effect_sources(None)
        else:
            raise AttributeError(
                "query_mgr does not implement get_all_variant_effect_source or get_variant_effect_sources"
            )
    except FileNotFoundError:
        # Under some development setups the packaged CSVs are missing; return
        # no-results so the tool system can respond gracefully instead of
        # crashing the app.
        return None
    return _vep_list_finalize(df)


def get_variant_effect_source_by_task(query_mgr, task_code: str) -> pd.DataFrame:
    """Return all variant effect sources (VEPs) for a specific task."""
    try:
        if hasattr(query_mgr, "get_variant_effect_source_by_task"):
            df = query_mgr.get_variant_effect_source_by_task(task_code)
        elif hasattr(query_mgr, "get_variant_effect_sources"):
            df = query_mgr.get_variant_effect_sources(task_code)
        else:
            raise AttributeError(
                "query_mgr does not implement get_variant_effect_source_by_task or get_variant_effect_sources"
            )
    except FileNotFoundError:
        return None
    return _vep_list_finalize(df)


# --- Tool schemas (OpenAI / Groq function-calling format) ------------------

_TASK_CODE_ENUM = list(TASK_CODES.keys())
_TASK_CODE_DESC = "; ".join(f"{code} = {name}" for code, name in TASK_CODES.items())

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_top_veps_for_task",
            "description": (
                "Return the ranked performance (ROC AUC) of all variant effect "
                "predictors (VEPs / score sources) for a benchmark task, across "
                "all genes in that task. The returned `auc` column contains ROC AUC "
                "values. Use this when the user asks about overall performance "
                "for a disease or task and does NOT name a specific gene."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_code": {
                        "type": "string",
                        "enum": _TASK_CODE_ENUM,
                        "description": f"The benchmark task code. {_TASK_CODE_DESC}.",
                    }
                },
                "required": ["task_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_veps_for_task_gene",
            "description": (
                "Return the ranked performance (ROC AUC) of all variant effect "
                "predictors (VEPs / score sources) for a specific gene within a "
                "benchmark task. The returned `auc` column contains ROC AUC "
                "values. Use this when the user names a specific gene "
                "(e.g. PTEN, BRCA1)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_code": {
                        "type": "string",
                        "enum": _TASK_CODE_ENUM,
                        "description": f"The benchmark task code. {_TASK_CODE_DESC}.",
                    },
                    "gene_symbol": {
                        "type": "string",
                        "description": (
                            "Official HGNC gene symbol, uppercased "
                            "(e.g. PTEN, BRCA1, TP53)."
                        ),
                    },
                },
                "required": ["task_code", "gene_symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_all_variant_effect_source",
            "description": (
                "Return all variant effect sources (VEPs) available in the database. "
                "This only returns a name and description of the vep. It does not "
                "return performance metrics."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_variant_effect_source_by_task",
            "description": (
                "Return all variant effect sources (VEPs) for a given benchmark task. "
                "This only returns a name and description of the vep. It does not return "
                "performance metrics."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_code": {
                        "type": "string",
                        "enum": _TASK_CODE_ENUM,
                        "description": f"The benchmark task code. {_TASK_CODE_DESC}.",
                    }
                },
                "required": ["task_code"],
            },
        },
    },
]


def dispatch(name: str, tool_input: dict, query_mgr):
    """Execute a tool by name.

    Returns (title, dataframe, result_text). `dataframe` is None when no rows
    were returned (e.g. unknown gene), in which case result_text explains it.
    """
    if name == "get_top_veps_for_task":
        task_code = tool_input["task_code"]
        df = get_top_veps_for_task(query_mgr, task_code)
        title = f"Top VEPs — {TASK_CODES.get(task_code, task_code)}"
    elif name == "get_top_veps_for_task_gene":
        task_code = tool_input["task_code"]
        gene = tool_input["gene_symbol"]
        df = get_top_veps_for_task_gene(query_mgr, task_code, gene)
        title = f"Top VEPs — {gene} in {TASK_CODES.get(task_code, task_code)}"
    elif name == "get_all_variant_effect_source":
        df = get_all_variant_effect_source(query_mgr)
        title = "Variant Effect Sources — All"
    elif name == "get_variant_effect_source_by_task":
        task_code = tool_input["task_code"]
        df = get_variant_effect_source_by_task(query_mgr, task_code)
        title = f"Variant Effect Sources — {TASK_CODES.get(task_code, task_code)}"
    else:
        return None, None, f"Error: unknown tool '{name}'."
    if df is None or df.empty:
        return (
            None,
            None,
            (
                "No results found for those inputs. The gene may not be in this "
                "task's benchmark, or the task/gene combination is unrecognized."
            ),
        )
    return title, df, df_to_tool_result(df)


def df_to_tool_result(df: pd.DataFrame, max_rows: int = 50) -> str:
    """Compact, token-bounded serialization of a result DataFrame for the model."""
    shown = df.head(max_rows)
    md = shown.to_markdown(index=False, floatfmt=".4f")
    note = ""
    if len(df) > max_rows:
        note = f"\n\n(Showing top {max_rows} of {len(df)} rows, ranked by AUC.)"
    return md + note
