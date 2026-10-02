"""Tools Claude can call: document search and arithmetic."""

import ast
import operator

from store import embed, get_collection

MAX_TOP_K = 10

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def search_documents(query: str, source: str | None = None, top_k: int = 5) -> str:
    """Retrieve the chunks most similar to query, optionally limited to one file."""
    top_k = max(1, min(top_k, MAX_TOP_K))
    where = {"source": source} if source else None

    results = get_collection().query(
        query_embeddings=embed([query]), n_results=top_k, where=where
    )
    documents = results["documents"][0]
    if not documents:
        return "No matching passages found."

    return "\n\n".join(
        f"[{meta['source']} chunk {meta['chunk']}]\n{text}"
        for text, meta in zip(documents, results["metadatas"][0])
    )


def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression without exposing eval() to the model."""
    try:
        return str(_eval_node(ast.parse(expression, mode="eval").body))
    except Exception:
        return f"Error: could not evaluate {expression!r} as arithmetic."


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("unsupported expression")


TOOL_SCHEMAS = [
    {
        "name": "search_documents",
        "description": (
            "Search the indexed documents for passages relevant to a query. "
            "Call this multiple times with different queries to gather all the "
            "facts a question needs. Use the source filter to restrict the "
            "search to a single file when you already know where to look."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for, in natural language.",
                },
                "source": {
                    "type": "string",
                    "description": "Optional exact filename to restrict the search to.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "How many passages to return (1-10, default 5).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculate",
        "description": (
            "Evaluate an arithmetic expression. Use this for any growth rate, "
            "difference, ratio, or percentage rather than doing the arithmetic "
            "yourself. Supports + - * / ** and parentheses."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "An arithmetic expression, e.g. (94425-91361)/91361*100",
                }
            },
            "required": ["expression"],
        },
    },
]

IMPLEMENTATIONS = {
    "search_documents": search_documents,
    "calculate": calculate,
}


def run_tool(name: str, payload: dict) -> tuple[str, bool]:
    """Execute a tool. Returns (result_text, is_error)."""
    function = IMPLEMENTATIONS.get(name)
    if function is None:
        return f"Error: unknown tool {name!r}.", True
    try:
        return function(**payload), False
    except Exception as exc:
        return f"Error running {name}: {exc}", True
