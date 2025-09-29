# utils/validation.py
from __future__ import annotations
from typing import Any, Dict, Optional, Callable, List
from pydantic import BaseModel, ValidationError
from langsmith import traceable

def _coerce_for_model(state: Dict[str, Any]) -> Dict[str, Any]:
    # Coerce the state into an object compatible with our Input/Output models
    return {
        "session_id": state.get("session_id", ""),
        "messages": state.get("messages", []) or [],
        "metadata": state.get("metadata", {}) or {},
    }

def validate_node(
    name: str,
    tags: Optional[List[str]] = None,
    input_model: Optional[type[BaseModel]] = None,
    output_model: Optional[type[BaseModel]] = None,
) -> Callable[[Callable[..., Dict[str, Any]]], Callable[..., Dict[str, Any]]]:
    """
    Decorator that:
      - accepts LangGraph's extra kwargs (e.g., config)
      - optionally validates state against Pydantic models
      - traces with LangSmith
    """
    tags = tags or []

    def decorator(fn: Callable[..., Dict[str, Any]]) -> Callable[..., Dict[str, Any]]:
        @traceable(name=name, tags=tags)
        def wrapper(state: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
            # LangGraph may pass 'config', 'abort', etc. in **kwargs; we just ignore them
            if input_model is not None:
                try:
                    input_model(**_coerce_for_model(state))
                except ValidationError as ve:
                    # surface a readable error and short-circuit this node
                    state.setdefault("_error", {})
                    state["_error"] = {
                        "node": name,
                        "where": "input_validation",
                        "errors": [e.model_dump() for e in ve.errors()] if hasattr(ve, "errors") else str(ve),
                    }
                    return state

            out = fn(state, *args, **kwargs)  # <-- forward any kwargs (e.g., config)

            if output_model is not None:
                try:
                    output_model(**_coerce_for_model(out))
                except ValidationError as ve:
                    out.setdefault("_error", {})
                    out["_error"] = {
                        "node": name,
                        "where": "output_validation",
                        "errors": [e.model_dump() for e in ve.errors()] if hasattr(ve, "errors") else str(ve),
                    }
            return out

        return wrapper
    return decorator