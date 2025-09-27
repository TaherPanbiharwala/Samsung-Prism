# utils/validation.py
from functools import wraps
from typing import Any, Callable, Type, Optional, Dict
from langsmith import traceable
from pydantic import BaseModel, ValidationError

class NodeValidationError(RuntimeError):
    def __init__(self, node_name: str, where: str, errors: Any, state_snapshot: Dict[str, Any]):
        super().__init__(f"{node_name} {where} validation failed")
        self.node_name = node_name
        self.where = where
        self.errors = errors
        self.state_snapshot = state_snapshot

def pmodel_dump(obj: Any) -> Dict[str, Any]:
    # works for dicts and pydantic models
    if hasattr(obj, "model_dump"):
        return obj.model_dump()  # type: ignore[attr-defined]
    return dict(obj)

def validate_node(
    *,
    name: str,
    input_model: Optional[Type[BaseModel]] = None,
    output_model: Optional[Type[BaseModel]] = None,
    error_key: str = "_error",
    tags: Optional[list[str]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Wrap a node function(state_dict)->state_dict with:
      - optional input validation (pydantic)
      - optional output validation (pydantic)
      - LangSmith nested tracing
      - transform exceptions into structured error in state
    """
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @traceable(name=f"{name}", tags=tags or [])
        @wraps(fn)
        def wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
            # Validate input
            if input_model is not None:
                try:
                    input_model.model_validate(pmodel_dump(state))
                except ValidationError as ve:
                    # attach error details to state and short-circuit
                    state[error_key] = {
                        "node": name,
                        "where": "input",
                        "errors": ve.errors(),
                    }
                    return state

            # Execute node
            try:
                out_state = fn(state)
            except Exception as e:
                # capture runtime error
                state[error_key] = {
                    "node": name,
                    "where": "runtime",
                    "errors": [{"msg": str(e), "type": e.__class__.__name__}],
                }
                return state

            # Validate output
            if output_model is not None:
                try:
                    output_model.model_validate(pmodel_dump(out_state))
                except ValidationError as ve:
                    out_state[error_key] = {
                        "node": name,
                        "where": "output",
                        "errors": ve.errors(),
                    }
                    return out_state

            return out_state
        return wrapper
    return decorator