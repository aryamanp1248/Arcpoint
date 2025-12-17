
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict
from app.services.context_engine import ContextEngine
router = APIRouter()
context_engine = ContextEngine()

# =============================
# Request schema
# =============================
class OperatorQuery(BaseModel):
    user_id: str
    query: str

# =============================
# Unified Context API
# =============================
@router.post("/v1/context/query")
def query_context(payload: OperatorQuery) -> Dict[str, str]:
    """
    Unified entrypoint for:
    - routing decisions
    - quality investigations
    - traffic forecasts
    - fleet / model questions
    The ContextEngine is the source of truth.
    This API layer is intentionally thin.
    """
    answer = context_engine.answer(payload.query)
    return {
        "user_id": payload.user_id,
        "query": payload.query,
        "answer": answer,
    }
