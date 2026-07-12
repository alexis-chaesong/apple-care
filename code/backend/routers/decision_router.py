# GET /api/decision/{id}/explain — Track4 설명가능성

"""
routers/decision_router.py
============================
Track4(설명가능성): tb_decision_audit에 이미 기록된 판정 행 하나를 조회해서
사람이 읽을 수 있는 설명으로 변환해 돌려주는 라우터.

이 라우터는 판단/숫자 계산을 전혀 하지 않는다 - services.decision_audit.
get_audit_log_by_id()로 이미 저장된 행을 그대로 조회하고, services.llm_service.
explain_decision()에 그 값을 그대로 넘겨 설명 문장만 생성한다 (숫자 지어내기
방지 원칙은 llm_service.explain_decision() 쪽에서 프롬프트로 강제함).
"""

import logging

from fastapi import APIRouter, HTTPException

from services.decision_audit import get_audit_log_by_id
from services.llm_service import (
    explain_decision as llm_explain_decision,
    LLMTimeoutError, LLMParseError, LLMAuthError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


@router.get("/decision/{audit_id}/explain", summary="판정 근거 설명 (Track4 설명가능성)")
async def explain_decision(audit_id: int):
    record = get_audit_log_by_id(audit_id)
    if record is None:
        raise HTTPException(
            status_code=404, detail=f"audit_id={audit_id}에 해당하는 판정 기록이 없습니다."
        )

    try:
        explanation = await llm_explain_decision(record)
    except (LLMTimeoutError, LLMParseError, LLMAuthError) as exc:
        logger.error("판정 설명 생성 실패 (audit_id=%d)", audit_id, exc_info=True)
        raise HTTPException(
            status_code=502, detail="설명 생성에 실패했습니다 (LLM 응답 문제)."
        ) from exc

    return {"result": "SUCCESS", "audit_id": audit_id, "explanation": explanation, "raw": record}
