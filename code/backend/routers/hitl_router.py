# OST /api/feedback, GET /api/policy/{fruit_type}
# STT로 받은 사람 답변을 POST로 받고, Policy 조회 GET 엔드포인트 제공

"""
hitl_router.py
===============
사람의 답변(STT로 받은 원문 텍스트, 또는 HMI에서 직접 입력한 텍스트)을 받아
LLM으로 해석하고, Policy Memory에 학습 결과를 반영한 뒤 로봇을 재개시키는 창구.

흐름:
    로봇 Hold + TTS 질문
    -> 사람 답변 (STT 또는 HMI 텍스트 입력)
    -> POST /api/feedback  (바로 이 라우터)
    -> llm_service.interpret_human_answer()로 destination 해석
    -> 해석 성공 -> bayesian_policy.record_human_feedback()로 학습 반영
                 -> robot_bridge에 RESUME 명령 발행
    -> 해석 실패(애매한 답변) -> RESUME 하지 않고 재질문 필요 응답 반환
       (실제 재질문 TTS 트리거는 state/hitl_state_machine.py가 담당 예정)

이 라우터는 "해석 + 저장 + 재개 트리거"까지만 책임지고,
"질문을 언제 어떻게 다시 하는지"의 대화 흐름 자체는 hitl_state_machine.py에 위임한다.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from models import RawFeedbackIn
from services import llm_service
from services.llm_service import LLMTimeoutError, LLMParseError, LLMAuthError
from services.bayesian_policy import get_policy, record_human_feedback, upsert_llm_policy
from robot_bridge import bridge_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


class FeedbackResult(BaseModel):
    # 이 라우터의 응답 규격. hitl_state_machine.py가 이 결과를 보고
    # "재개해도 되는지" 혹은 "다시 물어봐야 하는지"를 판단함
    resolved: bool
    destination: Optional[str] = None
    confidence: Optional[float] = None
    message: str


# hitl_router.py의 POST /api/feedback 교체
from state.hitl_state_machine import hitl_state_machine

@router.post("/feedback", summary="HMI에서 수동으로 답변 텍스트 제출")
async def receive_feedback(payload: RawFeedbackIn):
    session = hitl_state_machine.current_session
    if session is None:
        raise HTTPException(status_code=409, detail="현재 대기 중인 질문이 없습니다.")

    submitted = hitl_state_machine.submit_answer(payload.raw_answer)
    if not submitted:
        raise HTTPException(status_code=409, detail="답변 제출 시점이 맞지 않습니다.")

    return {"result": "SUBMITTED", "session_id": session.session_id}

@router.get("/policy/{fruit_type}/{condition}", summary="특정 조건의 현재 정책 조회")
async def get_policy_detail(fruit_type: str, condition: str):
    """
    HMI 디버그 화면이나 데모 시연 중 "지금 이 조건의 정책이 뭔지" 바로 확인할 때 사용.
    """
    policy = get_policy(fruit_type, condition)
    if policy is None:
        raise HTTPException(status_code=404, detail="해당 조건에 대한 정책이 아직 없습니다.")
    return {"result": "SUCCESS", "data": policy}


class PolicyCommandIn(BaseModel):
    raw_text: str


@router.post("/policy/command", summary="작업자 자연어 정책 명령 처리")
async def receive_policy_command(payload: PolicyCommandIn):
    try:
        policies = await llm_service.translate_policy_command(payload.raw_text)
    except (LLMTimeoutError, LLMParseError, LLMAuthError):
        raise HTTPException(status_code=502, detail="LLM 응답을 해석할 수 없습니다 (API 키 확인 필요).")

    if not policies:
        raise HTTPException(status_code=422, detail="명령에서 정책을 추출하지 못했습니다.")

    applied = []
    for p in policies:
        updated = upsert_llm_policy(
            fruit_type=p["fruit_type"], condition=p["condition"], destination=p["destination"],
        )
        applied.append(updated)

    return {"result": "SUCCESS", "raw_text": payload.raw_text, "applied_policies": applied}


class RobotResetRequest(BaseModel):
    destination: Optional[str] = None  # normal_box/processing_box/discard_box/ugly_box 중 하나, 또는 생략


@router.post("/robot/reset", summary="STUCK 상태 관리자 강제 복구")
async def reset_robot(payload: RobotResetRequest):
    result = await hitl_state_machine.force_reset(destination=payload.destination)
    if result["result"] == "NO_ACTIVE_SESSION":
        raise HTTPException(status_code=404, detail="현재 복구할 세션이 없습니다.")
    return result