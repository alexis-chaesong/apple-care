# OST /api/feedback, GET /api/policy/{fruit_type}

"""
hitl_router.py
===============
사람의 답변(STT로 받은 원문 텍스트, 또는 HMI에서 직접 입력한 텍스트)을
현재 진행 중인 HITL 세션에 그대로 전달만 하는 얇은 창구.

[설계 변경 이력]
state/hitl_state_machine.py가 생기기 전에는 이 라우터가
"LLM 해석 -> Bayesian 학습 반영 -> RESUME 발행"까지 전부 직접 처리했음.
지금은 그 책임이 전부 hitl_state_machine._ask_and_wait()로 이동했으므로,
이 라우터는 해석/저장/재개 로직을 절대 직접 하지 않고
"지금 답변을 기다리는 세션이 있으면 그 세션에 답을 하나 밀어넣는다"는
역할만 함. hitl_state_machine.submit_answer()가 STT 콜백(stt_service)과
완전히 동일하게 취급하는 두 번째 입력 경로(HMI 수동 입력)일 뿐임.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from models import RawFeedbackIn
from state.hitl_state_machine import hitl_state_machine
from services.bayesian_policy import get_policy

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


class FeedbackAck(BaseModel):
    # 이 엔드포인트는 답변 "접수 여부"만 알려줌.
    # 실제 해석 성공/실패, 재질문 여부, RESUME 여부는 전부 hitl_state_machine 내부에서
    # 비동기로 처리되고 WebSocket 브로드캐스트(VLA_DECISION 등)로 별도 전달됨
    accepted: bool
    session_id: Optional[str] = None
    message: str


@router.post("/feedback", response_model=FeedbackAck, summary="사람의 답변을 현재 HITL 세션에 전달")
async def receive_feedback(payload: RawFeedbackIn):
    """
    사람의 원문 답변(STT 또는 HMI 수동 입력)을 현재 진행 중인 HITL 세션에 전달만 함.
    해석(LLM)/학습(Bayesian)/재개(RESUME)는 전부 hitl_state_machine이 담당하므로
    여기서는 절대 직접 호출하지 않음.
    """
    session = hitl_state_machine.current_session
    if session is None:
        logger.warning("HITL 세션이 없는데 피드백이 도착함: raw_answer=%s", payload.raw_answer)
        raise HTTPException(status_code=409, detail="현재 진행 중인 HITL 세션이 없습니다.")

    accepted = hitl_state_machine.submit_answer(payload.raw_answer)

    if not accepted:
        # 세션은 있지만 마침 답변 대기 Future가 열려있지 않은 타이밍
        # (질문 생성/TTS 재생 중이거나, 이미 다른 답변으로 Future가 닫힌 직후 등)
        return FeedbackAck(
            accepted=False,
            session_id=session.session_id,
            message="지금은 답변을 받을 수 있는 시점이 아닙니다. 잠시 후 다시 시도해 주세요.",
        )

    return FeedbackAck(
        accepted=True,
        session_id=session.session_id,
        message="답변이 접수되었습니다.",
    )


@router.get("/policy/{fruit_type}/{condition}", summary="특정 조건의 현재 정책 조회")
async def get_policy_detail(fruit_type: str, condition: str):
    """
    HMI 디버그 화면이나 데모 시연 중 "지금 이 조건의 정책이 뭔지" 바로 확인할 때 사용.
    """
    policy = get_policy(fruit_type, condition)
    if policy is None:
        raise HTTPException(status_code=404, detail="해당 조건에 대한 정책이 아직 없습니다.")
    return {"result": "SUCCESS", "data": policy}
