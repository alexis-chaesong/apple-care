# 음성 정책 명령 처리 (STT → LLM 정책 변환 → Policy Memory 저장)

"""
services/voice_policy.py
==========================
음성으로 받은 정책 명령(자연어 → Policy JSON → tb_policy_memory 저장)을 하나의
함수로 통합. 웨이크워드 트리거(stt_tts/wakeup_listener.py)와 HMI push-to-talk
버튼(routers/hitl_router.py의 POST /api/voice/policy-command) 양쪽에서 이 함수
하나만 호출하면 됨 - 트리거 방식이 늘어나도 이 로직은 바뀌지 않음.

routers/hitl_router.py의 POST /api/policy/command(raw_text를 텍스트로 직접 받는
기존 엔드포인트)와 정책 적용 로직(translate_policy_command -> upsert_llm_policy)은
동일함. 이 함수는 그 앞단에 "마이크로 raw_text를 받는" 단계만 추가한 것.

mic_lock으로 hitl_state_machine의 STT 리스닝과 상호배제되어, 로봇이 Unknown Object를
질문하는 도중에는 이 함수가 마이크를 가로채지 않음(내부에서 자연히 대기함).
"""

import logging

from services import llm_service
from services.llm_service import LLMAuthError, LLMParseError, LLMTimeoutError
from services.bayesian_policy import upsert_llm_policy
from stt_tts import stt_service, tts_service
from stt_tts.mic_lock import mic_lock

logger = logging.getLogger(__name__)


async def run_voice_policy_command(listen_timeout_sec: float = 8.0) -> dict:
    """
    마이크로 정책 명령 1건을 듣고, 해석/저장까지 마친 뒤 결과를 반환.

    반환 result 필드:
      SUCCESS    - 정상 적용됨 (applied_policies 채워짐)
      NO_SPEECH  - 마이크에서 아무 말도 못 들음 (무음 또는 STT 실패)
      LLM_ERROR  - STT는 됐지만 LLM 해석 실패(타임아웃/인증/파싱 에러)
      NO_POLICY  - LLM은 응답했지만 정책을 하나도 추출 못함(명령이 아닌 잡담 등)
    """
    async with mic_lock:
        text = await stt_service.listen_once(timeout_sec=listen_timeout_sec)
        if text is None:
            await tts_service.speak("다시 말씀해 주세요.")
            return {"result": "NO_SPEECH"}

        try:
            policies = await llm_service.translate_policy_command(text)
        except (LLMTimeoutError, LLMParseError, LLMAuthError):
            logger.error("음성 정책 명령 해석 실패: %s", text, exc_info=True)
            await tts_service.speak("명령을 이해하지 못했습니다.")
            return {"result": "LLM_ERROR", "raw_text": text}

        if not policies:
            await tts_service.speak("정책을 추출하지 못했습니다.")
            return {"result": "NO_POLICY", "raw_text": text}

        applied = [
            upsert_llm_policy(
                fruit_type=p["fruit_type"], condition=p["condition"], destination=p["destination"],
            )
            for p in policies
        ]

        logger.info("음성 정책 명령 적용: raw_text=%s applied=%s", text, applied)
        await tts_service.speak(f"{len(applied)}건의 정책을 적용했습니다.")
        return {"result": "SUCCESS", "raw_text": text, "applied_policies": applied}
