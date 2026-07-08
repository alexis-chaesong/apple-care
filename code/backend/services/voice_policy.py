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
from typing import Optional

from services import llm_service
from services.llm_service import LLMAuthError, LLMParseError, LLMTimeoutError
from services.bayesian_policy import upsert_llm_policy
from stt_tts import stt_service, tts_service
from stt_tts.mic_lock import mic_lock
from state.hitl_state_machine import hitl_state_machine
from vision_bridge import vision_bridge_manager
from robot_bridge import bridge_manager

logger = logging.getLogger(__name__)

# 로봇 제어 명령은 안전이 걸려있어(특히 비상정지) LLM 분류에 맡기지 않고 키워드로
# 결정론적으로 판단함 - LLM은 애매한 표현을 잘못 해석할 위험이 있으므로 이런 이진
# 안전 명령에는 부적합. EMERGENCY_STOP 키워드를 MANUAL_PAUSE보다 먼저 검사해야 함
# ("비상정지"/"긴급정지"에도 "정지"가 들어있어서, 순서가 바뀌면 비상정지가 그냥
# 일시정지로 잘못 분류됨).
_EMERGENCY_STOP_KEYWORDS = ("비상정지", "긴급정지", "긴급 정지", "이머전시")
_START_KEYWORDS = ("시작해", "시작하자", "가동해", "가동시작", "로봇 시작", "동작 시작")
_PAUSE_KEYWORDS = ("정지해", "멈춰", "일시정지", "일시 정지", "잠깐 멈춰", "스톱")


def _detect_robot_command(text: str) -> Optional[str]:
    """
    STT 결과가 정책 명령이 아니라 로봇 제어 명령(START/MANUAL_PAUSE/EMERGENCY_STOP)인지
    키워드로 먼저 판별. 매칭되면 그 문자열을, 아니면 None을 반환해서 호출부가
    정책 명령(translate_policy_command) 경로로 계속 진행하게 함.
    """
    if any(kw in text for kw in _EMERGENCY_STOP_KEYWORDS):
        return "EMERGENCY_STOP"
    if any(kw in text for kw in _START_KEYWORDS):
        return "START"
    if any(kw in text for kw in _PAUSE_KEYWORDS):
        return "MANUAL_PAUSE"
    return None


_COMMAND_CONFIRM_MESSAGE = {
    "START": "로봇을 시작합니다.",
    "MANUAL_PAUSE": "로봇을 일시정지합니다.",
    "EMERGENCY_STOP": "비상정지합니다.",
}


async def run_voice_policy_command(listen_timeout_sec: float = 8.0) -> dict:
    """
    마이크로 정책 명령 1건을 듣고, 해석/저장까지 마친 뒤 결과를 반환.

    반환 result 필드:
      SUCCESS         - 정책 명령 정상 적용됨 (applied_policies 채워짐)
      ROBOT_COMMAND   - 정책이 아니라 로봇 제어 명령(START/MANUAL_PAUSE/EMERGENCY_STOP)이 발행됨
      NO_SPEECH       - 마이크에서 아무 말도 못 들음 (무음 또는 STT 실패)
      LLM_ERROR       - STT는 됐지만 LLM 해석 실패(타임아웃/인증/파싱 에러)
      NO_POLICY       - LLM은 응답했지만 정책을 하나도 추출 못함(명령이 아닌 잡담 등)
    """
    async with mic_lock:
        text = await stt_service.listen_once(timeout_sec=listen_timeout_sec)
        if text is None:
            await tts_service.speak("다시 말씀해 주세요.")
            return {"result": "NO_SPEECH"}

        robot_command = _detect_robot_command(text)
        if robot_command is not None:
            # 정책 명령(translate_policy_command) 경로를 아예 타지 않고 여기서
            # 바로 끝냄 - LLM 호출 자체를 생략해서 지연도 줄이고, 안전 명령이
            # LLM의 정책 스키마로 잘못 해석될 여지도 원천 차단함.
            bridge_manager.publish_command(command_type=robot_command)
            logger.info("음성 로봇 제어 명령 발행: %s (raw_text=%s)", robot_command, text)
            await tts_service.speak(_COMMAND_CONFIRM_MESSAGE[robot_command])
            return {"result": "ROBOT_COMMAND", "command": robot_command, "raw_text": text}

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

        await _apply_to_current_apple(applied)

        return {"result": "SUCCESS", "raw_text": text, "applied_policies": applied}


async def _apply_to_current_apple(applied: list[dict]) -> None:
    """
    카메라 앞에 이미 사과가 있고, 그게 마침 "이거 뭔지 몰라서" 사람에게 물어보던
    중(HITL ask_human 대기 세션)이었다면, 방금 내린 음성 명령이 곧 그 질문에 대한
    답이 되도록 세션을 즉시 해소함.

    처음엔 policy["condition"] == session.condition(보통 "unknown")으로 엄격하게
    매칭하려 했으나 실제로는 전혀 안 맞았음 - 사람은 "이거 못난이야, b2로 보내"처럼
    자연스럽게 말하지 "unknown 조건을 이걸로 바꿔"라고 말하지 않기 때문에, LLM이
    뽑아내는 condition은 "ugly"/"normal" 등이 되고 세션의 "unknown"과 글자 그대로
    일치할 수가 없었음 (그래서 정책은 저장되는데 세션은 계속 안 풀리는 증상이 났음).

    카메라 앞에는 항상 사과가 하나뿐인 단일 처리 시스템이므로, 지금 대기 중인
    질문이 있다면 이 시점에 내려진 음성 명령은 곧 "그 질문"에 대한 답이라고 보고
    조건 매칭 없이 첫 번째로 적용된 정책의 destination으로 바로 해소함.

    이게 없으면: 정책은 DB에 저장되지만, 이미 감지되어 ask_human으로 넘어간 그
    사과는 vision_bridge.py의 중복 감지 방지(디바운스) 로직 때문에 다음 폴링에서도
    "이미 처리한 것"으로 취급되어 재평가되지 않음 - 결과적으로 "정책은 들어가는데
    로봇은 그 사과를 안 옮긴다"는 증상으로 나타남 (실제로 겪은 문제).
    """
    session = hitl_state_machine.current_session
    if session is not None and applied:
        destination = applied[0]["destination"]
        logger.info(
            "대기 중인 HITL 세션(fruit=%s condition=%s)을 음성 명령으로 즉시 해소: "
            "destination=%s",
            session.fruit_type, session.condition, destination,
        )
        await hitl_state_machine.force_reset(destination=destination)

    # HITL 세션 유무와 무관하게, 비전 쪽 디바운스 상태도 초기화해둔다.
    # (이미 execute로 자동 처리됐던 사과에 대해 정책을 새로 덮어쓴 경우 등,
    # HITL 세션이 없는 케이스에서도 다음 폴링부터 새 정책이 반영되게 하기 위함)
    vision_bridge_manager.force_redetect()
