# Track 6: VLM Gate (Stage 2, §4.4) — unknown 전용 GPT-4o Vision 물체 식별

"""
vlm_gate.py
=============
condition="unknown"일 때 GPT-4o Vision으로 물체명을 식별하는 Stage 2 모듈.

이 파일이 책임지는 것: "이미지를 GPT-4o에 보내서 물체명을 받아온다"는 I/O
하나뿐이다. destination 결정, posterior 갱신, 게이팅 로직은 전혀 모른다 -
그건 각각 services/decision_planner.py(Stage3 게이트 호출)와
services/bayesian_policy.py(사람 답변 후 record_human_feedback)의 책임이다.

§4.4 원칙: "VLM 응답은 절대 theta_f,c(posterior)를 직접 갱신하지 않는다."
이 파일의 함수들은 순수하게 "무엇을 봤는가"만 반환하고, 어떤 DB도 쓰지 않는다
(tb_decision_audit 기록은 호출부인 decision_planner.py가
services.decision_audit.log_stage2_vlm_call()로 별도 수행).

트리거 단순화 (§4.4의 EVSI_VLM(x_t) > Cost_VLM 원칙 근사, tau_yolo/k1,k2와
같은 급의 "잠정 근사"): 문서에 EVSI_VLM/Cost_VLM 구체 계산식이 없어(다이어그램
레이블뿐), condition == "unknown"이면 무조건 VLM 호출을 시도하는 것으로
단순화했다. 실제 호출 여부의 최종 결정은 이미지 캡처 성공 여부
(get_captured_frame_for_vlm이 None을 반환하면 애초에 시도 자체를 안 함)로
자연히 게이팅된다.
"""

import base64
import logging
from typing import Optional

from openai import AsyncOpenAI, APIError, APITimeoutError

from config import settings

logger = logging.getLogger(__name__)

_client = AsyncOpenAI(api_key=settings.openai_api_key)
# 모듈 레벨 싱글톤 - llm_service.py의 _client와 동일한 패턴(매 호출마다 새로
# 만들지 않고 커넥션 재사용). llm_service._client와 별도 인스턴스로 두는 이유는
# 이 파일이 텍스트 전용 LLM 호출(llm_service.py)과 완전히 독립적인 관심사
# (이미지 인식)를 담당하는 별도 서비스 모듈이기 때문 - 기존 계층 분리 원칙 유지.

# fruit_type 자리에 들어갈 폴백 라벨. VLM 호출 자체를 안 했거나(이미지 없음)
# 호출은 했지만 식별에 실패한 경우 공통으로 사용한다. "unknown"(condition 값)과
# 혼동되지 않도록 별도 문자열로 둔다 - 이렇게 해야 "식별 시도했으나 실패"들끼리는
# 한 정책 버킷으로 학습되면서도, "unknown"이라는 condition 자체와는 구분된다.
UNIDENTIFIED_OBJECT_FRUIT_TYPE = "unidentified_object"


async def get_captured_frame_for_vlm(request_id: str) -> Optional[bytes]:
    """판정 순간의 원본 RGB 프레임을 JPEG 바이트로 가져온다.

    Task1(이미지 캡처 인터페이스 연결)에서 실제 구현으로 교체됨 - vision_ws에
    새로 추가한 온디맨드 서비스 `capture_frame`(SrvCaptureFrame, condition=
    "unknown"일 때만 호출되는 별도 경로, 기존 1Hz get_apple_status 폴링과는
    무관)을 호출한다. vision_bridge.py가 SrvAppleStatus를 다루는 것과 동일한
    패턴(모듈 로드 시 apple_care_msgs try/except로 방어, 클라이언트는
    vision_bridge_manager가 이미 띄워둔 ROS2 노드/executor/스레드를 재사용)을
    그대로 따른다 - 이 함수를 위해 새 ROS2 노드를 따로 띄우지 않는다.

    vision_bridge를 여기서 지연 import(함수 본문 안에서 import)하는 이유:
    이 파일(services/vlm_gate.py)은 원래 "OpenAI만 안다"는 계층 원칙을 갖고
    있었고(decision_planner.py Track6 docstring 참고), 최상단에서 ROS 의존
    모듈을 끌어오면 이 서비스 계층 파일이 ROS 없이는 아예 import조차 안 되는
    상태가 된다. hitl_state_machine.py가 stt_tts를 지연 import하는 것과 동일한
    이유("순환 참조 방지 및 아직 미구현 상태 대비") - vision_bridge.py 자체가
    이미 apple_care_msgs 부재를 우아하게 처리하므로 최악의 경우에도 여기서
    ImportError 없이 아래에서 None을 반환하는 정상 폴백 경로를 탄다.

    §4.4/Track6 설계상 이 함수가 None을 반환하면 호출부(decision_planner.py의
    _decide_unknown_object)가 VLM 호출 자체를 스킵하고 곧장 Stage3(사람에게
    확인)로 넘어간다 - 그래서 아래 모든 실패 케이스(서비스 미준비/타임아웃/
    success=False)를 예외 없이 전부 None으로 통일해서 반환한다.

    Args:
        request_id: 이 프레임을 요청하는 판정 건을 식별하는 값(현재는
            VisionFeatureIn.frame_id를 그대로 전달받는 것을 전제로 함). 실제
            capture_frame 서비스 요청에는 상관관계를 실을 필드가 없어(사용자
            확정 스펙 - 호출 자체가 트리거) 지금은 로그 외 용도로 쓰이지
            않는다 - vision_bridge.request_captured_frame() docstring의 동기화
            근사 설계 결정 참고.

    Returns:
        성공 시 JPEG로 인코딩된 이미지 바이트. 실패/미연결 시 None.
    """
    from vision_bridge import vision_bridge_manager

    logger.debug("이미지 캡처 요청 (request_id=%s)", request_id)
    return await vision_bridge_manager.request_captured_frame(
        timeout_sec=settings.image_capture_timeout_sec
    )


async def call_gpt4o_vlm(image_bytes: bytes) -> dict:
    """GPT-4o Vision에 이미지를 보내 물체명만 식별하도록 요청한다.

    실제 OpenAI API를 호출한다 (mock 아님) - config.settings.openai_api_key를
    llm_service.py와 동일한 방식으로 사용한다.

    프롬프트 설계상 destination(목적지)은 절대 묻지 않는다 - destination은
    항상 사람이 최종 결정한다는 원칙(§4.4, 사용자 확정 사항)을 지키기 위해,
    모델에게는 "이게 물리적으로 무엇인가"만 묻는다.

    타임아웃(settings.vlm_call_timeout_sec, placeholder - 실측 근거 없음)이나
    API 에러가 나면 예외를 삼키고 identified_object=None인 실패 값을 반환한다 -
    호출부가 이 값을 보고 "식별 실패"로 안전하게 폴백 처리할 수 있게 하기 위함
    (unknown은 안전 계열이라 식별 실패해도 항상 사람에게 물어보는 경로로
    이어지므로, 여기서 예외를 위로 전파해 로봇을 멈추게 할 필요가 없다).

    Returns:
        {"identified_object": str | None, "raw_response": str | None}
    """
    system_prompt = (
        "너는 컨베이어벨트 위에 놓인 물체를 식별하는 비전 어시스턴트다. "
        "이 이미지에 보이는 물체가 정확히 무엇인지 한국어 명사(구)로 간결하게 "
        "답해라. 이 물체를 어디로 보내야 하는지, 어떻게 처리해야 하는지는 "
        "절대 판단하지 마라 - 오직 물체가 무엇인지만 답한다. "
        '{"identified_object": string} 형식의 JSON으로만 출력해라. '
        "물체를 전혀 특정할 수 없으면 identified_object를 null로 출력해라."
    )
    user_prompt = "이 이미지에 보이는 물체는 무엇인가?"

    b64_image = base64.b64encode(image_bytes).decode("ascii")

    try:
        response = await _client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                        },
                    ],
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            timeout=settings.vlm_call_timeout_sec,
        )
    except APITimeoutError:
        logger.error("GPT-4o Vision 호출 타임아웃 (%.1fs)", settings.vlm_call_timeout_sec)
        return {"identified_object": None, "raw_response": None}
    except APIError:
        logger.error("GPT-4o Vision API 에러", exc_info=True)
        return {"identified_object": None, "raw_response": None}
    except Exception:  # noqa: BLE001
        # 예상 못한 예외(네트워크 단절 등)까지 전부 삼킨다 - unknown 판정
        # 파이프라인이 VLM 실패로 인해 통째로 죽으면 안 되고, 항상 Stage3
        # (사람에게 확인)로 안전하게 폴백해야 하기 때문.
        logger.error("GPT-4o Vision 호출 중 예기치 못한 에러", exc_info=True)
        return {"identified_object": None, "raw_response": None}

    raw = response.choices[0].message.content

    try:
        import json

        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("GPT-4o Vision 응답 JSON 파싱 실패: %r", raw)
        return {"identified_object": None, "raw_response": raw}

    identified_object = parsed.get("identified_object")
    if not identified_object or not isinstance(identified_object, str):
        identified_object = None

    return {"identified_object": identified_object, "raw_response": raw}
