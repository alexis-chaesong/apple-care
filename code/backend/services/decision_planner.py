# confidence>threshold / memory / ask_human 로직
# Vision feature + Policy Memory + Confidence를 받아 execute / ask_human 결정
# Track 2/3: 하드코딩 threshold 대신 동적 EVPI 게이트(services/human_query_gate) 사용
# Track 6: unknown 전용 Stage2(VLM) 식별 -> Stage3 게이트 연결

"""
decision_planner.py
====================
Vision Feature + Policy Memory를 결합해 "이 사과를 어떻게 할지" 최종 결정을 내리는 파일.

이 파일은 ROS도, HTTP도 모른다. Track 6부터는 unknown 전용 경로에서
services.vlm_gate(OpenAI GPT-4o Vision 호출)를 알아야 해서 "OpenAI도 모른다"는
더 이상 사실이 아니다 - 다만 이것도 여전히 services 계층 내부 협업이므로
(services.bayesian_policy/human_query_gate/decision_audit과 동일한 원칙)
계층 구조 자체는 깨지지 않는다. hitl_state_machine 같은 오케스트레이션/ROS/
웹소켓 계층은 이 파일이 직접 알지 않고, 호출부(main.py)가 n_t/tau_hold_t를
조회해서 넘겨준다.

핵심 알고리즘:
    if condition == "unknown":
        identified_object = Stage2(VLM) 식별 시도 (실패/이미지 없으면 None)
        effective_fruit_type = identified_object 또는 폴백 라벨
        # 이후로는 아래와 완전히 동일한 Stage3 게이트를 effective_fruit_type으로 통과
    if vision.confidence < confidence_threshold:
        ask_human()  # Vision 자체가 확신 못 함 (기획서 12번 예외1, Stage1 실패)
        # 주의: unknown은 이 Stage1보다 먼저 분기되어 여기로 안 옴 (아래 "Stage1
        # 우회" 설명 참고)
    else:
        condition, policy = 정책 조회
        query_human, evpi, cost = should_query_human(...)  # §4.3.5
        if query_human:
            ask_human()   # HOLD -> TTS 질문 -> STT/HMI 답변 -> posterior 갱신 -> RESUME
        else:
            execute(policy.destination)  # 위험 감수 실행 (§4.3.5 "No" 분기)

과거엔 policy.confidence를 고정 임계값(bayesian_auto_threshold=0.65)과 직접 비교했으나,
지금은 services/human_query_gate.should_query_human()이 §4.3.3(EVPI_human)과
§4.3.4(Congestion 기반 동적 비용 Cost_human(t))를 결합한 게이트로 대체함.

주의(Track4 "재발 방지" 항목): 이 파일의 유일한 "execute" 분기는 Stage1
fast-path가 아니라 Stage3의 risk-accept 분기다 - vision.confidence 게이트를
통과한 뒤에도 항상 Stage3 EVPI 게이트(should_query_human)를 거치며, 그
게이트를 생략하고 곧장 실행하는 별도 코드 경로는 없다. 그래서 execute 분기의
감사 로그는 log_stage1_auto_execute가 아니라 log_stage3_risk_accept_execute를
호출해야 한다.

Track 6 "unknown은 Stage1을 우회한다": vision_bridge.py는 condition="unknown"으로
이어지는 두 경로(depth 블롭 전용/YOLO 저신뢰) 모두에서 vision.confidence를
0.0 또는 0.4 미만으로 채운다(vision_ws 조사 결과). 이 값을 그대로 Stage1
게이트(confidence_threshold=0.7 기본값)에 통과시키면 unknown은 정책 조회도
Stage3 게이트도 못 가보고 항상 즉시 ask_human으로 반환되어, VLM 식별이나
물체별 학습이 발붙일 자리가 없다. 그래서 condition이 "unknown"으로 확정되면
Stage1 확인 자체를 건너뛰고(vision.confidence는 애초에 unknown에게 의미 있는
신호가 아님) 곧장 Stage2(VLM)로 넘어간다 - Stage2 이후에는 다시 기존 Stage3
게이트(should_query_human)를 그대로, 아무 수정 없이 통과시킨다 (아래 "unknown
전용 처리" 섹션 참고). policy=None(정책 미존재)이면 이 게이트가 이미 무조건
query_human=True를 반환하므로, "unknown은 안전 계열이라 무조건 사람에게
물어야 한다"는 요구사항은 새 분기를 추가하지 않고도 기존 게이트의 자연스러운
동작만으로 충족된다 - SAFETY_CONDITIONS에 "unknown"이 포함돼 있어(§4.3.1)
L_error가 항상 1000이 되는 것과 결합되어, 정책이 쌓여도 웬만큼 압도적으로
일관된 사람 답변이 누적되기 전까지는 계속 질문 쪽으로 기운다.

Priority Rule (기획서 12번 예외5, "Discard > Damaged > Processing > Normal"):
    사과 하나에 결함이 여러 개 동시에 적용될 수 있는 경우
    (예: 작은 사과인데 멍도 있음), 위험도가 높은 조건을 우선 적용.
    이 우선순위를 코드 레벨 상수로 고정해서 LLM 판단에 맡기지 않음
    (결정성 보장이 목적 - config_priority_reminder 참고)
"""

import logging

from models import VisionFeatureIn, DecisionResult
from services.bayesian_policy import get_policy, PolicyRecord
from services.decision_audit import (
    log_stage2_vlm_call,
    log_stage3_query_human,
    log_stage3_risk_accept_execute,
)
from services.human_query_gate import should_query_human
from services.vlm_gate import (
    UNIDENTIFIED_OBJECT_FRUIT_TYPE,
    call_gpt4o_vlm,
    get_captured_frame_for_vlm,
)
from config import settings

logger = logging.getLogger(__name__)

# Priority Rule을 코드 상수로 고정 (기획서 12번 항목).
# 인덱스가 낮을수록(=리스트 앞쪽일수록) 우선순위가 높음.
# mold(곰팡이 의심) -> Discard, bruise/scratch -> Damaged/Processing, small -> Normal 계열
CONDITION_PRIORITY = ["mold", "bruise", "scratch"]


def _resolve_condition_candidates(vision: VisionFeatureIn) -> list[str]:
    """
    Vision Feature 하나에서 실제로 확인해야 할 policy condition 후보들을
    우선순위 순서(위험도 높은 것 먼저)로 나열.

    예) size="small", defect_type="bruise" -> ["bruise", "small", "normal"]
        멍이 있으면 작다는 사실보다 멍이 우선 적용됨 (Priority Rule)
    """
    if vision.unknown_flag or vision.fruit_type == "unknown":
        # Unknown Object는 다른 조건과 절대 섞이지 않음 - 단독으로 처리
        return ["unknown"]

    candidates: list[str] = []

    if vision.defect_type in CONDITION_PRIORITY:
        candidates.append(vision.defect_type)

    if vision.size == "small":
        candidates.append("small")

    candidates.append("normal")
    # 항상 마지막 fallback으로 "normal"을 둠 - 어떤 조건도 안 걸리면 정상 취급
    return candidates


def _find_matching_policy(
    fruit_type: str, candidates: list[str]
) -> tuple[str, PolicyRecord | None]:
    """
    후보 condition들을 우선순위 순서대로 조회해서,
    실제로 DB에 정책이 존재하는 첫 번째 조합을 반환.

    존재하는 정책이 하나도 없으면 -> 가장 위험도 높은 후보(candidates[0])를 반환하되
    policy는 None. (예: 처음 보는 결함 유형이면 그 결함 기준으로 ask_human 처리되어야
    "이 결함은 아직 모르는 조건이다"라는 사실이 로그에 남음)
    """
    for condition in candidates:
        policy = get_policy(fruit_type, condition)
        if policy is not None:
            return condition, policy
    return candidates[0], None


def _stage3_decide(
    vision: VisionFeatureIn,
    fruit_type: str,
    condition: str,
    policy: PolicyRecord | None,
    n_t: int,
    tau_hold_t: float,
) -> tuple[DecisionResult, bool, float, float]:
    """Stage3 EVPI 게이트(§4.3.5)를 실행해 최종 DecisionResult를 만들고,
    Stage3 감사 로그(log_stage3_query_human/log_stage3_risk_accept_execute)까지
    남긴다.

    일반 경로(decide())와 Track6 unknown 경로(_decide_unknown_object())가 이
    함수 하나를 공유한다 - "게이트 로직은 단 한 곳에만 있다"는 원칙을 지키기
    위함이며, unknown이라고 해서 이 게이트를 우회하거나 새 분기를 추가하지
    않는다.

    fruit_type은 vision.fruit_type과 다를 수 있다(Track6: unknown 경로에서
    VLM이 식별한 이름으로 대체됨). 그래서 감사 로그에도 vision을 그대로
    넘기지 않고, fruit_type을 반영한 사본(vision.model_copy)을 넘겨 audit
    행의 fruit_type이 실제 정책 조회에 쓰인 값과 일치하도록 한다 - 안 그러면
    tb_decision_audit에는 "unknown"만 남고 VLM이 식별한 이름은
    log_stage2_vlm_call의 vlm_response 안에만 파묻혀 추적이 어려워진다.

    Returns:
        (DecisionResult, query_human, evpi, cost) - 뒤 3개는 호출부(Track6
        unknown 경로)가 Stage2 감사 로그의 final_action/destination을 결정할
        때 재사용한다.
    """
    vision_for_audit = (
        vision if fruit_type == vision.fruit_type else vision.model_copy(update={"fruit_type": fruit_type})
    )

    query_human, evpi, cost = should_query_human(
        fruit_type, condition, policy, n_t, tau_hold_t,
        settings.human_query_cost_k1, settings.human_query_cost_k2,
    )

    if query_human:
        reason = "unknown_object" if condition == "unknown" else "low_confidence"
        logger.info(
            "Stage3 게이트: 사람에게 확인 (EVPI=%.2f > Cost=%.2f 또는 정책 부재/ask_human): "
            "fruit=%s condition=%s policy=%s",
            evpi, cost, fruit_type, condition, policy,
        )
        log_stage3_query_human(
            vision_for_audit, condition,
            theta_exists=policy is not None,
            dstored=policy["destination"] if policy else None,
            evpi_human=evpi, cost_human=cost, n_pending=n_t, tau_hold_sec=tau_hold_t,
        )
        result = DecisionResult(
            action="ask_human",
            destination=None,
            reason=reason,
            fruit_type=fruit_type,
            condition=condition,
            confidence=vision.confidence,
            position=vision.center,
        )
        return result, query_human, evpi, cost

    # 위험 감수 실행 (§4.3.5 "No" 분기) - EVPI_human <= Cost_human(t)이므로 정책을 신뢰.
    # source(llm_policy/human_feedback)와 무관하게 reason은 항상 "risk_accept_execute" -
    # 이 분기에 도달하는 모든 execute는 EVPI vs Cost 비교를 통과한 위험 감수 판단의
    # 결과이지 "규칙이라 무조건 신뢰"가 아니기 때문 (명시적으로 로그에 남길 것, Track2/3 요구사항)
    logger.info(
        "Stage3 게이트: 위험 감수 실행 (EVPI=%.2f <= Cost=%.2f): fruit=%s condition=%s destination=%s",
        evpi, cost, fruit_type, condition, policy["destination"],
    )
    log_stage3_risk_accept_execute(
        vision_for_audit, condition, policy["destination"],
        evpi_human=evpi, cost_human=cost, n_pending=n_t, tau_hold_sec=tau_hold_t,
    )
    result = DecisionResult(
        action="execute",
        destination=policy["destination"],
        reason="risk_accept_execute",
        fruit_type=fruit_type,
        condition=condition,
        confidence=vision.confidence,
        position=vision.center,
    )
    return result, query_human, evpi, cost


async def _decide_unknown_object(
    vision: VisionFeatureIn, n_t: int, tau_hold_t: float
) -> DecisionResult:
    """Track6: condition="unknown" 전용 경로 (§4.4).

    Stage2(VLM 식별)를 먼저 시도해 "이게 무엇인지"를 알아낸 뒤, 그 이름을
    fruit_type 삼아 기존 Stage3 게이트(_stage3_decide, 수정 없음)를 그대로
    통과시킨다. 이렇게 해야 같은 물체(예: 망치)가 나중에 또 나왔을 때
    (fruit_type="hammer", condition="unknown") 조합으로 정책을 조회해 학습된
    결과를 재사용할 수 있다 - VLM 식별 없이 vision.fruit_type("unknown"
    문자열 그대로)을 계속 썼다면 모든 미확인 물체가 구분 없이 정책 하나를
    공유해서 물체별 학습이 애초에 불가능했을 것이다.

    §4.4 원칙: VLM 응답은 posterior(theta_f,c)를 절대 직접 갱신하지 않는다 -
    여기서는 tb_decision_audit에 기록만 하고(log_stage2_vlm_call),
    tb_policy_memory는 전혀 건드리지 않는다. posterior 갱신은 사람이 실제로
    답변한 뒤 hitl_state_machine.py가 bayesian_policy.record_human_feedback()을
    호출할 때만 일어난다(그때 fruit_type은 여기서 반환하는 DecisionResult의
    fruit_type을 그대로 물려받는다 - main.py -> hitl_state_machine.
    handle_ask_human() -> HITLSession.fruit_type 경로).
    """
    condition = "unknown"
    identified_object: str | None = None
    vlm_response: dict | None = None
    vlm_called = False

    # 트리거 단순화(§4.4 EVSI_VLM(x_t) > Cost_VLM의 잠정 근사, vlm_gate.py
    # 모듈 docstring 참고): condition이 unknown으로 확정되면 무조건 시도한다.
    # 실제 시도 여부는 이미지 캡처 가능 여부로 자연히 게이팅된다.
    # Task1부터 get_captured_frame_for_vlm()이 실제 ROS2 서비스(capture_frame)를
    # 호출하는 async 함수라 await 필요 - 더 이상 항상 None을 반환하는 스텁이 아님.
    frame = await get_captured_frame_for_vlm(vision.frame_id or "")
    if frame is not None:
        vlm_called = True
        vlm_response = await call_gpt4o_vlm(frame)
        identified_object = vlm_response.get("identified_object")
        if identified_object:
            logger.info("Stage2 VLM 식별 성공: %s", identified_object)
        else:
            logger.warning("Stage2 VLM 식별 실패 또는 응답 파싱 실패 - 미상 라벨로 폴백")
    else:
        logger.info("Stage2 VLM 스킵 - 이미지 캡처 실패/미연결(get_captured_frame_for_vlm=None)")

    effective_fruit_type = identified_object or UNIDENTIFIED_OBJECT_FRUIT_TYPE

    policy = get_policy(effective_fruit_type, condition)
    result, query_human, evpi, cost = _stage3_decide(
        vision, effective_fruit_type, condition, policy, n_t, tau_hold_t
    )

    if vlm_called:
        # 성공/실패와 무관하게 기록한다(§4.4) - 방금 계산된 Stage3 게이트
        # 결과(final_action/destination)를 그대로 반영해 "VLM이 이렇게
        # 답했고, 그 결과 최종적으로 이렇게 됐다"를 한 행에서 확인 가능하게 함.
        log_stage2_vlm_call(
            vision, condition,
            stage2_evsi_gate_passed=True,
            vlm_response=vlm_response,
            final_action="hold" if query_human else "execute",
            final_destination=None if query_human else policy["destination"] if policy else None,
        )

    return result


async def decide(vision: VisionFeatureIn, n_t: int = 0, tau_hold_t: float = 0.0) -> DecisionResult:
    """
    이 프로젝트의 핵심 의사결정 함수. hitl_router.py, vla_router.py의
    consumer 루프가 Vision Feature를 받을 때마다 이 함수 하나만 호출하면 됨.

    Track6부터 이 함수는 async다 - condition="unknown" 경로가 실제 GPT-4o
    Vision 네트워크 호출(services.vlm_gate.call_gpt4o_vlm)을 할 수 있어서,
    동기 함수로 두면 그 호출이 끝날 때까지 이벤트 루프 전체(HMI 웹소켓,
    로봇 상태 브로드캐스트 등)가 멈춘다. 그 외 경로(정책 조회, Stage3 게이트)는
    여전히 순수 동기 DB 조회뿐이라 빠르다.

    Args:
        vision: Vision이 관측한 x_t (c_yolo, defect_type, position 등).
        n_t: §4.3.4 n(t) — 현재 HITL 요청 backlog(대기열+활성 세션).
            hitl_state_machine.congestion_n에서 조회해 호출부(main.py)가 넘겨준다.
            기본값 0은 "혼잡 없음"으로, 혼잡도를 모르는 호출부(테스트 등)를 위한 안전한 기본값.
        tau_hold_t: §4.3.4 tau_hold(t) — 최장 대기 세션의 HOLD 지속 시간(초).
            hitl_state_machine.congestion_tau_hold에서 조회.
    """
    # 0) Condition 후보를 먼저 산출 (DB 조회와 무관하게 vision 정보만으로 결정됨).
    candidates = _resolve_condition_candidates(vision)

    # Track6: unknown은 Stage1(vision.confidence) 검사보다 먼저 분기된다.
    # vision_bridge.py는 unknown으로 이어지는 두 경로(depth 블롭 전용/YOLO
    # 저신뢰) 모두에서 confidence를 0.0 또는 0.4 미만으로 채우므로, 이걸
    # 그대로 Stage1(기본 임계값 0.7)에 통과시키면 unknown은 정책 조회도
    # Stage2(VLM)도 못 가보고 항상 즉시 ask_human으로 끝나버린다 (모듈
    # docstring "unknown은 Stage1을 우회한다" 참고).
    if candidates[0] == "unknown":
        return await _decide_unknown_object(vision, n_t, tau_hold_t)

    # 1) Vision 자체가 확신 못 하는 경우 - 정책/메모리와 무관하게 무조건 확인
    #    (기획서 12번 예외1: "AI가 확신하지 못하는 경우에는 강제로 분류하지 않는다",
    #     Stage1 실패 - EVPI/Cost 계산 없이 무조건 질문)
    if vision.confidence < settings.confidence_threshold:
        logger.info(
            "낮은 confidence(%.2f < %.2f)로 인해 ask_human 처리: fruit=%s",
            vision.confidence, settings.confidence_threshold, vision.fruit_type,
        )
        audit_policy = get_policy(vision.fruit_type, candidates[0])
        log_stage3_query_human(
            vision, candidates[0],
            theta_exists=audit_policy is not None,
            dstored=audit_policy["destination"] if audit_policy else None,
        )
        return DecisionResult(
            action="ask_human",
            destination=None,
            reason="low_confidence",
            fruit_type=vision.fruit_type,
            condition=candidates[0],
            confidence=vision.confidence,
            position=vision.center,
        )

    # 2) 매칭되는 정책 조회
    condition, policy = _find_matching_policy(vision.fruit_type, candidates)

    # 3)+4) Stage3 EVPI 게이트 (§4.3.5) 실행 + 결과 구성/로깅
    result, _query_human, _evpi, _cost = _stage3_decide(
        vision, vision.fruit_type, condition, policy, n_t, tau_hold_t
    )
    return result
