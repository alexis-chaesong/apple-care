# confidence>threshold / memory / ask_human 로직
# Vision feature + Policy Memory + Confidence를 받아 execute / ask_human 결정

"""
decision_planner.py
====================
Vision Feature + Policy Memory를 결합해 "이 사과를 어떻게 할지" 최종 결정을 내리는 파일.

이 파일은 OpenAI도, ROS도, HTTP도 모른다.
오직 VisionFeatureIn을 받아서 DecisionResult를 내놓는 순수 판단 로직만 책임진다.
(models.py, bayesian_policy.py만 알면 됨 - services 내부 계층 간 원칙 그대로 유지)

핵심 알고리즘 (기획서 4번 항목 그대로):
    if vision.confidence < threshold:
        ask_human()  # Vision 자체가 확신 못 함
    elif policy_exists and policy.confidence >= bayesian_auto_threshold:
        execute(policy.destination)  # 규칙 또는 학습된 정책을 신뢰
    else:
        ask_human()  # 정책이 없거나 아직 신뢰가 덜 쌓임

Priority Rule (기획서 12번 예외5, "Discard > Damaged > Processing > Normal"):
    사과 하나에 결함이 여러 개 동시에 적용될 수 있는 경우
    (예: 작은 사과인데 멍도 있음), 위험도가 높은 조건을 우선 적용.
    이 우선순위를 코드 레벨 상수로 고정해서 LLM 판단에 맡기지 않음
    (결정성 보장이 목적 - config_priority_reminder 참고)
"""

import logging

from models import VisionFeatureIn, DecisionResult
from services.bayesian_policy import get_policy, PolicyRecord
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


def decide(vision: VisionFeatureIn) -> DecisionResult:
    """
    이 프로젝트의 핵심 의사결정 함수. hitl_router.py, vla_router.py의
    consumer 루프가 Vision Feature를 받을 때마다 이 함수 하나만 호출하면 됨.
    """

    # 0) Condition 후보를 먼저 산출 (DB 조회와 무관하게 vision 정보만으로 결정됨).
    #    confidence 미달로 조기 반환하는 경우에도 DecisionResult.condition을
    #    채워야 하므로, 정책 조회(_find_matching_policy)보다 먼저 계산해둠
    candidates = _resolve_condition_candidates(vision)

    # 1) Vision 자체가 확신 못 하는 경우 - 정책/메모리와 무관하게 무조건 확인
    #    (기획서 12번 예외1: "AI가 확신하지 못하는 경우에는 강제로 분류하지 않는다")
    if vision.confidence < settings.confidence_threshold:
        logger.info(
            "낮은 confidence(%.2f < %.2f)로 인해 ask_human 처리: fruit=%s",
            vision.confidence, settings.confidence_threshold, vision.fruit_type,
        )
        return DecisionResult(
            action="ask_human",
            destination=None,
            reason="low_confidence",
            fruit_type=vision.fruit_type,
            condition=candidates[0],
            confidence=vision.confidence,
        )

    # 2) 매칭되는 정책 조회
    condition, policy = _find_matching_policy(vision.fruit_type, candidates)

    # 3) 정책이 아예 없거나(신규 조건), 있어도 신뢰도가 auto_threshold 미달이거나,
    #    명시적으로 "ask_human"으로 지정된 경우 -> 사람에게 확인
    if (
        policy is None
        or policy["destination"] == "ask_human"
        or policy["confidence"] < settings.bayesian_auto_threshold
    ):
        reason = "unknown_object" if condition == "unknown" else "low_confidence"
        logger.info(
            "정책 신뢰 부족으로 ask_human 처리: fruit=%s condition=%s policy=%s",
            vision.fruit_type, condition, policy,
        )
        return DecisionResult(
            action="ask_human",
            destination=None,
            reason=reason,
            fruit_type=vision.fruit_type,
            condition=condition,
            confidence=vision.confidence,
        )

    # 4) 신뢰할 수 있는 정책 존재 -> 자동 실행
    #    source가 llm_policy(작업자가 직접 지정한 명시적 규칙)인지,
    #    human_feedback(점진적으로 학습된 정책)인지에 따라 reason만 구분
    #    (둘 다 destination은 그대로 신뢰하고 실행함)
    reason = "memory_match" if policy["source"] == "human_feedback" else "rule_match"
    logger.info(
        "정책 매칭으로 자동 실행: fruit=%s condition=%s destination=%s (reason=%s)",
        vision.fruit_type, condition, policy["destination"], reason,
    )
    return DecisionResult(
        action="execute",
        destination=policy["destination"],
        reason=reason,
        fruit_type=vision.fruit_type,
        condition=condition,
        confidence=vision.confidence,
    )