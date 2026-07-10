# Track 2/3: 동적 EVPI 게이트 (기존 하드코딩 threshold 0.65 대체)

"""
human_query_gate.py
======================
리서치 문서 §4.3.3(EVPI 계산), §4.3.4(Congestion 기반 동적 비용),
§4.3.5(최종 게이트 조건)을 코드로 옮긴 파일.

이 파일은 기존에 decision_planner.py에 하드코딩되어 있던
"policy['confidence'] < settings.bayesian_auto_threshold(0.65)" 게이트를
대체한다. config.py의 bayesian_auto_threshold 필드와 bayesian_policy.py의
should_ask_human()(둘 다 이 하드코딩 게이트를 구현하던 죽은 코드)는 이 모듈
도입과 함께 제거되었다.

의존 관계: services.bayesian_policy(PolicyRecord 타입, θ_f,c의 alpha/beta/
confidence)와 services.loss_matrix(§4.1/4.3.1/4.3.2, L_error)만 참조한다.
Beta(α,β) posterior 갱신 자체(§4.4 마지막 줄, alpha += 1[일치] 등)는
bayesian_policy.record_human_feedback()이 이미 담당하므로 이 파일은
건드리지 않는다 - 이 파일은 오직 "물어볼지 말지"만 결정한다.
"""

from typing import Optional

from services.bayesian_policy import PolicyRecord
from services.loss_matrix import L_error, SAFETY_CONDITIONS, VALUE_CONDITIONS

# §4.3.4 원문: "C = 6: 경사로 물리적 수용 한계 — barrier function이 n → 6에서
# 발산해 DeCCaF의 hard constraint를 연속적 자기회피 압력으로 대체"
RAMP_CAPACITY_C = 6

# condition "normal"(무결점, §4.3.1에 미분류)을 위한 gap-fill: L_error()는
# VALUE_CONDITIONS 계열이면 실제로 어느 condition 문자열이 넘어오든(c 자체를
# 계산에 쓰지 않고 dstored만으로 손실을 결정) 동일한 결과를 반환하므로,
# VALUE_CONDITIONS 중 아무 원소나 "정상 프록시"로 재사용해도 수치적으로 안전하다.
_VALUE_SERIES_PROXY_CONDITION = next(iter(VALUE_CONDITIONS))


def compute_p(policy: Optional[PolicyRecord]) -> float:
    """§4.3.3: "p = alpha_f,c / (alpha_f,c + beta_f,c) (저장된 정책이 맞을 사후 확률)".

    이 코드베이스에서는 tb_policy_memory.confidence 컬럼이 이미 alpha/(alpha+beta)로
    계산되어 저장되므로(bayesian_policy.record_human_feedback 참고) 이를 그대로
    사용한다. source="llm_policy"(작업자가 자연어로 직접 지정한 확정 규칙) 행은
    alpha/beta가 prior(1.0/1.0)에 머물러 있어도 confidence를 1.0으로 고정
    저장해두므로(upsert_llm_policy, 즉시 최고 신뢰), 여기서 alpha/beta로
    재계산하면 이 명시적 규칙을 무시하게 된다 - 반드시 저장된 confidence를
    그대로 신뢰해야 한다.

    "정책이 아직 존재하지 않는 완전 신규 조합은 p=0.5(최대 불확실성)로 간주"
    """
    if policy is None:
        return 0.5
    return policy["confidence"]


def evpi_human(f: str, c: str, policy: Optional[PolicyRecord]) -> float:
    """§4.3.3: "EVPI_human(theta_f,c) = (1 - p) * L_error(f, c)"."""
    p = compute_p(policy)
    dstored = policy["destination"] if policy is not None else None
    theta_f_c = policy["confidence"] if policy is not None else None

    if c in SAFETY_CONDITIONS or c in VALUE_CONDITIONS:
        loss = L_error(f, c, theta_f_c, dstored)
    else:
        loss = L_error(f, _VALUE_SERIES_PROXY_CONDITION, theta_f_c, dstored)
    return (1 - p) * loss


def cost_human(n_t: int, tau_hold_t: float, k1: float, k2: float) -> float:
    """§4.3.4: "Cost_human(t) = k1 * n(t)/(C-n(t)) + k2 * tau_hold(t)".

    "C = 6: 경사로 물리적 수용 한계 — barrier function이 n → 6에서 발산해
     DeCCaF의 hard constraint를 연속적 자기회피 압력으로 대체"
    n(t)가 물리적 한계(C=6)에 도달하면(또는 어떤 이유로 그 이상이 되면)
    barrier function을 명시적으로 무한대로 발산시켜, 그 이상으로는 절대
    사람에게 새로 묻지 않도록(항상 위험 감수 실행 쪽으로 강제) 만든다.

    k1, k2: "잠정 상수(placeholder). Vision·Robot 모듈과의 완전한 물리적
    통합 이전에는 실측 근거가 없으므로 임의값으로 시작하고, 통합 이후 실제
    컨베이어 운용 데이터로 재보정한다 (§8 Implementation Status 참고)."
    (config.settings.human_query_cost_k1/k2로 관리, 호출부에서 주입받음)
    """
    if n_t >= RAMP_CAPACITY_C:
        return float("inf")
    congestion_term = k1 * (n_t / (RAMP_CAPACITY_C - n_t))
    hold_term = k2 * tau_hold_t
    return congestion_term + hold_term


def should_query_human(
    f: str,
    c: str,
    policy: Optional[PolicyRecord],
    n_t: int,
    tau_hold_t: float,
    k1: float,
    k2: float,
) -> tuple[bool, float, float]:
    """§4.3.5 최종 게이트 조건:

        "query_human <=> (theta_f,c = ∅) ∨ (d_stored = ask_human) ∨ (EVPI > Cost)"

    Returns:
        (query_human, evpi_human 값, cost_human 값) — 뒤 두 값은 게이트 판단과
        무관하게 항상 계산해서 반환한다 (theta_f,c가 없거나 dstored가
        "ask_human"이어도 tb_decision_audit에 근거로 남길 수 있도록).
    """
    evpi = evpi_human(f, c, policy)
    cost = cost_human(n_t, tau_hold_t, k1, k2)

    if policy is None:
        return (True, evpi, cost)
    if policy["destination"] == "ask_human":
        return (True, evpi, cost)
    return (evpi > cost, evpi, cost)
