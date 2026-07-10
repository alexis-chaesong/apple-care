# Track 2/3: services/human_query_gate.py 단위 테스트

"""
test_human_query_gate.py
==========================
services/human_query_gate.py (Stage3 Human Query Gate, 리서치 문서 §4.3.3/§4.3.4/§4.3.5)에
대한 pytest 단위 테스트.

과거 하드코딩 게이트(policy.confidence < settings.bayesian_auto_threshold=0.65)를
대체하는 EVPI_human vs Cost_human(t) 게이트가 사양대로 동작하는지 검증한다.

사용법:
    cd code/backend && pytest test_human_query_gate.py -v
"""

import pytest

from services.human_query_gate import RAMP_CAPACITY_C, compute_p, cost_human, evpi_human, should_query_human


def _policy(destination, alpha, beta, confidence):
    """PolicyRecord 형태의 최소 dict. TypedDict라 런타임엔 그냥 dict면 충분."""
    return {
        "fruit_type": "apple",
        "condition": "small",
        "destination": destination,
        "alpha": alpha,
        "beta": beta,
        "confidence": confidence,
        "source": "human_feedback",
        "updated_at": "2026-01-01 00:00:00",
    }


def test_no_policy_always_queries_human_regardless_of_congestion():
    """§4.3.5: "query_human <=> (theta_f,c = ∅) ∨ ...". 정책이 아예 없으면
    n(t)/tau_hold(t)가 무엇이든(혼잡이 전혀 없어도, 최대치에 가까워도) 항상 True."""
    query_human, evpi, cost = should_query_human("apple", "small", None, 0, 0, 1, 1)
    assert query_human is True

    query_human_congested, _, _ = should_query_human("apple", "small", None, 5, 100, 1, 1)
    assert query_human_congested is True


def test_compute_p_defaults_to_half_when_no_policy():
    """§4.3.3: "정책이 아직 존재하지 않는 완전 신규 조합은 p=0.5(최대 불확실성)"."""
    assert compute_p(None) == pytest.approx(0.5)


def test_high_confidence_policy_flips_query_human_to_false():
    """c="small", dstored="normal_box" -> L_error = normal 행 최댓값 = 10.
    n_t=1, k1=k2=1.0 -> cost_human = 1*(1/(6-1)) = 0.2 (§4.3.4).

    p=0.5(신규/낮은 신뢰) -> evpi=(1-0.5)*10=5.0 > 0.2 -> query_human=True
    p=0.99(매우 높은 신뢰) -> evpi=(1-0.99)*10=0.1 < 0.2 -> query_human=False
    """
    low_confidence_policy = _policy("normal_box", 1, 1, 0.5)
    high_confidence_policy = _policy("normal_box", 99, 1, 0.99)

    query_low, evpi_low, cost_low = should_query_human("apple", "small", low_confidence_policy, 1, 0, 1, 1)
    query_high, evpi_high, cost_high = should_query_human("apple", "small", high_confidence_policy, 1, 0, 1, 1)

    assert cost_low == cost_high == pytest.approx(0.2)
    assert evpi_low == pytest.approx(5)
    assert evpi_high == pytest.approx(0.1)
    assert query_low is True
    assert query_high is False
    assert evpi_high < evpi_low


def test_congestion_near_capacity_suppresses_query_human():
    """c="scratch", dstored="ugly_box" -> L_error = ugly 행 최댓값 = 5.
    policy confidence=0.5(alpha=beta=1) -> evpi=(1-0.5)*5=2.5 (n(t)/tau_hold(t)와 무관하게 고정).

    n_t=0(혼잡 없음) -> cost=0 -> 2.5 > 0 -> query_human=True
    n_t=5(=C-1, 경사로 거의 가득 참) -> cost=1*(5/(6-5))=5.0 -> 2.5 <= 5.0 -> query_human=False
    (동일한 EVPI인데도 혼잡도만으로 게이트 결과가 뒤집힘 = 혼잡 회피 확인)
    """
    policy = _policy("ugly_box", 1, 1, 0.5)

    query_idle, evpi_idle, cost_idle = should_query_human("apple", "scratch", policy, 0, 0, 1, 1)
    query_congested, evpi_congested, cost_congested = should_query_human("apple", "scratch", policy, 5, 0, 1, 1)

    assert evpi_idle == evpi_congested == pytest.approx(2.5)
    assert cost_idle == pytest.approx(0)
    assert cost_congested == pytest.approx(5)
    assert query_idle is True
    assert query_congested is False
    assert cost_congested > cost_idle


def test_cost_human_barrier_function_diverges_at_capacity():
    """§4.3.4: "C = 6: 경사로 물리적 수용 한계 — barrier function이 n → 6에서
    발산해 DeCCaF의 hard constraint를 연속적 자기회피 압력으로 대체".

    n(t)가 0..5로 늘어날수록 비용이 단조 증가하고, n(t)=C=6에 도달하면
    (분모 C-n=0) 명시적으로 무한대를 반환해야 한다.
    """
    assert RAMP_CAPACITY_C == 6

    costs = [cost_human(n_t=n, tau_hold_t=0, k1=1, k2=1) for n in range(RAMP_CAPACITY_C)]
    for earlier, later in zip(costs, costs[1:]):
        assert later > earlier
    assert costs[-1] == pytest.approx(5)

    assert cost_human(n_t=RAMP_CAPACITY_C, tau_hold_t=0, k1=1, k2=1) == float("inf")
    assert cost_human(n_t=RAMP_CAPACITY_C + 1, tau_hold_t=0, k1=1, k2=1) == float("inf")


def test_cost_human_tau_hold_term_adds_linearly():
    """k2 * tau_hold(t) 항이 congestion 항과 독립적으로 선형으로 더해지는지 확인."""
    base = cost_human(n_t=0, tau_hold_t=0, k1=1, k2=2)
    with_hold = cost_human(n_t=0, tau_hold_t=3, k1=1, k2=2)
    assert base == pytest.approx(0)
    assert with_hold == pytest.approx(6)


def test_dstored_ask_human_forces_query_regardless_of_evpi():
    """§4.3.5: "query_human <=> ... ∨ (d_stored = ask_human) ∨ ...".

    confidence=1.0(p=1.0)이면 EVPI_human=(1-1.0)*L_error=0이 되어, 혼잡이 전혀
    없는 상황(cost=0)에서는 "EVPI > Cost" 비교만으로는 0 > 0 = False가 나와야
    정상이다. 그런데도 dstored="ask_human"이면 이 비교 결과와 무관하게
    query_human=True가 되어야 한다.
    """
    policy = _policy("ask_human", 99, 1, 1)
    evpi = evpi_human("apple", "small", policy)
    cost = cost_human(n_t=0, tau_hold_t=0, k1=1, k2=1)
    assert evpi == pytest.approx(0)
    assert cost == pytest.approx(0)
    assert not (evpi > cost)

    query_human, returned_evpi, returned_cost = should_query_human("apple", "small", policy, 0, 0, 1, 1)
    assert query_human is True
    assert returned_evpi == pytest.approx(evpi)
    assert returned_cost == pytest.approx(cost)


def test_compute_p_trusts_stored_confidence_not_recomputed_alpha_beta():
    """upsert_llm_policy()가 만드는 llm_policy 행은 alpha=beta=1.0(prior)이지만
    confidence=1.0으로 고정 저장된다(즉시 최고 신뢰). compute_p()는 alpha/beta로
    재계산하지 말고 저장된 confidence를 그대로 신뢰해야 한다."""
    llm_policy_row = _policy("discard_box", 1, 1, 1)
    assert compute_p(llm_policy_row) == pytest.approx(1)


def test_safety_condition_evpi_is_high_even_with_confident_policy():
    """안전 계열(mold/rotten/unknown)은 L_error가 항상 1000이라, 어지간히 높은
    신뢰도가 아니면 EVPI가 여전히 커서 쉽게 query_human=False로 전환되지 않는다."""
    policy = _policy("processing_box", 9, 1, 0.9)
    evpi = evpi_human("apple", "unknown", policy)
    assert evpi == pytest.approx((1 - 0.9) * 1000)


def test_normal_condition_does_not_raise_and_matches_value_series_proxy():
    """§4.3.1이 명시적으로 분류하지 않는 "normal"(결함 없음, decision_planner.py의
    기본 fallback condition)도 예외 없이 계산되어야 하며, 계산 결과는 dstored만
    같으면 실제 VALUE_CONDITIONS 원소(예: "small")로 계산한 것과 동일해야 한다."""
    policy = _policy("normal_box", 1, 1, 0.5)
    evpi_normal = evpi_human("apple", "normal", policy)
    evpi_small = evpi_human("apple", "small", policy)
    assert evpi_normal == pytest.approx(evpi_small)
