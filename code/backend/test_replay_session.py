# Track 5-prep: replay_session.py 단위 테스트 (synthetic/dummy 로그)

"""
test_replay_session.py
=========================
replay_session.py(§5.2 검증 리플레이 스크립트)에 대한 pytest.

실제 사과 5세트 데이터가 아직 없으므로, 여기서는 더미/합성 tb_decision_audit
로그를 직접 만들어 스크립트의 동작(로딩/시간 근사/두 게이팅 정책 재현/지표
집계/리포트 생성)만 검증한다. 실제 data/robot_system.db 대신 tmp_path 임시
SQLite 파일 사용.

사용법:
    cd code/backend && pytest test_replay_session.py -v
"""

from datetime import datetime, timedelta

import pytest

import database
import replay_session as rs
from config import settings


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "test_replay.db"))
    database.init_db()
    yield


def _insert_row(
    branch, timestamp, fruit_type, condition,
    *, n_pending=None, tau_hold_sec=None, actual_label=None, session_id=None, final_action="execute",
):
    """tb_decision_audit에 합성 행 1건을 직접 삽입 (테스트 전용 - 실제 코드는
    services/decision_audit.py의 log_*() 함수만 사용해야 함). timestamp를
    자유롭게 통제해야 재생 로직을 결정적으로 검증할 수 있어 raw INSERT를 씀."""
    conn = database.get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO tb_decision_audit
                (branch, timestamp, fruit_type, condition, n_pending, tau_hold_sec,
                 actual_label, session_id, final_action, vlm_called)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (branch, timestamp.isoformat(), fruit_type, condition, n_pending, tau_hold_sec, actual_label, session_id, final_action),
        )
        conn.commit()
    finally:
        conn.close()


T0 = datetime(2026, 7, 9, 9, 0, 0)


def test_load_decision_events_filters_branches_and_orders_by_timestamp(isolated_db):
    _insert_row("stage3_risk_accept_execute", T0 + timedelta(seconds=2), "apple", "normal")
    _insert_row("stage3_query_human", T0 + timedelta(seconds=0), "apple", "unknown")
    _insert_row("stage3_human_resolved", T0 + timedelta(seconds=1), "apple", "unknown")
    _insert_row("prior_initialized", T0 + timedelta(seconds=1, milliseconds=500), "apple", "unknown")

    events = rs.load_decision_events()
    assert [e.branch for e in events] == ["stage3_query_human", "stage3_risk_accept_execute"]
    assert events[0].timestamp < events[1].timestamp


def test_load_decision_events_filters_by_session_ids(isolated_db):
    _insert_row("stage3_query_human", T0, "apple", "unknown", session_id="s1")
    _insert_row("stage3_query_human", T0 + timedelta(seconds=1), "kiwi", "unknown", session_id="s2")

    events = rs.load_decision_events(session_ids=["s2"])
    assert len(events) == 1
    assert events[0].fruit_type == "kiwi"


def test_estimate_ask_duration_sec_uses_matched_query_resolved_pairs():
    queries = [T0, T0 + timedelta(seconds=10)]
    resolved = [T0 + timedelta(seconds=3), T0 + timedelta(seconds=14)]
    avg = rs.estimate_ask_duration_sec(queries, resolved)
    assert avg == pytest.approx(3.5)


def test_estimate_ask_duration_sec_falls_back_when_no_data():
    assert rs.estimate_ask_duration_sec([], []) == pytest.approx(settings.hitl_response_timeout_sec)
    assert rs.estimate_ask_duration_sec([T0], []) == pytest.approx(settings.hitl_response_timeout_sec)


def test_estimate_execute_interval_sec_ignores_query_branches():
    events = [
        rs.DecisionEvent(1, "stage3_risk_accept_execute", T0, "apple", "normal", 0, 0, None, None),
        rs.DecisionEvent(2, "stage3_risk_accept_execute", T0 + timedelta(seconds=2), "apple", "normal", 0, 0, None, None),
        rs.DecisionEvent(3, "stage3_query_human", T0 + timedelta(seconds=100), "apple", "unknown", 0, 0, None, None),
        rs.DecisionEvent(4, "stage3_risk_accept_execute", T0 + timedelta(seconds=104), "apple", "normal", 0, 0, None, None),
    ]
    avg = rs.estimate_execute_interval_sec(events)
    assert avg == pytest.approx(2)


def test_estimate_execute_interval_sec_falls_back_when_no_pure_pairs():
    events = [
        rs.DecisionEvent(1, "stage3_query_human", T0, "apple", "unknown", 0, 0, None, None),
        rs.DecisionEvent(2, "stage3_query_human", T0 + timedelta(seconds=1), "apple", "unknown", 0, 0, None, None),
    ]
    assert rs.estimate_execute_interval_sec(events) == pytest.approx(settings.vision_poll_interval_sec)


def test_fixed_threshold_stops_asking_once_confidence_crosses_065_no_congestion(isolated_db):
    """apple/scratch가 항상 실제로도 "processing"(참값)인 8건 연속 관측.
    고정 threshold(0.65)는 최초 1회만 묻고, confidence=2/3≈0.667로 임계값을
    넘은 뒤부터는 계속 자동 실행 -> 항상 정답이므로 손실도 0."""
    events = [
        rs.DecisionEvent(i, "stage3_risk_accept_execute", T0 + timedelta(seconds=i), "apple", "scratch", 0, 0, "processing", None)
        for i in range(8)
    ]
    set_rows = rs.replay(events, items_per_set=100, ask_duration_sec=5, execute_interval_sec=1)
    assert len(set_rows) == 1
    fixed = set_rows[0].fixed
    assert fixed.query_count == 1
    assert fixed.total_loss == pytest.approx(0)
    assert fixed.n_events == 8


def test_dynamic_evpi_keeps_asking_under_zero_congestion(isolated_db):
    """같은 8건을 동적 EVPI로 재생하면, 혼잡이 전혀 없어(Cost_human=0) EVPI>0인 한
    계속 사람을 부른다 (§4.3.4: 물어보는 데 비용이 안 드는 상황에서는 조금이라도
    남은 불확실성이 있으면 계속 확인) - 고정 threshold보다 훨씬 더 자주 묻는다."""
    events = [
        rs.DecisionEvent(i, "stage3_risk_accept_execute", T0 + timedelta(seconds=i), "apple", "scratch", 0, 0, "processing", None)
        for i in range(8)
    ]
    set_rows = rs.replay(events, items_per_set=100, ask_duration_sec=5, execute_interval_sec=1)
    dynamic = set_rows[0].dynamic
    assert dynamic.query_count == 8
    assert dynamic.total_loss == pytest.approx(0)


def test_dynamic_evpi_stops_asking_under_heavy_congestion(isolated_db):
    """동일 시나리오에 n_pending=5(=C-1)로 혼잡을 걸면, Cost_human(t)이 급등해
    (§4.3.4) 동적 EVPI도 첫 질문 이후부터는 곧장 자동 실행으로 전환된다 -
    혼잡 회피 확인 (test_dynamic_evpi_keeps_asking_under_zero_congestion과 대비)."""
    events = [
        rs.DecisionEvent(i, "stage3_risk_accept_execute", T0 + timedelta(seconds=i), "apple", "scratch", 5, 0, "processing", None)
        for i in range(8)
    ]
    set_rows = rs.replay(events, items_per_set=100, ask_duration_sec=5, execute_interval_sec=1)
    dynamic = set_rows[0].dynamic
    assert dynamic.query_count == 1
    assert dynamic.total_loss == pytest.approx(0)


def test_dynamic_evpi_avoids_loss_that_fixed_threshold_incurs(isolated_db):
    """고정 threshold가 0.667 confidence만 믿고 너무 일찍 자동 실행으로 전환한
    뒤, 실제로는 "ugly"인 사과를 "normal_box"로 잘못 보내 손실을 발생시키는 반면
    (성급한 확신의 대가), 동적 EVPI는 혼잡이 없는 상황에서 계속 확인 질문을
    하다가 그 사건에서도 사람에게 물어 손실을 피한다."""
    events = [
        rs.DecisionEvent(0, "stage3_risk_accept_execute", T0, "apple", "small", 0, 0, "normal", None),
        rs.DecisionEvent(1, "stage3_risk_accept_execute", T0 + timedelta(seconds=1), "apple", "small", 0, 0, "normal", None),
        rs.DecisionEvent(2, "stage3_risk_accept_execute", T0 + timedelta(seconds=2), "apple", "small", 0, 0, "ugly", None),
    ]
    set_rows = rs.replay(events, items_per_set=100, ask_duration_sec=5, execute_interval_sec=1)
    fixed = set_rows[0].fixed
    dynamic = set_rows[0].dynamic

    assert fixed.query_count == 1
    assert fixed.total_loss > 0
    assert dynamic.total_loss == pytest.approx(0)
    assert dynamic.total_loss < fixed.total_loss


def test_unlabeled_events_excluded_from_loss_but_counted(isolated_db):
    events = [
        rs.DecisionEvent(0, "stage3_risk_accept_execute", T0, "apple", "small", 0, 0, None, None),
    ]
    set_rows = rs.replay(events, items_per_set=100, ask_duration_sec=5, execute_interval_sec=1)
    fixed = set_rows[0].fixed
    assert fixed.total_loss == pytest.approx(0)
    assert fixed.n_unlabeled == 1


def test_replay_chunks_events_into_sets_of_items_per_set(isolated_db):
    events = [
        rs.DecisionEvent(i, "stage3_risk_accept_execute", T0 + timedelta(seconds=i), "apple", "normal", 0, 0, "normal", None)
        for i in range(14)
    ]
    set_rows = rs.replay(events, items_per_set=6, ask_duration_sec=5, execute_interval_sec=1)
    assert len(set_rows) == 3
    assert set_rows[0].fixed.n_events == 6
    assert set_rows[1].fixed.n_events == 6
    assert set_rows[2].fixed.n_events == 2


def _five_sets_of_events():
    """§5.1 "총 5세트"와 동일한 N=5 세트를 만드는 합성 로그 (세트당 6개)."""
    events = []
    for set_idx in range(5):
        for item_idx in range(6):
            i = set_idx * 6 + item_idx
            events.append(
                rs.DecisionEvent(i, "stage3_risk_accept_execute", T0 + timedelta(seconds=i), "apple", "normal", 0, 0, "normal", None)
            )
    return events


def test_generate_report_contains_five_set_rows_and_n5_low_power_caveat(isolated_db):
    events = _five_sets_of_events()
    set_rows = rs.replay(events, items_per_set=6, ask_duration_sec=5, execute_interval_sec=1)
    assert len(set_rows) == 5

    report = rs.generate_report(set_rows)
    assert "세트 수(N) = 5" in report
    assert "N=5의 통계적 함의" in report
    assert "저검정력" in report
    assert "관찰된 경향" in report
    assert "Wilcoxon signed-rank test" in report
    assert "총 손실 sum(L(y,d))" in report
    assert "세트 처리 총 소요 시간" in report
    assert "Stage3(사람 호출) 도달 횟수" in report


def test_try_wilcoxon_handles_identical_values_without_crashing():
    message = rs._try_wilcoxon([1, 1, 1], [1, 1, 1])
    assert "계산 불가" in message or "동일" in message


def test_try_wilcoxon_handles_mismatched_lengths():
    message = rs._try_wilcoxon([1, 2], [1])
    assert "표본 부족" in message


def test_try_wilcoxon_reports_statistic_and_p_value_with_caveat():
    fixed_vals = [10, 12, 8, 15, 9]
    dynamic_vals = [2, 3, 1, 4, 2]
    message = rs._try_wilcoxon(fixed_vals, dynamic_vals)
    assert "statistic=" in message
    assert "p=" in message
    assert "N=5" in message
    assert "과도한" in message


def test_full_pipeline_end_to_end_with_synthetic_db(isolated_db):
    """load_decision_events() -> replay() -> generate_report() 전체 파이프라인이
    실제 tb_decision_audit(임시 DB)에 삽입한 합성 로그로 에러 없이 동작하는지 확인."""
    for set_idx in range(5):
        for item_idx in range(6):
            i = set_idx * 6 + item_idx
            _insert_row(
                "stage3_risk_accept_execute", T0 + timedelta(seconds=i), "apple", "normal",
                n_pending=0, tau_hold_sec=0, actual_label="normal",
            )

    events = rs.load_decision_events()
    assert len(events) == 30

    set_rows = rs.replay(events, items_per_set=6)
    assert len(set_rows) == 5

    report = rs.generate_report(set_rows)
    assert "세트 수(N) = 5" in report
