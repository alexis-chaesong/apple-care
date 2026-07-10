# Track 4: services/decision_audit.py 단위 테스트

"""
test_decision_audit.py
========================
services/decision_audit.py (tb_decision_audit, 리서치 문서 §8 Track 4)에 대한
pytest 단위 테스트.

스키마대로 1건이 기록되는지, 그리고 append-only가 깨지지 않는지(기존 행이
덮어써지지 않는지)를 검증한다. 실제 데이터(data/robot_system.db)를 건드리지
않도록 매 테스트마다 tmp_path 아래 임시 SQLite 파일로 database.DB_PATH를
monkeypatch한다.

사용법:
    cd code/backend && pytest test_decision_audit.py -v
"""

import pytest

import database
from models import VisionFeatureIn
from services.decision_audit import (
    _insert_audit_row,
    get_audit_log_by_id,
    log_prior_initialized,
    log_stage1_auto_execute,
    log_stage3_human_resolved,
    log_stage3_query_human,
    log_stage3_risk_accept_execute,
)


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """실제 data/robot_system.db 대신 테스트 전용 임시 SQLite 파일 사용.

    decision_audit.py는 `from database import get_db_connection`으로 함수 객체를
    가져다 쓰지만, 그 함수 본문은 매 호출 시 database 모듈의 전역 DB_PATH를
    조회하므로 monkeypatch.setattr(database, "DB_PATH", ...)만으로 충분히 격리된다
    (monkeypatch가 테스트 종료 시 자동으로 원래 값을 복원함).
    """
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "test_decision_audit.db"))
    database.init_db()
    yield


def _make_vision(**overrides) -> VisionFeatureIn:
    defaults = dict(
        fruit_type="apple",
        size="small",
        defect_type=None,
        confidence=0.95,
        unknown_flag=False,
        center=[100, 200, 300],
    )
    defaults.update(overrides)
    return VisionFeatureIn(**defaults)


def test_stage1_auto_execute_writes_row_matching_schema(isolated_db):
    vision = _make_vision(confidence=0.95, defect_type=None)
    audit_id = log_stage1_auto_execute(vision, condition="normal", dstored="normal_box")
    row = get_audit_log_by_id(audit_id)

    assert row is not None
    assert row["branch"] == "stage1_auto_execute"
    assert row["fruit_type"] == "apple"
    assert row["condition"] == "normal"
    assert row["c_yolo"] == pytest.approx(0.95)
    assert row["position"] == [100, 200, 300]
    assert row["tau_yolo"] == pytest.approx(0.9)
    assert row["stage1_gate_passed"] is True
    assert row["theta_exists"] is True
    assert row["dstored"] == "normal_box"
    assert row["query_human"] is False
    assert row["final_action"] == "execute"
    assert row["final_destination"] == "normal_box"
    assert row["alpha_before"] is None
    assert row["beta_before"] is None
    assert row["alpha_after"] is None
    assert row["beta_after"] is None
    assert row["actual_label"] is None
    assert row["timestamp"] is not None


def test_stage3_query_human_writes_row_with_null_posterior(isolated_db):
    vision = _make_vision(confidence=0.3, defect_type="scratch")
    audit_id = log_stage3_query_human(
        vision, condition="scratch", theta_exists=False, dstored=None, session_id="sess-001"
    )
    row = get_audit_log_by_id(audit_id)

    assert row["branch"] == "stage3_query_human"
    assert row["theta_exists"] is False
    assert row["dstored"] is None
    assert row["query_human"] is True
    assert row["final_action"] == "hold"
    assert row["final_destination"] is None
    assert row["session_id"] == "sess-001"
    assert row["evpi_human"] is None
    assert row["cost_human"] is None
    assert row["alpha_before"] is None
    assert row["beta_after"] is None


def test_stage3_query_human_records_evpi_and_cost_when_provided(isolated_db):
    vision = _make_vision(confidence=0.8, defect_type="bruise")
    audit_id = log_stage3_query_human(
        vision,
        condition="bruise",
        theta_exists=True,
        dstored="processing_box",
        evpi_human=4.2,
        cost_human=1.1,
        n_pending=2,
        tau_hold_sec=3.5,
    )
    row = get_audit_log_by_id(audit_id)

    assert row["evpi_human"] == pytest.approx(4.2)
    assert row["cost_human"] == pytest.approx(1.1)
    assert row["n_pending"] == 2
    assert row["tau_hold_sec"] == pytest.approx(3.5)


def test_stage3_risk_accept_execute_writes_execute_row(isolated_db):
    vision = _make_vision(confidence=0.85, defect_type="bruise")
    audit_id = log_stage3_risk_accept_execute(
        vision, condition="bruise", dstored="processing_box", evpi_human=0.5, cost_human=2.0
    )
    row = get_audit_log_by_id(audit_id)

    assert row["branch"] == "stage3_risk_accept_execute"
    assert row["theta_exists"] is True
    assert row["query_human"] is False
    assert row["final_action"] == "execute"
    assert row["final_destination"] == "processing_box"


def test_stage3_human_resolved_fills_posterior_before_and_after(isolated_db):
    audit_id = log_stage3_human_resolved(
        fruit_type="kiwi",
        condition="unknown",
        destination="processing_box",
        alpha_before=1.0,
        beta_before=1.0,
        alpha_after=2.0,
        beta_after=1.0,
        session_id="sess-002",
    )
    row = get_audit_log_by_id(audit_id)

    assert row["branch"] == "stage3_human_resolved"
    assert row["final_action"] == "execute"
    assert row["final_destination"] == "processing_box"
    assert row["query_human"] is True
    assert row["alpha_before"] == pytest.approx(1.0)
    assert row["beta_before"] == pytest.approx(1.0)
    assert row["alpha_after"] == pytest.approx(2.0)
    assert row["beta_after"] == pytest.approx(1.0)


def test_prior_initialized_records_uniform_or_hierarchical(isolated_db):
    audit_id = log_prior_initialized(
        fruit_type="kiwi",
        condition="scratch",
        alpha=1.0,
        beta=1.0,
        init_method="uniform",
        destination="processing_box",
    )
    row = get_audit_log_by_id(audit_id)

    assert row["branch"] == "prior_initialized"
    assert row["theta_exists"] is False
    assert row["dstored"] == "processing_box"
    assert row["alpha_after"] == pytest.approx(1.0)
    assert row["beta_after"] == pytest.approx(1.0)
    assert row["prior_init_method"] == "uniform"
    assert row["prior_pooled_mean"] is None
    assert row["prior_pooled_sample_size"] is None


def test_unknown_branch_or_final_action_rejected(isolated_db):
    with pytest.raises(ValueError):
        _insert_audit_row(branch="not_a_real_branch", fruit_type="apple", condition="normal", final_action="execute")

    with pytest.raises(ValueError):
        _insert_audit_row(
            branch="stage1_auto_execute", fruit_type="apple", condition="normal", final_action="not_a_real_action"
        )


def test_append_only_never_overwrites_previous_rows(isolated_db):
    """같은 fruit_type/condition/session_id로 반복 기록해도 새 행만 추가되고,
    이전에 기록된 행은 절대 수정되지 않아야 한다 (append-only 보장)."""
    vision = _make_vision(confidence=0.3, defect_type="scratch")

    first_id = log_stage3_query_human(
        vision, condition="scratch", theta_exists=True, dstored="ugly_box", session_id="dup-session"
    )
    second_id = log_stage3_query_human(
        vision, condition="scratch", theta_exists=True, dstored="ugly_box", session_id="dup-session"
    )

    assert first_id != second_id
    assert second_id == first_id + 1

    first_row_again = get_audit_log_by_id(first_id)
    second_row = get_audit_log_by_id(second_id)
    assert first_row_again["dstored"] == "ugly_box"
    assert second_row["dstored"] == "ugly_box"
    assert first_row_again["audit_id"] != second_row["audit_id"]

    conn = database.get_db_connection()
    try:
        count = conn.execute("SELECT COUNT(*) FROM tb_decision_audit").fetchone()[0]
    finally:
        conn.close()
    assert count == 2


def test_module_contains_no_update_or_delete_sql():
    """append-only 보장의 정적 근거: 소스 코드 자체에 tb_decision_audit을 대상으로
    하는 UPDATE/DELETE 구문이 전혀 없어야 한다."""
    import services.decision_audit as decision_audit_module

    source = open(decision_audit_module.__file__, encoding="utf-8").read().upper()
    assert "UPDATE TB_DECISION_AUDIT" not in source
    assert "DELETE FROM TB_DECISION_AUDIT" not in source
