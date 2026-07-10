# Track 6: unknown condition -> VLM 식별 -> Stage3 강제 연결 통합 테스트

"""
test_unknown_object_vlm_integration.py
=========================================
condition="unknown"일 때 Stage2(VLM 식별)와 기존 Stage3 게이트가 어떻게
연결되는지 end-to-end로 검증한다.

이 파일은 services.decision_planner에 바인딩된 get_captured_frame_for_vlm/
call_gpt4o_vlm 참조를 monkeypatch한다(decision_planner.py가
`from services.vlm_gate import ...`로 이름을 직접 가져와 썼으므로,
services.vlm_gate 쪽을 patch해도 decision_planner의 바인딩에는 반영되지 않음 -
반드시 services.decision_planner.<이름>을 patch해야 함). §4.4 원칙에 따라
프로덕션 코드가 실제 OpenAI를 호출하지 않도록, 이 테스트에서는 VLM 관련
함수만 patch하고 나머지(get_policy, should_query_human, decision_audit 로깅,
record_human_feedback)는 전부 실제 코드 경로를 그대로 태운다.

사용법:
    cd code/backend && pytest test_unknown_object_vlm_integration.py -v
"""

import asyncio

import pytest

import database
from models import VisionFeatureIn
from services.bayesian_policy import get_policy, record_human_feedback
from services.decision_planner import decide
from services.vlm_gate import UNIDENTIFIED_OBJECT_FRUIT_TYPE


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "test_unknown_vlm.db"))
    database.init_db()
    yield


def _vision(**overrides) -> VisionFeatureIn:
    defaults = dict(
        fruit_type="unknown", confidence=0.0, unknown_flag=True, center=[10.0, 20.0, 30.0],
        frame_id="frame-001",
    )
    defaults.update(overrides)
    return VisionFeatureIn(**defaults)


def _decide(*args, **kwargs):
    return asyncio.run(decide(*args, **kwargs))


def _patch_vlm_success(monkeypatch, identified_object="망치"):
    """이미지 캡처 성공 + VLM 식별 성공을 흉내낸다.

    get_captured_frame_for_vlm()은 Task1부터 async 함수(실제 capture_frame
    ROS2 서비스를 await로 호출)라, patch로 넣는 대체 함수도 코루틴을
    반환해야 한다 - 동기 lambda를 넣으면 decision_planner.py의
    `await get_captured_frame_for_vlm(...)`가 "coroutine이 아닌 값을
    await할 수 없다"는 TypeError를 던진다.
    """
    async def _fake_get_captured_frame_for_vlm(request_id):
        return b"fake-jpeg-bytes"

    monkeypatch.setattr(
        "services.decision_planner.get_captured_frame_for_vlm",
        _fake_get_captured_frame_for_vlm,
    )

    async def _fake_call_gpt4o_vlm(image_bytes):
        return {"identified_object": identified_object, "raw_response": f'{{"identified_object": "{identified_object}"}}'}

    monkeypatch.setattr("services.decision_planner.call_gpt4o_vlm", _fake_call_gpt4o_vlm)


def _patch_vlm_identification_failure(monkeypatch):
    """이미지는 있지만(캡처 성공) VLM이 식별에 실패한 경우."""
    async def _fake_get_captured_frame_for_vlm(request_id):
        return b"fake-jpeg-bytes"

    monkeypatch.setattr(
        "services.decision_planner.get_captured_frame_for_vlm",
        _fake_get_captured_frame_for_vlm,
    )

    async def _fake_call_gpt4o_vlm(image_bytes):
        return {"identified_object": None, "raw_response": None}

    monkeypatch.setattr("services.decision_planner.call_gpt4o_vlm", _fake_call_gpt4o_vlm)


# ------------------------------------------------------------------
# 1) image 있음/없음 두 경우 모두 -> 강제 query_human=True (첫 등장 기준)
# ------------------------------------------------------------------

def test_unknown_with_vlm_identification_success_asks_human_with_identified_fruit_type(isolated_db, monkeypatch):
    """VLM이 성공적으로 식별했고(첫 등장, 정책 없음) -> theta_f,c=∅라서
    무조건 ask_human. DecisionResult.fruit_type이 식별된 이름으로 바뀌어야
    다음번에 같은 물체가 나왔을 때 policy lookup이 이 이름으로 맞아떨어진다."""
    _patch_vlm_success(monkeypatch, identified_object="망치")

    result = _decide(_vision(), n_t=0, tau_hold_t=0)

    assert result.action == "ask_human"
    assert result.condition == "unknown"
    assert result.fruit_type == "망치"


def test_unknown_without_captured_frame_falls_back_and_still_asks_human(isolated_db, monkeypatch):
    """get_captured_frame_for_vlm()이 None을 반환하면(카메라/ROS2 서비스 미연결
    등으로 이미지 캡처 실패) VLM 호출 자체를 스킵하고도 Stage3 질문까지는
    정상 도달해야 한다."""
    async def _fake_get_captured_frame_for_vlm(request_id):
        return None

    monkeypatch.setattr(
        "services.decision_planner.get_captured_frame_for_vlm",
        _fake_get_captured_frame_for_vlm,
    )
    call_attempted = False

    async def _should_not_be_called(image_bytes):
        nonlocal call_attempted
        call_attempted = True
        return {"identified_object": "should-not-happen", "raw_response": None}

    monkeypatch.setattr("services.decision_planner.call_gpt4o_vlm", _should_not_be_called)

    result = _decide(_vision(), n_t=0, tau_hold_t=0)

    assert call_attempted is False
    assert result.action == "ask_human"
    assert result.condition == "unknown"
    assert result.fruit_type == UNIDENTIFIED_OBJECT_FRUIT_TYPE


def test_unknown_with_vlm_identification_failure_falls_back_and_still_asks_human(isolated_db, monkeypatch):
    """이미지는 캡처됐지만 VLM이 식별에 실패한 경우(타임아웃/에러/모델이
    모르겠다고 답함) - call_gpt4o_vlm이 이미 예외를 삼키고 identified_object=None을
    반환하므로, 호출부는 그 실패를 감지해 미상 라벨로 폴백하고 Stage3 질문까지
    정상 도달해야 한다."""
    _patch_vlm_identification_failure(monkeypatch)

    result = _decide(_vision(), n_t=0, tau_hold_t=0)

    assert result.action == "ask_human"
    assert result.fruit_type == UNIDENTIFIED_OBJECT_FRUIT_TYPE


# ------------------------------------------------------------------
# 2) §4.4 불변조건: VLM 호출 직후 posterior 불변, 사람 확인 후에만 변경
# ------------------------------------------------------------------

def test_vlm_call_does_not_mutate_posterior_only_human_answer_does(isolated_db, monkeypatch):
    """§4.4: "VLM 응답은 절대 theta_f,c(posterior)를 직접 갱신하지 않는다."""
    _patch_vlm_success(monkeypatch, identified_object="망치")

    # decide() 호출(VLM까지 포함) 전후로 tb_policy_memory에 "망치" 관련 행이
    # 생기면 안 됨.
    assert get_policy("망치", "unknown") is None
    result = _decide(_vision(), n_t=0, tau_hold_t=0)
    assert result.action == "ask_human"
    assert get_policy("망치", "unknown") is None, "VLM 호출만으로 posterior가 생기면 안 됨(§4.4 위반)"

    # 사람이 실제로 답변한 뒤에야(record_human_feedback) posterior가 생긴다.
    updated = record_human_feedback(
        fruit_type=result.fruit_type, condition=result.condition,
        destination="discard_box", raw_answer="그냥 버려",
    )
    assert updated["destination"] == "discard_box"
    assert get_policy("망치", "unknown") is not None


def test_human_answer_after_vlm_identification_enables_future_auto_learning(isolated_db, monkeypatch):
    """사람 답변 후 기록된 정책(fruit_type=식별된 이름)이, 같은 물체가 다시
    나타났을 때 정확히 그 이름으로 다시 조회되는지 확인 (per-object 학습이
    실제로 작동하는지의 핵심 근거)."""
    _patch_vlm_success(monkeypatch, identified_object="망치")

    first = _decide(_vision(), n_t=0, tau_hold_t=0)
    assert first.action == "ask_human"
    record_human_feedback(
        fruit_type=first.fruit_type, condition=first.condition,
        destination="discard_box", raw_answer="그냥 버려",
    )

    policy = get_policy("망치", "unknown")
    assert policy is not None
    assert policy["destination"] == "discard_box"
    assert policy["source"] == "human_feedback"


# ------------------------------------------------------------------
# 3) Stage2 감사 로그가 실제로 남는지 (이미지가 있어 VLM을 호출한 경우만)
# ------------------------------------------------------------------

def test_stage2_vlm_call_is_logged_to_decision_audit_when_image_available(isolated_db, monkeypatch):
    _patch_vlm_success(monkeypatch, identified_object="망치")

    _decide(_vision(), n_t=0, tau_hold_t=0)

    conn = database.get_db_connection()
    try:
        row = conn.execute(
            "SELECT * FROM tb_decision_audit WHERE branch = 'stage2_vlm_call'"
        ).fetchone()
    finally:
        conn.close()

    assert row is not None
    assert row["vlm_called"] == 1
    assert "망치" in (row["vlm_response"] or "")
    assert row["final_action"] == "hold"  # 정책 없음 -> query_human=True -> hold


def test_stage2_vlm_call_not_logged_when_frame_unavailable(isolated_db, monkeypatch):
    """이미지 캡처 자체가 안 됐으면(get_captured_frame_for_vlm=None) VLM을
    호출한 적이 없으므로 stage2_vlm_call 로그도 남으면 안 된다."""
    async def _fake_get_captured_frame_for_vlm(request_id):
        return None

    monkeypatch.setattr(
        "services.decision_planner.get_captured_frame_for_vlm",
        _fake_get_captured_frame_for_vlm,
    )

    _decide(_vision(), n_t=0, tau_hold_t=0)

    conn = database.get_db_connection()
    try:
        row = conn.execute(
            "SELECT * FROM tb_decision_audit WHERE branch = 'stage2_vlm_call'"
        ).fetchone()
    finally:
        conn.close()

    assert row is None
