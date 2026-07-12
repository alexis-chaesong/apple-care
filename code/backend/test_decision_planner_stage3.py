# Track 2/3: services/decision_planner.py Stage3 통합 테스트
# Track 6: decide()가 async로 바뀌면서 asyncio.run()으로 호출하도록 갱신

"""
test_decision_planner_stage3.py
==================================
services/decision_planner.py의 Stage3 통합(§4.3.5 EVPI 게이트로 완전히
대체된 decide())에 대한 end-to-end pytest.

단위 테스트(test_human_query_gate.py)가 게이트 함수 자체를 검증한다면, 이
파일은 decide()가 실제 tb_policy_memory 기본 시드 데이터와 결합했을 때
execute/ask_human/risk_accept_execute가 기대대로 나오는지 확인한다.
실제 data/robot_system.db는 건드리지 않도록 tmp_path 임시 SQLite 파일 사용.

decide()는 Track6부터 async다(condition="unknown" 경로가 실제 GPT-4o Vision
호출을 할 수 있어서). 이 파일의 테스트는 unknown 케이스에서 이미지 캡처
스텁(get_captured_frame_for_vlm)이 항상 None을 반환하는 현재 상태를 그대로
쓰므로 실제 OpenAI 호출은 발생하지 않는다 (VLM 관련 시나리오는
test_vlm_gate.py/test_unknown_object_vlm_integration.py 참고).

사용법:
    cd code/backend && pytest test_decision_planner_stage3.py -v
"""

import asyncio

import pytest

import database
from models import VisionFeatureIn
from services.bayesian_policy import record_human_feedback
from services.decision_planner import decide
from services.vlm_gate import UNIDENTIFIED_OBJECT_FRUIT_TYPE


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "test_stage3.db"))
    database.init_db()
    yield


def _vision(**overrides) -> VisionFeatureIn:
    defaults = dict(fruit_type="apple", confidence=0.95, unknown_flag=False, center=[1, 2, 3])
    defaults.update(overrides)
    return VisionFeatureIn(**defaults)


def _decide(*args, **kwargs):
    """decide()는 async - 동기 pytest 함수에서 asyncio.run()으로 감싸서 호출."""
    return asyncio.run(decide(*args, **kwargs))


def test_default_normal_policy_executes_without_asking(isolated_db):
    """시드 데이터: ("apple","normal","normal_box","llm_policy",confidence=1.0).
    p=1.0 -> EVPI=0 -> 혼잡이 전혀 없어도(0 > 0 = False) query_human=False -> execute."""
    result = _decide(_vision(size="normal"), n_t=0, tau_hold_t=0)
    assert result.action == "execute"
    assert result.destination == "normal_box"
    assert result.reason == "risk_accept_execute"
    assert result.condition == "normal"


def test_default_mold_policy_executes_to_discard(isolated_db):
    """시드 데이터: ("apple","mold","discard_box","llm_policy",confidence=1.0).
    안전 계열(mold)이라 L_error=1000이지만, p=1.0 -> EVPI=0이라 여전히 execute."""
    result = _decide(_vision(defect_type="mold"), n_t=0, tau_hold_t=0)
    assert result.action == "execute"
    assert result.destination == "discard_box"


def test_unknown_with_no_vlm_image_falls_back_to_unidentified_and_asks_human(isolated_db):
    """Track6: 이미지 캡처 스텁이 항상 None을 반환하는 현재 상태에서는 VLM을
    아예 호출하지 않고, fruit_type이 UNIDENTIFIED_OBJECT_FRUIT_TYPE로 폴백된다.
    그 조합엔 정책이 없으므로(theta_f,c=∅) 무조건 ask_human.

    주의: vision.fruit_type="apple"을 줘도(unknown_flag=True로만 unknown
    경로에 들어감), Stage3 게이트가 실제로 조회하는 fruit_type은 vision.fruit_type이
    아니라 VLM 식별 결과(지금은 항상 실패)의 폴백 라벨이다 - 그래서
    ("apple","unknown","ask_human",...) 시드 데이터는 이 경로에서 더 이상 쓰이지
    않는다 (아래 test_unknown_dstored_ask_human... 테스트가 그 대체 시나리오).
    """
    result = _decide(_vision(unknown_flag=True), n_t=0, tau_hold_t=0)
    assert result.action == "ask_human"
    assert result.destination is None
    assert result.condition == "unknown"
    assert result.fruit_type == UNIDENTIFIED_OBJECT_FRUIT_TYPE


def test_unknown_dstored_ask_human_forces_query_for_unidentified_bucket(isolated_db):
    """UNIDENTIFIED_OBJECT_FRUIT_TYPE/"unknown" 조합에 dstored="ask_human" 정책이
    쌓여 있으면(예: 관리자가 "식별 안 되는 건 항상 물어봐"로 명시 등록),
    confidence가 아무리 높아도 무조건 ask_human이어야 한다 (§4.3.5)."""
    record_human_feedback(
        fruit_type=UNIDENTIFIED_OBJECT_FRUIT_TYPE, condition="unknown",
        destination="ask_human", raw_answer="모르겠으면 항상 물어봐",
    )
    result = _decide(_vision(unknown_flag=True), n_t=5, tau_hold_t=100)
    assert result.action == "ask_human"
    assert result.fruit_type == UNIDENTIFIED_OBJECT_FRUIT_TYPE


def test_brand_new_fruit_with_no_policy_asks_human(isolated_db):
    """정책이 아예 없는 신규 조합 -> theta_f_c=∅ -> 항상 ask_human.

    fruit_type="apple"은 시드 데이터의 ("apple","normal",...) 기본 정책이
    항상 존재해 _find_matching_policy가 "normal"로 폴백 매칭되므로, 정말로
    정책이 하나도 없는 상태를 재현하려면 시드 데이터에 없는 fruit_type(kiwi)을
    써야 한다 - "bruise"도 "normal"도 이 fruit_type엔 정책이 전혀 없음.
    """
    result = _decide(_vision(fruit_type="kiwi", defect_type="bruise"), n_t=0, tau_hold_t=0)
    assert result.action == "ask_human"
    assert result.condition == "bruise"


def test_low_vision_confidence_still_asks_human_before_stage3(isolated_db):
    """Stage1 실패(vision.confidence < confidence_threshold)는 Stage3 EVPI 게이트보다
    우선하는 하드 세이프티 규칙이라, 정책 신뢰도와 무관하게 항상 ask_human."""
    result = _decide(_vision(size="normal", confidence=0.1), n_t=0, tau_hold_t=0)
    assert result.action == "ask_human"
    assert result.reason == "low_confidence"


def test_learned_low_confidence_policy_asks_human_at_zero_congestion(isolated_db):
    """사람 피드백으로 학습되었지만 아직 신뢰도가 낮은 정책(alpha=beta=1, p=0.5)은
    혼잡이 없는 상태(cost=0)에서는 EVPI>0이라 항상 ask_human이어야 한다."""
    record_human_feedback(
        fruit_type="apple", condition="scratch", destination="processing_box", raw_answer="가공용으로 보내"
    )
    result = _decide(_vision(defect_type="scratch"), n_t=0, tau_hold_t=0)
    assert result.action == "ask_human"


def test_learned_low_confidence_policy_risk_accepts_execute_under_congestion(isolated_db):
    """같은 학습 정책이라도, 대기열이 혼잡해지면(n_t=5) Cost_human(t)이 급등해
    risk_accept_execute로 전환되어야 한다 (congestion-aware 핵심 동작)."""
    record_human_feedback(
        fruit_type="apple", condition="scratch", destination="processing_box", raw_answer="가공용으로 보내"
    )
    result = _decide(_vision(defect_type="scratch"), n_t=5, tau_hold_t=0)
    assert result.action == "execute"
    assert result.destination == "processing_box"
    assert result.reason == "risk_accept_execute"
