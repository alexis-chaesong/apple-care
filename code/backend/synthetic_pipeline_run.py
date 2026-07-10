# Track 통합 드라이런: synthetic Vision observation -> 실제 프로덕션 코드 경로 end-to-end 검증

"""
synthetic_pipeline_run.py
============================
실제 사과 데이터/실제 Vision·Robot 하드웨어 없이, mock VisionFeatureIn 관측치를
실제 프로덕션 코드 경로(services.decision_planner.decide(), 그 내부의
services.human_query_gate.should_query_human(), services.bayesian_policy 모듈,
services.decision_audit 로깅 함수)에 그대로 흘려서 end-to-end 동작을 확인하는
통합 드라이런 스크립트.

0단계에서 확인한 실제 Vision 응답 포맷(vision_bridge.py의
STATUS_TO_FRUIT_DEFECT/STATUS_TO_SIZE 매핑 결과)을 기준으로 VisionFeatureIn을
직접 구성한다 - HTTP나 ROS 계층은 건드리지 않고, decide()를 순수 함수처럼
바로 호출한다(main.py의 _vla_consumer_loop가 하는 일과 동일한 호출 형태).

5개 필수 케이스 + Track6 unknown 케이스 1개:
    1. 확실한 정상과일           - 정책 신뢰(confidence=1.0) -> execute
    2. 낮은 confidence + 높은 신뢰 정책 - Stage1 하드 세이프티가 Stage3보다 우선 -> ask_human
    3. 완전 신규 조합             - theta_f,c=∅ -> ask_human
    4. safety condition          - L_error=1000이 지배적이라 웬만한 신뢰로는 안 뚫림 -> ask_human
    5. congestion 효과            - 동일 정책, n_t만 바꿔 risk-accept 전환 확인
    6. (Track6) unknown 물체      - 이미지 캡처 인터페이스가 아직 vision_ws와 연결되지
       않은 현재 프로덕션 상태 그대로(get_captured_frame_for_vlm이 항상 None) 실행 -
       VLM 호출은 자연히 스킵되고 UNIDENTIFIED_OBJECT_FRUIT_TYPE로 폴백해 ask_human.
       (실제 GPT-4o 호출이 섞인 시나리오는 test_unknown_object_vlm_integration.py에서
       OpenAI 클라이언트만 patch해서 별도로 검증함 - 이 드라이런은 API 키 유무와
       무관하게 항상 재현 가능해야 하므로 여기서는 실제 호출을 유도하지 않는다.)

마지막으로 이번 드라이런이 tb_decision_audit(임시 DB)에 남긴 로그를 그대로
replay_session.load_decision_events() -> replay() -> generate_report()에 흘려
CLI 파이프라인 전체가 실제로 이어지는지 확인한다.

decide()는 Track6부터 async다(unknown 경로가 실제 GPT-4o Vision 호출을 할 수
있어서) - 이 스크립트의 run_cases()도 그래서 async고, main()이 asyncio.run()으로
구동한다.

사용법:
    cd code/backend && python3 synthetic_pipeline_run.py [--db-path PATH]
"""

import argparse
import asyncio

import database
import replay_session
from models import VisionFeatureIn
from services.bayesian_policy import record_human_feedback
from services.decision_audit import BRANCH_STAGE2_VLM_CALL, get_audit_log_by_id, log_stage2_vlm_call
from services.decision_planner import decide
from services.vlm_gate import UNIDENTIFIED_OBJECT_FRUIT_TYPE


def _vision(**overrides) -> VisionFeatureIn:
    """0단계에서 확인한 vision_bridge.py의 실제 매핑 결과를 기준으로 한 기본값.
    (status=apple_normal -> fruit_type=apple, defect_type=None, confidence는
    서비스 응답 그대로 0~1 스케일, position=center)"""
    defaults = dict(fruit_type="apple", confidence=0.95, unknown_flag=False, center=[100.0, 200.0, 300.0])
    defaults.update(overrides)
    return VisionFeatureIn(**defaults)


def _seed_learned_policy(fruit_type: str, condition: str, destination: str, repeats: int) -> None:
    """record_human_feedback()(실제 프로덕션 함수)을 반복 호출해 alpha를 쌓는다 -
    raw SQL로 alpha/beta를 직접 심지 않고, 실제 학습 경로를 그대로 재현한다."""
    for _ in range(repeats):
        record_human_feedback(fruit_type=fruit_type, condition=condition, destination=destination, raw_answer="테스트 피드백")


def _print_result(case_name: str, result) -> None:
    print(f"[{case_name}] action={result.action} destination={result.destination} "
          f"reason={result.reason} condition={result.condition}")


async def run_cases() -> None:
    print("=" * 90)
    print("Case 1 — 확실한 정상과일 (시드 정책: apple/normal -> normal_box, confidence=1.0)")
    result1 = await decide(_vision(size="normal"), n_t=0, tau_hold_t=0)
    _print_result("Case1", result1)
    assert result1.action == "execute"
    assert result1.destination == "normal_box"
    assert result1.reason == "risk_accept_execute"

    print("=" * 90)
    print("Case 2 — 낮은 confidence + 높은 신뢰 정책 (Stage1이 Stage3보다 우선하는 하드 세이프티)")
    _seed_learned_policy("apple", "scratch", "processing_box", repeats=2)  # confidence=0.75(>구 하드코딩 임계값 0.65)
    result2 = await decide(_vision(defect_type="scratch", confidence=0.2), n_t=0, tau_hold_t=0)
    _print_result("Case2", result2)
    assert result2.action == "ask_human"
    assert result2.reason == "low_confidence"

    print("=" * 90)
    print("Case 3 — 완전 신규 조합 (fruit_type=durian, 정책 자체가 없음)")
    result3 = await decide(_vision(fruit_type="durian", size="normal"), n_t=0, tau_hold_t=0)
    _print_result("Case3", result3)
    assert result3.action == "ask_human"
    assert result3.condition == "normal"

    print("=" * 90)
    print("Case 4 — safety condition (kiwi/mold, 학습된 confidence=0.75여도 L_error=1000이 지배)")
    _seed_learned_policy("kiwi", "mold", "discard_box", repeats=2)  # confidence=0.75
    result4 = await decide(_vision(fruit_type="kiwi", defect_type="mold"), n_t=0, tau_hold_t=0)
    _print_result("Case4", result4)
    assert result4.action == "ask_human"
    assert result4.condition == "mold"

    print("=" * 90)
    print("Case 5 — congestion 효과 (동일 학습 정책, n_t만 0 -> 5로 변경)")
    result5_idle = await decide(_vision(defect_type="scratch"), n_t=0, tau_hold_t=0)
    result5_congested = await decide(_vision(defect_type="scratch"), n_t=5, tau_hold_t=0)
    _print_result("Case5-idle", result5_idle)
    _print_result("Case5-congested", result5_congested)
    assert result5_idle.action == "ask_human"
    assert result5_congested.action == "execute"
    assert result5_congested.reason == "risk_accept_execute"

    print("=" * 90)
    print("Case 6 (Track6) — unknown 물체, 이미지 캡처 인터페이스 미연결 상태 그대로")
    result6 = await decide(
        _vision(fruit_type="unknown", unknown_flag=True, confidence=0.0), n_t=0, tau_hold_t=0
    )
    _print_result("Case6", result6)
    assert result6.action == "ask_human"
    assert result6.condition == "unknown"
    assert result6.fruit_type == UNIDENTIFIED_OBJECT_FRUIT_TYPE, (
        "get_captured_frame_for_vlm()이 아직 스텁(None)이므로 VLM 호출 없이 "
        "폴백 라벨로 떨어져야 함 - vision_ws 이미지 캡처 인터페이스 연결 후 "
        "재검증 필요"
    )

    print("=" * 90)


def verify_stage2_never_updates_posterior() -> None:
    """§4.4: "VLM 응답은 절대 theta_f,c(posterior)를 직접 갱신하지 않는다."

    log_stage2_vlm_call()은 tb_decision_audit에 기록만 할 뿐, tb_policy_memory를
    전혀 건드리지 않는다는 것을 실제로 호출해서 확인한다 (Stage2 실 호출부는
    아직 코드에 없지만, 감사 로그 함수 자체가 posterior에 부작용이 없어야 함).
    """
    conn = database.get_db_connection()
    try:
        before = conn.execute("SELECT * FROM tb_policy_memory ORDER BY fruit_type, condition").fetchall()
        before_snapshot = [dict(row) for row in before]
    finally:
        conn.close()

    vision = _vision(defect_type="bruise")
    audit_id = log_stage2_vlm_call(
        vision, condition="bruise", stage2_evsi_gate_passed=True,
        vlm_response={"guess": "processing_box"}, final_action="hold",
    )
    audit_row = get_audit_log_by_id(audit_id)
    assert audit_row["branch"] == BRANCH_STAGE2_VLM_CALL
    assert audit_row["vlm_called"] is True

    conn = database.get_db_connection()
    try:
        after = conn.execute("SELECT * FROM tb_policy_memory ORDER BY fruit_type, condition").fetchall()
        after_snapshot = [dict(row) for row in after]
    finally:
        conn.close()

    assert before_snapshot == after_snapshot, (
        "Stage2(VLM) 감사 로그 호출이 tb_policy_memory(posterior)를 변경했습니다 - §4.4 위반"
    )
    print("[Stage2 순수성 확인] log_stage2_vlm_call() 호출 후에도 tb_policy_memory 변화 없음 - OK")


def run_replay_on_synthetic_log(items_per_set: int = 6) -> str:
    """이번 드라이런이 남긴 tb_decision_audit 로그를 그대로 replay_session에 흘려
    CLI 파이프라인(load_decision_events -> replay -> generate_report)이 실제로
    이어지는지 확인한다."""
    events = replay_session.load_decision_events()
    if not events:
        return "재생할 이벤트가 없습니다 (drai run이 감사 로그를 남기지 않음)."
    set_rows = replay_session.replay(events, items_per_set=items_per_set)
    return replay_session.generate_report(set_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="synthetic Vision observation을 실제 decide()/should_query_human()/decision_audit 경로에 흘리는 통합 드라이런")
    parser.add_argument("--db-path", default=None, help="격리된 SQLite 파일 경로 (기본: 이 프로세스 안에서만 쓰이는 임시 파일)")
    args = parser.parse_args()

    if args.db_path:
        database.DB_PATH = args.db_path
    else:
        import tempfile
        database.DB_PATH = tempfile.mktemp(suffix=".db", prefix="synthetic_pipeline_")
    print(f"격리된 DB 사용: {database.DB_PATH}")
    database.init_db()

    asyncio.run(run_cases())
    verify_stage2_never_updates_posterior()

    print("=" * 90)
    print("synthetic 로그를 replay_session.py CLI 파이프라인에 그대로 흘려 확인:")
    print(run_replay_on_synthetic_log())


if __name__ == "__main__":
    main()
