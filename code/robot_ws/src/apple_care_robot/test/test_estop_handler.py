import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from apple_care_robot.estop_handler import (
    EstopRecoveryTracker, execute_recovery, check_and_recover,
    RECOVERY_LIFT_MM, RECOVERY_SETTLE_SEC, RECOVERY_POLL_SEC,
)

# check_motion=lambda: False(항상 "정지")인 테스트에서, _wait_until_fully_stopped가
# REQUIRED_IDLE_CONFIRMATIONS(3)번 연속 확인 후 빠져나가므로 매 구간마다
# RECOVERY_POLL_SEC 대기가 3번 찍히고 그다음 RECOVERY_SETTLE_SEC이 한 번 찍힘.
_IDLE_WAITS = [("wait", RECOVERY_POLL_SEC)] * 3 + [("wait", RECOVERY_SETTLE_SEC)]


# ── EstopRecoveryTracker: 상태 추적 + 복구 계획 결정 ──────────────────────

def test_initial_state_resumes_at_camera():
    tracker = EstopRecoveryTracker()
    plan = tracker.get_recovery_plan()
    assert plan.action == "RESUME_AT_CAMERA"
    assert plan.target_pos is None


def test_begin_pick_attempt_marks_return_to_origin():
    tracker = EstopRecoveryTracker()
    tracker.begin_pick_attempt(pick_pos="PICK_POS_1")
    plan = tracker.get_recovery_plan()
    assert plan.action == "RETURN_TO_ORIGIN"
    assert plan.target_pos == "PICK_POS_1"


def test_begin_pick_attempt_covers_ambiguous_grasp_window():
    # 파지 성공/실패가 아직 판단되지 않은 구간(A5)에서는 보수적으로 "잡았다"고 간주해야 함
    tracker = EstopRecoveryTracker()
    tracker.begin_pick_attempt(pick_pos="PICK_POS_1")
    plan = tracker.get_recovery_plan()
    assert plan.action == "RETURN_TO_ORIGIN"


def test_mark_pick_failed_resets_to_resume_at_camera():
    tracker = EstopRecoveryTracker()
    tracker.begin_pick_attempt(pick_pos="PICK_POS_1")
    tracker.mark_pick_failed()
    plan = tracker.get_recovery_plan()
    assert plan.action == "RESUME_AT_CAMERA"
    assert plan.target_pos is None


def test_mark_placed_resets_to_resume_at_camera():
    tracker = EstopRecoveryTracker()
    tracker.begin_pick_attempt(pick_pos="PICK_POS_1")
    tracker.mark_placed()
    plan = tracker.get_recovery_plan()
    assert plan.action == "RESUME_AT_CAMERA"
    assert plan.target_pos is None


def test_new_pick_attempt_overwrites_previous_origin():
    tracker = EstopRecoveryTracker()
    tracker.begin_pick_attempt(pick_pos="PICK_POS_1")
    tracker.mark_pick_failed()
    tracker.begin_pick_attempt(pick_pos="PICK_POS_2")
    plan = tracker.get_recovery_plan()
    assert plan.target_pos == "PICK_POS_2"


# ── execute_recovery: 상태별 실제 복구 동작 순서 ──────────────────────────

class _FakeLogger:
    def __init__(self):
        self.messages = []

    def info(self, msg):
        self.messages.append(("info", msg))

    def error(self, msg):
        self.messages.append(("error", msg))


class _FakeNode:
    def __init__(self):
        self._logger = _FakeLogger()

    def get_logger(self):
        return self._logger


# 정지 시점의 "현재 위치" 가짜 값. (x, y, z, rx, ry, rz) 형태로 posx처럼 언패킹 가능.
_CURRENT_POS = (100.0, 200.0, 50.0, 10.0, 20.0, 30.0)


def _fake_posx_factory(x, y, z, rx, ry, rz):
    return (x, y, z, rx, ry, rz)


def _fake_get_current_pos():
    return _CURRENT_POS


def test_execute_recovery_flushes_pending_target_before_lifting():
    # 정지 직전 컨트롤러의 "아직 가는 목표"가 남아있을 수 있으므로, 상승하기 전에
    # 먼저 "지금 있는 그 자리로 가라"는 명령으로 잔여 목표를 덮어써야 함
    # (force_place.py의 amovel(stop_pos, ...) 취소 기법과 동일한 원리).
    node = _FakeNode()
    calls = []
    tracker = EstopRecoveryTracker()  # 정상 상태 (RESUME_AT_CAMERA)
    plan = tracker.get_recovery_plan()

    execute_recovery(
        node,
        plan,
        movel=lambda pos: calls.append(("movel", pos)),
        movej=lambda pos: calls.append(("movej", pos)),
        wait=lambda sec: calls.append(("wait", sec)),
        check_motion=lambda: False,  # 즉시 정지 완료로 응답
        gripper_open=lambda: True,
        home_pos="HOME",
        camera_pos="CAMERA",
        get_current_pos=_fake_get_current_pos,
        posx_factory=_fake_posx_factory,
    )

    flush_pos = _CURRENT_POS  # 현재 위치 그대로 (잔여 목표 취소용)
    assert calls[0] == ("movel", flush_pos)


def test_execute_recovery_lifts_up_from_current_position_after_flush():
    # 비상정지가 걸린 바로 그 자리에서, 벽/테이블과의 충돌을 피하려고
    # 잔여 목표를 지운 뒤 수직으로 최소 15cm(150mm) 들어올려야 함.
    node = _FakeNode()
    calls = []
    tracker = EstopRecoveryTracker()  # 정상 상태 (RESUME_AT_CAMERA)
    plan = tracker.get_recovery_plan()

    execute_recovery(
        node,
        plan,
        movel=lambda pos: calls.append(("movel", pos)),
        movej=lambda pos: calls.append(("movej", pos)),
        wait=lambda sec: calls.append(("wait", sec)),
        check_motion=lambda: False,  # 즉시 정지 완료로 응답
        gripper_open=lambda: True,
        home_pos="HOME",
        camera_pos="CAMERA",
        get_current_pos=_fake_get_current_pos,
        posx_factory=_fake_posx_factory,
    )

    lifted_pos = (100.0, 200.0, 50.0 + RECOVERY_LIFT_MM, 10.0, 20.0, 30.0)  # z만 +RECOVERY_LIFT_MM
    # calls[0]=flush movel, calls[1:5]=flush 구간의 _wait_until_fully_stopped가
    # 남기는 4번의 wait(_IDLE_WAITS) 다음이 상승 이동임.
    assert calls[len(_IDLE_WAITS) + 1] == ("movel", lifted_pos)


def test_execute_recovery_return_to_origin_moves_flush_then_lift_then_home_then_origin_then_camera():
    node = _FakeNode()
    calls = []
    tracker = EstopRecoveryTracker()
    tracker.begin_pick_attempt(pick_pos="PICK_POS_1")
    plan = tracker.get_recovery_plan()

    execute_recovery(
        node,
        plan,
        movel=lambda pos: calls.append(("movel", pos)),
        movej=lambda pos: calls.append(("movej", pos)),
        wait=lambda sec: calls.append(("wait", sec)),
        check_motion=lambda: False,  # 즉시 정지 완료로 응답
        gripper_open=lambda: calls.append(("gripper_open",)) or True,
        home_pos="HOME",
        camera_pos="CAMERA",
        get_current_pos=_fake_get_current_pos,
        posx_factory=_fake_posx_factory,
    )

    flush_pos = _CURRENT_POS
    lifted_pos = (100.0, 200.0, 50.0 + RECOVERY_LIFT_MM, 10.0, 20.0, 30.0)
    assert calls == [
        ("movel", flush_pos),
        *_IDLE_WAITS,
        ("movel", lifted_pos),
        *_IDLE_WAITS,
        ("movej", "HOME"),
        *_IDLE_WAITS,
        ("movel", "PICK_POS_1"),
        *_IDLE_WAITS,
        ("gripper_open",),
        ("movel", "CAMERA"),
        *_IDLE_WAITS,
    ]


def test_execute_recovery_resume_at_camera_skips_origin_and_gripper():
    node = _FakeNode()
    calls = []
    tracker = EstopRecoveryTracker()  # 집지 않은 정상 상태

    plan = tracker.get_recovery_plan()
    execute_recovery(
        node,
        plan,
        movel=lambda pos: calls.append(("movel", pos)),
        movej=lambda pos: calls.append(("movej", pos)),
        wait=lambda sec: calls.append(("wait", sec)),
        check_motion=lambda: False,  # 즉시 정지 완료로 응답
        gripper_open=lambda: calls.append(("gripper_open",)) or True,
        home_pos="HOME",
        camera_pos="CAMERA",
        get_current_pos=_fake_get_current_pos,
        posx_factory=_fake_posx_factory,
    )

    flush_pos = _CURRENT_POS
    lifted_pos = (100.0, 200.0, 50.0 + RECOVERY_LIFT_MM, 10.0, 20.0, 30.0)
    assert calls == [
        ("movel", flush_pos),
        *_IDLE_WAITS,
        ("movel", lifted_pos),
        *_IDLE_WAITS,
        ("movej", "HOME"),
        *_IDLE_WAITS,
        ("movel", "CAMERA"),
        *_IDLE_WAITS,
    ]


def test_execute_recovery_logs_error_when_gripper_open_fails():
    node = _FakeNode()
    tracker = EstopRecoveryTracker()
    tracker.begin_pick_attempt(pick_pos="PICK_POS_1")
    plan = tracker.get_recovery_plan()

    execute_recovery(
        node,
        plan,
        movel=lambda pos: None,
        movej=lambda pos: None,
        wait=lambda sec: None,
        check_motion=lambda: False,
        gripper_open=lambda: False,
        home_pos="HOME",
        camera_pos="CAMERA",
        get_current_pos=_fake_get_current_pos,
        posx_factory=_fake_posx_factory,
    )

    assert any(level == "error" for level, _ in node.get_logger().messages)


# ── check_and_recover: /robot/command EMERGENCY_STOP 감지 + 복구 통합 ──

class _FakeStatusBus:
    def __init__(self):
        self.states = []
        self.gripper_statuses = []
        self.safety_events = []

    def set_state(self, state, msg=None):
        self.states.append((state, msg))

    def publish_gripper_status(self, grasped):
        self.gripper_statuses.append(grasped)

    def publish_safety(self, error_code, error_msg=""):
        self.safety_events.append((error_code, error_msg))


def test_check_and_recover_returns_false_when_not_triggered():
    import threading

    node = _FakeNode()
    status_bus = _FakeStatusBus()
    tracker = EstopRecoveryTracker()
    event = threading.Event()  # 아직 안걸림

    triggered = check_and_recover(
        node, status_bus, tracker, event,
        movel=lambda pos: None,
        movej=lambda pos: None,
        wait=lambda sec: None,
        check_motion=lambda: False,
        gripper_open=lambda: True,
        home_pos="HOME",
        camera_pos="CAMERA",
        get_current_pos=_fake_get_current_pos,
        posx_factory=_fake_posx_factory,
    )

    assert triggered is False
    assert status_bus.states == []


def test_check_and_recover_runs_recovery_and_clears_event_when_triggered():
    import threading

    node = _FakeNode()
    status_bus = _FakeStatusBus()
    tracker = EstopRecoveryTracker()
    tracker.begin_pick_attempt(pick_pos="PICK_POS_1")
    event = threading.Event()
    event.set()
    calls = []

    triggered = check_and_recover(
        node, status_bus, tracker, event,
        movel=lambda pos: calls.append(("movel", pos)),
        movej=lambda pos: calls.append(("movej", pos)),
        wait=lambda sec: calls.append(("wait", sec)),
        check_motion=lambda: False,  # 즉시 정지 완료로 응답
        gripper_open=lambda: calls.append(("gripper_open",)) or True,
        home_pos="HOME",
        camera_pos="CAMERA",
        get_current_pos=_fake_get_current_pos,
        posx_factory=_fake_posx_factory,
    )

    flush_pos = _CURRENT_POS
    lifted_pos = (100.0, 200.0, 50.0 + RECOVERY_LIFT_MM, 10.0, 20.0, 30.0)
    assert triggered is True
    assert event.is_set() is False  # 리셋되어야 다음 루프에서 또 걸림
    assert calls == [
        ("movel", flush_pos),
        *_IDLE_WAITS,
        ("movel", lifted_pos),
        *_IDLE_WAITS,
        ("movej", "HOME"),
        *_IDLE_WAITS,
        ("movel", "PICK_POS_1"),
        *_IDLE_WAITS,
        ("gripper_open",),
        ("movel", "CAMERA"),
        *_IDLE_WAITS,
    ]
    # 원위치에 내려놨으니 이제 정상 상태로 리셋되어야 함
    assert tracker.get_recovery_plan().action == "RESUME_AT_CAMERA"
    assert ("ERROR", "emergency_stop") in status_bus.states
    # 복구 후 로봇은 실제로 CAMERA에서 그리퍼를 연 채 대기 중이므로, 백엔드가
    # 다음 Vision 감지를 다시 판단할 수 있도록 READY로 돌아가야 하고
    # (MOVING으로 남으면 백엔드가 영원히 무시함 - 실제로 겪은 문제),
    # 그리퍼도 다시 열렸음을 알려야 함(gripper_grasped가 안 풀리는 문제 방지).
    assert ("READY", None) in status_bus.states
    assert False in status_bus.gripper_statuses


def test_check_and_recover_blocks_and_keeps_event_set_when_hw_safety_stopped():
    # 펜던트/안전펜스의 물리 비상정지 버튼이 눌려 컨트롤러가 하드웨어
    # 안전정지 상태(예: SAFE_STOP)이면, emergency_stop_event가 걸려 있어도
    # 복구 이동(movel/movej)을 내보내면 안 되고, 이벤트도 clear하면 안 됨 -
    # 그래야 호출부가 다음 decision을 처리하지 않고 계속 재확인하게 됨.
    import threading

    node = _FakeNode()
    status_bus = _FakeStatusBus()
    tracker = EstopRecoveryTracker()
    tracker.begin_pick_attempt(pick_pos="PICK_POS_1")
    event = threading.Event()
    event.set()
    calls = []

    triggered = check_and_recover(
        node, status_bus, tracker, event,
        movel=lambda pos: calls.append(("movel", pos)),
        movej=lambda pos: calls.append(("movej", pos)),
        wait=lambda sec: calls.append(("wait", sec)),
        check_motion=lambda: False,
        gripper_open=lambda: calls.append(("gripper_open",)) or True,
        home_pos="HOME",
        camera_pos="CAMERA",
        get_current_pos=_fake_get_current_pos,
        posx_factory=_fake_posx_factory,
        check_hw_safety_stop=lambda: (True, "SAFE_STOP"),
    )

    assert triggered is True
    assert event.is_set() is True  # 계속 걸린 채로 유지되어야 다음 루프에서 다시 확인함
    # 실제 복구 이동(movel/movej/gripper_open)은 하나도 나가면 안 됨
    assert not any(call[0] in ("movel", "movej", "gripper_open") for call in calls)
    assert any(level == "error" for level, _ in node.get_logger().messages)
    assert ("ERROR", "controller_safety_stop:SAFE_STOP") in status_bus.states
    # 원위치 복귀 계획도 그대로 남아있어야 함 (복구가 아직 시작되지 않았으므로)
    assert tracker.get_recovery_plan().action == "RETURN_TO_ORIGIN"


def test_check_and_recover_proceeds_when_hw_safety_stop_check_reports_clear():
    # 컨트롤러가 안전정지 상태가 아니면(is_blocked=False) 평소처럼 복구가 진행돼야 함.
    import threading

    node = _FakeNode()
    status_bus = _FakeStatusBus()
    tracker = EstopRecoveryTracker()  # 정상 상태 (RESUME_AT_CAMERA)
    event = threading.Event()
    event.set()
    calls = []

    triggered = check_and_recover(
        node, status_bus, tracker, event,
        movel=lambda pos: calls.append(("movel", pos)),
        movej=lambda pos: calls.append(("movej", pos)),
        wait=lambda sec: calls.append(("wait", sec)),
        check_motion=lambda: False,
        gripper_open=lambda: calls.append(("gripper_open",)) or True,
        home_pos="HOME",
        camera_pos="CAMERA",
        get_current_pos=_fake_get_current_pos,
        posx_factory=_fake_posx_factory,
        check_hw_safety_stop=lambda: (False, ""),
    )

    assert triggered is True
    assert event.is_set() is False  # 정상 복구가 끝났으므로 이번엔 clear되어야 함
    assert any(call[0] == "movel" for call in calls)


def test_check_and_recover_when_not_holding_skips_origin_and_resumes_at_camera():
    import threading

    node = _FakeNode()
    status_bus = _FakeStatusBus()
    tracker = EstopRecoveryTracker()  # 정상 상태
    event = threading.Event()
    event.set()
    calls = []

    triggered = check_and_recover(
        node, status_bus, tracker, event,
        movel=lambda pos: calls.append(("movel", pos)),
        movej=lambda pos: calls.append(("movej", pos)),
        wait=lambda sec: calls.append(("wait", sec)),
        check_motion=lambda: False,  # 즉시 정지 완료로 응답
        gripper_open=lambda: calls.append(("gripper_open",)) or True,
        home_pos="HOME",
        camera_pos="CAMERA",
        get_current_pos=_fake_get_current_pos,
        posx_factory=_fake_posx_factory,
    )

    flush_pos = _CURRENT_POS
    lifted_pos = (100.0, 200.0, 50.0 + RECOVERY_LIFT_MM, 10.0, 20.0, 30.0)
    assert triggered is True
    assert calls == [
        ("movel", flush_pos),
        *_IDLE_WAITS,
        ("movel", lifted_pos),
        *_IDLE_WAITS,
        ("movej", "HOME"),
        *_IDLE_WAITS,
        ("movel", "CAMERA"),
        *_IDLE_WAITS,
    ]
    assert ("READY", None) in status_bus.states
    assert False in status_bus.gripper_statuses


# ── check_motion()의 단발성 오독(레이스 컨디션) 회귀 테스트 ──────────────────
# 실제 하드웨어에서 관찰된 문제: amovel 직후 check_motion()이 컨트롤러 내부
# 상태가 아직 갱신되기 전이라 "정지"로 잘못 읽힌 뒤(False), 그다음 폴링에서는
# 제대로 "이동 중"(True)으로 나오는 경우가 있었음. 단순히 첫 확인만 믿으면
# 로봇이 한창 이동 중인데도 "멈췄다"고 착각하게 됨 - REQUIRED_IDLE_CONFIRMATIONS
# 연속 확인 방식이 이런 단발성 오독에 흔들리지 않는지 검증함.

def test_wait_until_fully_stopped_ignores_single_false_reading_from_check_motion():
    from apple_care_robot.estop_handler import _wait_until_fully_stopped

    # 순서: False(오독) -> True -> True -> False -> False -> False(진짜 정지)
    # 첫 False만 믿고 바로 빠져나가면 안 되고, 그 뒤 True가 나오면 연속 카운트가
    # 리셋되어 다시 3번 연속 False가 나올 때까지 기다려야 함.
    responses = iter([False, True, True, False, False, False])
    poll_calls = []

    def fake_check_motion():
        try:
            value = next(responses)
        except StopIteration:
            value = False
        poll_calls.append(value)
        return value

    waits = []
    _wait_until_fully_stopped(
        fake_check_motion, lambda sec: waits.append(sec), settle_sec=1.0,
    )

    # check_motion이 실제로 True를 반환한 뒤에도(2,3번째) 계속 확인했어야 함 -
    # 즉 최소 6번(오독 1 + True 2 + 진짜 정지 확인 3)은 폴링했어야 정상
    assert len(poll_calls) >= 6
    assert poll_calls[-3:] == [False, False, False]
