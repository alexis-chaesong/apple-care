import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from apple_care_robot.estop_handler import (
    EstopRecoveryTracker, execute_recovery, check_and_recover,
    RECOVERY_LIFT_MM, RECOVERY_SETTLE_SEC,
)


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
    assert calls[2] == ("movel", lifted_pos)


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
        ("wait", RECOVERY_SETTLE_SEC),
        ("movel", lifted_pos),
        ("wait", RECOVERY_SETTLE_SEC),
        ("movej", "HOME"),
        ("wait", RECOVERY_SETTLE_SEC),
        ("movel", "PICK_POS_1"),
        ("wait", RECOVERY_SETTLE_SEC),
        ("gripper_open",),
        ("movel", "CAMERA"),
        ("wait", RECOVERY_SETTLE_SEC),
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
        ("wait", RECOVERY_SETTLE_SEC),
        ("movel", lifted_pos),
        ("wait", RECOVERY_SETTLE_SEC),
        ("movej", "HOME"),
        ("wait", RECOVERY_SETTLE_SEC),
        ("movel", "CAMERA"),
        ("wait", RECOVERY_SETTLE_SEC),
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

    def set_state(self, state, msg=None):
        self.states.append((state, msg))

    def publish_gripper_status(self, grasped):
        self.gripper_statuses.append(grasped)


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
        ("wait", RECOVERY_SETTLE_SEC),
        ("movel", lifted_pos),
        ("wait", RECOVERY_SETTLE_SEC),
        ("movej", "HOME"),
        ("wait", RECOVERY_SETTLE_SEC),
        ("movel", "PICK_POS_1"),
        ("wait", RECOVERY_SETTLE_SEC),
        ("gripper_open",),
        ("movel", "CAMERA"),
        ("wait", RECOVERY_SETTLE_SEC),
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
        ("wait", RECOVERY_SETTLE_SEC),
        ("movel", lifted_pos),
        ("wait", RECOVERY_SETTLE_SEC),
        ("movej", "HOME"),
        ("wait", RECOVERY_SETTLE_SEC),
        ("movel", "CAMERA"),
        ("wait", RECOVERY_SETTLE_SEC),
    ]
    assert ("READY", None) in status_bus.states
    assert False in status_bus.gripper_statuses
