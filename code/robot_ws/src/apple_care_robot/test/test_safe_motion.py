import sys
import os
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from apple_care_robot.safe_motion import (
    EmergencyStopError, raise_if_emergency_stop, safe_movel, safe_movej,
)


class _FakeLogger:
    def __init__(self):
        self.messages = []

    def info(self, msg):
        self.messages.append(("info", msg))

    def warning(self, msg):
        self.messages.append(("warning", msg))

    def error(self, msg):
        self.messages.append(("error", msg))


class _FakeNode:
    def __init__(self):
        self._logger = _FakeLogger()

    def get_logger(self):
        return self._logger


# ── raise_if_emergency_stop: 감지 + 정지 요청 + 예외 발생 ──────────────────

def test_raise_if_emergency_stop_does_nothing_when_not_set():
    node = _FakeNode()
    event = threading.Event()  # 걸안됨
    stop_calls = []

    raise_if_emergency_stop(node, event, stop_motion_fn=lambda n, mode: stop_calls.append(mode))

    assert stop_calls == []


def test_raise_if_emergency_stop_stops_and_raises_when_set():
    node = _FakeNode()
    event = threading.Event()
    event.set()
    stop_calls = []

    try:
        raise_if_emergency_stop(node, event, stop_motion_fn=lambda n, mode: stop_calls.append(mode))
        assert False, "EmergencyStopError가 발생했어야 함"
    except EmergencyStopError:
        pass

    assert len(stop_calls) == 1  # 실제로 정지 요청을 보냈는지 확인


# ── safe_movel / safe_movej: 비동기 이동 + 폴링 중 비상정지 감시 ──────────────

def test_safe_movel_starts_amovel_with_target_and_kwargs():
    node = _FakeNode()
    event = threading.Event()
    calls = []

    safe_movel(
        node, "TARGET_POS", event,
        amovel=lambda pos, vel=None, acc=None, ref=None: calls.append(("amovel", pos, vel, acc, ref)),
        check_motion=lambda: False,  # 시작하자마자 이동 완료로 처리
        wait=lambda sec: calls.append(("wait", sec)),
        vel=30, acc=20, ref="BASE",
    )

    # amovel 호출 직후 폴링 시작 전 짧게 한 번 더 대기함 (check_motion()이
    # "이동 중" 상태로 갱신되기 전에 잘못 "정지"로 읽히는 걸 방지 - safe_motion.py
    # 참고). check_motion=False라 폴링 루프 자체는 안 돌고 이 초기 대기만 찍힘.
    assert calls == [("amovel", "TARGET_POS", 30, 20, "BASE"), ("wait", 0.1)]


def test_safe_movel_completes_normally_when_motion_finishes():
    node = _FakeNode()
    event = threading.Event()
    # check_motion: 처음 두 번은 "이동 중"(True), 세 번째부터 "완료"(False)
    motion_states = [True, True, False]

    safe_movel(
        node, "TARGET_POS", event,
        amovel=lambda pos, vel=None, acc=None, ref=None: None,
        check_motion=lambda: motion_states.pop(0),
        wait=lambda sec: None,
    )
    # 예외 없이 정상 종료되면 성공


def test_safe_movel_raises_and_stops_motion_when_emergency_stop_set():
    node = _FakeNode()
    event = threading.Event()
    event.set()  # 이동 시작 시점부터 이미 걸려있는 상황
    stop_calls = []

    try:
        safe_movel(
            node, "TARGET_POS", event,
            amovel=lambda pos, vel=None, acc=None, ref=None: None,
            check_motion=lambda: True,  # 계속 이동 중이라고 응답 (멈추라고 치면 됨)
            wait=lambda sec: None,
            stop_motion_fn=lambda n, mode: stop_calls.append(mode),
        )
        assert False, "EmergencyStopError가 발생했어야 함"
    except EmergencyStopError:
        pass

    assert len(stop_calls) == 1


def test_safe_movej_starts_amovej_and_supports_emergency_stop():
    node = _FakeNode()
    event = threading.Event()
    calls = []

    safe_movej(
        node, "TARGET_POSJ", event,
        amovej=lambda pos, vel=None, acc=None: calls.append(("amovej", pos, vel, acc)),
        check_motion=lambda: False,
        wait=lambda sec: None,
        vel=30, acc=30,
    )

    assert calls == [("amovej", "TARGET_POSJ", 30, 30)]
