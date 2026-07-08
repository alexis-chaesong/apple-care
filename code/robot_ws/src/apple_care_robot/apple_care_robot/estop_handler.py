"""
estop_handler.py
=================
비상정지(E-STOP) 해제 시 로봇을 어떻게 복구시킬지 결정하고 실행하는 모듈.

핵심 규칙 (사용자와 합의된 정책):
    - 사과를 손에 쥔 채로 멈췄으면(RETURN_TO_ORIGIN) -> HOME 경유 -> 원래 집었던
      위치(apple_origin_pos)로 돌아가서 그리퍼를 열어 내려놓음 -> CAMERA로 이동.
    - 사과를 안 쥔 채로 멈췄으면(RESUME_AT_CAMERA) -> HOME 경유 -> CAMERA로
      바로 이동해서 사이클을 이어서 재개.

EstopRecoveryTracker는 상태 추적/결정 로직만 담당 (하드웨어 의존 없음, 유닛테스트 가능).
execute_recovery()는 실제 이동/그리퍼 함수를 인자로 받아 그 계획을 실행하는 얇은 layer.

호출하는 쪽(box_sequence_test.py 등)이 지켜야 할 규칙:
    - 사과 집기 시도 직전(movel(pick_pos) 이후, grasp 호출 직전)에
      begin_pick_attempt(pick_pos)를 호출할 것 - 파지 성공/실패가 아직 판단되지
      않은 구간(A5)에도 "잡았을 수 있다"고 보수적으로 간주하기 위함.
    - 파지가 확실히 실패로 판단되면 mark_pick_failed()를 호출할 것.
    - 사과를 박스에 성공적으로 내려놓았으면 mark_placed()를 호출할 것.

주의: force_controlled_place() 실행 중(컴플라이언스/힘제어 모드 활성 중) E-STOP이
걸리는 경우, release_force()/release_compliance_ctrl() 호출은 이 모듈의 책임이
아님 - 그 모듈을 열고 닫는 force_place.py 쪽에서 별도로 처리해야 함.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class RecoveryPlan:
    action: str  # "RETURN_TO_ORIGIN" 또는 "RESUME_AT_CAMERA"
    target_pos: Optional[Any]  # RETURN_TO_ORIGIN일 때만 원위치 좌표, 아니면 None


class EstopRecoveryTracker:
    """사과 보유 상태(holding_apple)와 원위치(apple_origin_pos)를 추적."""

    def __init__(self):
        self.holding_apple = False
        self.apple_origin_pos = None

    def begin_pick_attempt(self, pick_pos):
        """집기 시도 직전 호출. 파지 성공/실패가 판단 되기 전에도 보수적으로
        "잡았다"고 간주해서, 이 구간에 E-STOP이 걸려도 원위치 복귀 절차를 타게 함."""
        self.holding_apple = True
        self.apple_origin_pos = pick_pos

    def mark_pick_failed(self):
        """파지 실패가 확실해지면 호출 - 안 잡은 상태로 리셋."""
        self.holding_apple = False
        self.apple_origin_pos = None

    def mark_placed(self):
        """사과를 박스에 성공적으로 내려놓았으면 호출 - 안 잡은 상태로 리셋."""
        self.holding_apple = False
        self.apple_origin_pos = None

    def get_recovery_plan(self) -> RecoveryPlan:
        if self.holding_apple:
            return RecoveryPlan(action="RETURN_TO_ORIGIN", target_pos=self.apple_origin_pos)
        return RecoveryPlan(action="RESUME_AT_CAMERA", target_pos=None)


def execute_recovery(
    node,
    plan: RecoveryPlan,
    *,
    movel: Callable[[Any], None],
    gripper_open: Callable[[], bool],
    home_pos,
    camera_pos,
) -> None:
    """
    plan에 따라 실제 복구 동작을 수행.

    movel/gripper_open은 실제로는 DSR_ROBOT2.movel / openclose.gripper_open이지만,
    여기서는 주입받아 사용하므로 이 함수 자체는 하드웨어 의존 없이 테스트 가능함.
    """
    node.get_logger().info(f"[E-STOP] 복구 시작: {plan.action}")

    movel(home_pos)

    if plan.action == "RETURN_TO_ORIGIN":
        movel(plan.target_pos)
        if not gripper_open():
            node.get_logger().error("[E-STOP] 원위치 복귀 중 그리퍼 오픈 실패 - 하드웨어 상태를 확인하세요.")

    movel(camera_pos)
    node.get_logger().info("[E-STOP] 복구 완료, CAMERA에서 재개")


def check_and_recover(
    node,
    status_bus,
    tracker: EstopRecoveryTracker,
    emergency_stop_event,
    *,
    movel: Callable[[Any], None],
    gripper_open: Callable[[], bool],
    home_pos,
    camera_pos,
) -> bool:
    """
    emergency_stop_event(threading.Event)가 걸려 있으면 복구를 수행하고 True를 반환.
    호출부(apple_sorting_cycle.py의 메인 루프)는 True가 반환되면 지금 처리 중이던
    사과를 중단하고 루프 top(다시 decision 대기)으로 돌아가야 함.

    status_bus는 set_state(state, msg=None)를 제공하는 객체(StatusBus)면 됨.

    복구 후에는 emergency_stop_event를 clear()해서 다음 루프에서 다시 걸리지 않게
    하고, "무조건 CAMERA로 보내서 재개" 정책에 따라 별도의 재가 명령 없이
    바로 다음 decision을 받을 수 있는 상태로 돌린다.
    """
    if not emergency_stop_event.is_set():
        return False

    node.get_logger().error("EMERGENCY_STOP 감지 - 복구 절차를 시작합니다.")
    status_bus.set_state("ERROR", "emergency_stop")

    plan = tracker.get_recovery_plan()
    execute_recovery(
        node, plan, movel=movel, gripper_open=gripper_open,
        home_pos=home_pos, camera_pos=camera_pos,
    )

    if plan.action == "RETURN_TO_ORIGIN":
        # execute_recovery가 이미 원위치에서 그리퍼를 열어 내려놨으므로 이제 안 잡은 상태
        tracker.mark_placed()

    emergency_stop_event.clear()
    status_bus.set_state("MOVING")
    return True
