"""
estop_handler.py
=================
비상정지(E-STOP) 해제 시 로봇을 어떻게 복구시킬지 결정하고 실행하는 모듈.

핵심 규칙 (사용자와 합의된 정책):
    - 어떤 경우든 가장 먼저, 멈춘 바로 그 자리에서 수직으로 150mm(15cm) 상승
      (RECOVERY_LIFT_MM) - 정지 지점이 벽/테이블에 가까울 수 있어(예: 박스
      웨이포인트 근처) 다른 이동 전에 반드시 먼저 충돌 회피용으로 들어올림.
    - 사과를 잡은 채로 멈췄으면(RETURN_TO_ORIGIN) -> (상승) -> HOME 경유 ->
      원래 집었던 위치(apple_origin_pos)로 돌아가서 그리퍼를 열어 내려놓음 ->
      CAMERA로 이동.
    - 사과를 안 잡은 채로 멈췄으면(RESUME_AT_CAMERA) -> (상승) -> HOME 경유 ->
      CAMERA로 바로 이동해서 사이클을 이어서 재개.

EstopRecoveryTracker는 상태 추적/결정 로직만 담당 (하드웨어 의존 없음, 유닛테스트 가능).
execute_recovery()는 실제 이동/그리퍼 함수를 인자로 받아 그 계획을 실행하는 실행 layer.

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


RECOVERY_LIFT_MM = 400  # 비상정지 직후 벽/테이블 회피용 최소 상승량(mm)
RECOVERY_SETTLE_SEC = 1.0  # 완전히 멈춘 다음에도 추가로 더 대기하는 여유 시간(초)
RECOVERY_POLL_SEC = 0.05  # check_motion()으로 정지 여부를 확인하는 폴링 간격(초)

# amovel/amovej로 이동 명령을 낸 직후, 바로 check_motion()을 확인하면 컨트롤러
# 내부 상태가 아직 "이동 중"으로 갱신되기 전이라 옛 상태(정지)가 그대로 한 번
# 읽힐 수 있음(실측 확인된 문제: RECOVERY_LIFT_MM=400mm를 기본 속도 45mm/s로
# 이동하면 9초 가까이 걸리는데, 실제 로그에서 이 구간이 폴링 0회·1.2초 만에
# "정지"로 판정된 사례가 나옴 - 로봇이 한창 상승 중인데 다음 이동이 먼저 나가서
# 겹쳐 보였음). 고정된 초기 대기 시간만으로는 이 레이스를 완전히 막을 수
# 없다는 게 실측으로 증명됐으므로, "정지"가 연속으로 REQUIRED_IDLE_CONFIRMATIONS
# 번 확인돼야만 진짜로 멈췄다고 신뢰하는 방식으로 바꿈 - 단발성 오독이 섞여도
# 다음 확인에서 "이동 중"이 다시 잡히면 카운트가 리셋되므로 구조적으로 안전함.
REQUIRED_IDLE_CONFIRMATIONS = 3


def _wait_until_fully_stopped(check_motion, wait, settle_sec, node=None, label=""):
    """
    이동이 "확실히" 끝날 때까지 기다림. movel/movej는 이론적으로 물리적으로
    끝날 때까지 블록해야 하지만, 고정된 wait(0.3)만으로는 다음 이동이
    위치오차가 아직 다 끝나기도 전에 스케줄되어(블렌딩) 다음과의 의도치 않은
    충돌하는 문제가 있었음. check_motion()이 "정지"를 REQUIRED_IDLE_CONFIRMATIONS
    번 연속으로 확인해줄 때까지 반복 확인한 뒤, 추가로 settle_sec만큼 한 번 더
    여유를 둠.

    node/label을 넘기면(선택) 폴링 횟수와 실제 걸린 시간을 로그로 남김 -
    실제 하드웨어에서 상승/이동이 여전히 겹쳐 보이는 문제의 원인을 추측이 아니라
    확인하기 위함 (폴링이 적게 나오는데도 여전히 겹쳐 보이면 소프트웨어 타이밍이
    아니라 DSR 컨트롤러 자체의 모션 블렌딩 문제일 가능성으로 좁혀짐).
    """
    import time as _time
    start = _time.monotonic()

    poll_count = 0
    idle_confirmations = 0
    while idle_confirmations < REQUIRED_IDLE_CONFIRMATIONS:
        if check_motion():
            idle_confirmations = 0
            poll_count += 1
        else:
            idle_confirmations += 1
        wait(RECOVERY_POLL_SEC)
    wait(settle_sec)

    if node is not None:
        elapsed = _time.monotonic() - start
        node.get_logger().info(
            f"[E-STOP][{label}] 정지 확인 완료: 폴링 {poll_count}회, 총 {elapsed:.2f}초 소요"
        )


def execute_recovery(
    node,
    plan: RecoveryPlan,
    *,
    movel: Callable[[Any], None],
    movej: Callable[[Any], None],
    wait: Callable[[float], None],
    check_motion: Callable[[], Any],
    gripper_open: Callable[[], bool],
    home_pos,
    camera_pos,
    get_current_pos: Callable[[], Any],
    posx_factory: Callable[..., Any],
    lift_mm: int = RECOVERY_LIFT_MM,
    settle_sec: float = RECOVERY_SETTLE_SEC,
) -> None:
    """
    plan에 따라 실제 복구 동작을 수행.

    movel/gripper_open은 실제로는 DSR_ROBOT2.movel / openclose.gripper_open이지만,
    여기서는 주입받아 사용하므로 이 함수 자체는 하드웨어 의존 없이 테스트 가능함.
    get_current_pos/posx_factory/check_motion도 마찬가지로 주입받음 (실제로는
    DSR_ROBOT2.get_current_posx()[0] / DSR_ROBOT2.posx / DSR_ROBOT2.check_motion).

    home_pos는 posj(관절 공간) 값이라 movel(데카르트 좌표 이동)에 넣으면 안 됨 -
    posj(0,0,90,0,90,0)을 movel에 넘기면 "x=0,y=0,z=90mm"(로봇 베이스 바로 위
    90mm)로 해석되어 팔이 몸통 쪽으로 확 이동하는 사고가 났었음. 그래서
    HOME 이동만 movej로 분리함.

    가장 먼저 하는 일 두 가지:
    1) "지금 있는 그 자리로 가라"는 명령으로 컨트롤러에 남아있을 수 있는 잔여
       이동 목표를 덮어써서 취소함 (force_place.py의 amovel(stop_pos, ...) 취소
       기법과 동일한 원리 - motion/move_stop으로 물리적으로는 멈춰도, 아직
       가려던 목표가 컨트롤러 내부에 남아있으면 다음 명령과 의사 잔여 방향으로
       움직일 수 있음).
    2) 그 다음 수직으로만 lift_mm(기본 400mm)만큼 들어올림. HOME/원위치/
       CAMERA로 바로 이동하면, 멈춘 위치가 벽/테이블에 가까울 때(예: 박스
       웨이포인트 근처) 그 경로 도중 충돌할 수 있어서, 다른 어떤 이동보다
       먼저 이 위치에서 거침.

    매 movel/movej 호출 뒤에는 반드시 _wait_until_fully_stopped()로 확실히
    멈춘 걸 확인한 다음에만 다음 이동으로 넘어감 - 고정 시간 대기만으로는
    위치오차가 아직 다 끝나기도 전에 다음 이동이 스케줄되어(블렌딩) 다음과
    의도치 않은 방향으로 움직이다 충돌하는 문제가 있었음.
    """
    node.get_logger().info(f"[E-STOP] 복구 시작: {plan.action}")

    current_x, current_y, current_z, current_rx, current_ry, current_rz = get_current_pos()

    flush_pos = posx_factory(current_x, current_y, current_z, current_rx, current_ry, current_rz)
    movel(flush_pos)
    _wait_until_fully_stopped(check_motion, wait, settle_sec, node=node, label="flush")

    lifted_pos = posx_factory(
        current_x, current_y, current_z + lift_mm, current_rx, current_ry, current_rz
    )
    node.get_logger().info(f"[E-STOP] 충돌 회피를 위해 {lift_mm}mm 상승합니다.")
    movel(lifted_pos)
    _wait_until_fully_stopped(check_motion, wait, settle_sec, node=node, label="lift")

    movej(home_pos)
    _wait_until_fully_stopped(check_motion, wait, settle_sec, node=node, label="home")

    if plan.action == "RETURN_TO_ORIGIN":
        movel(plan.target_pos)
        _wait_until_fully_stopped(check_motion, wait, settle_sec, node=node, label="origin")
        if not gripper_open():
            node.get_logger().error("[E-STOP] 원위치 복귀 중 그리퍼 오픈 실패 - 하드웨어 상태를 확인하세요.")

    movel(camera_pos)
    _wait_until_fully_stopped(check_motion, wait, settle_sec, node=node, label="camera")
    node.get_logger().info("[E-STOP] 복구 완료, CAMERA에서 재개")


def check_and_recover(
    node,
    status_bus,
    tracker: EstopRecoveryTracker,
    emergency_stop_event,
    *,
    movel: Callable[[Any], None],
    movej: Callable[[Any], None],
    wait: Callable[[float], None],
    check_motion: Callable[[], Any],
    gripper_open: Callable[[], bool],
    home_pos,
    camera_pos,
    get_current_pos: Callable[[], Any],
    posx_factory: Callable[..., Any],
) -> bool:
    """
    emergency_stop_event(threading.Event)가 걸려 있으면 복구를 수행하고 True를 반환.
    호출부(box_sequence_test.py의 메인 루프)는 True가 반환되면 지금 처리 중이던
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
        node, plan, movel=movel, movej=movej, wait=wait, check_motion=check_motion,
        gripper_open=gripper_open,
        home_pos=home_pos, camera_pos=camera_pos,
        get_current_pos=get_current_pos, posx_factory=posx_factory,
    )

    if plan.action == "RETURN_TO_ORIGIN":
        # execute_recovery가 이미 원위치에서 그리퍼를 열어 내려놨으므로 이제 안 잡은 상태
        tracker.mark_placed()

    # 복구가 끝나면 로봇은 실제로 CAMERA 위치에 서 있고 그리퍼도 열려있는 상태임
    # (RESUME_AT_CAMERA는 애초에 아무것도 안 쥐고 있었고, RETURN_TO_ORIGIN은
    # 위에서 명시적으로 gripper_open()을 호출함). 그런데 이 사실을 백엔드에
    # 알리지 않으면(둘 다 실제로 겪은 문제):
    #   1) /gripper/status를 다시 발행 안 하면, E-STOP 이전에 "잡고 있었음"으로
    #      백엔드에 기록된 gripper_grasped=True가 그대로 남아있음
    #   2) process_state를 READY가 아닌 다른 값으로 두면, 백엔드의
    #      _vla_consumer_loop가 "카메라 위치에서 대기 중"으로 인식 못해서
    #      다음 Vision 감지를 영원히 무시함
    # 로봇은 멀쩡히 다음 사과를 받을 준비가 됐는데, 백엔드만 그걸 모른 채
    # 새 작업을 영영 안 보내는 상태로 멈춰버림 - 그래서 반드시 둘 다 갱신함.
    status_bus.publish_gripper_status(False)
    emergency_stop_event.clear()
    status_bus.set_state("READY")
    return True
