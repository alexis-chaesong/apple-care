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


RECOVERY_LIFT_MM = 100  # 비상정지 직후 벽/테이블 회피용 최소 상승량(mm)
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

# 상승(lift) 이동이 실제로 일어났는지 검증할 때 쓰는 값들. 실측(E-STOP 로그)으로
# lift movel이 check_motion() 타이밍 레이스 때문에 "정지 확인 완료"로 오판되면서도
# 실제로는 로봇이 전혀 안 움직인 사례(z 변화 -0.0mm)가 반복 확인됐음 - 그래서
# execute_recovery()의 lift 단계는 매 시도 뒤 실제 z 변화량을 검증하고, 목표치의
# LIFT_MIN_ACHIEVED_FRAC 미만이면 상승이 안 된 것으로 보고 재시도함.
LIFT_RETRY_COUNT = 3
LIFT_MIN_ACHIEVED_FRAC = 0.5

# movel(pos, mod=...)의 mod 값 - DSR_ROBOT2.DR_MV_MOD_ABS/REL과 숫자가 같지만, 이
# 모듈은 하드웨어 의존을 피하려고 DSR_ROBOT2를 import하지 않으므로 값만 그대로 복제함.
MOVE_MOD_ABS = 0
MOVE_MOD_REL = 1


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
    movel: Callable[..., None],
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
    resume_motion: Optional[Callable[[], Any]] = None,
    describe_robot_state: Optional[Callable[[], str]] = None,
) -> None:
    """
    plan에 따라 실제 복구 동작을 수행.

    resume_motion: 넘겨주면(예: safe_motion.resume_motion을 노드에 바인딩한 callable),
        복구 이동을 내보내기 전에 한 번 호출함. E-STOP 시 stop_motion()이
        move_stop(stop_mode=DR_HOLD)로 로봇을 "일시정지" 상태로 만드는데, 실측으로
        확인된 문제 - 이 상태에서 곧바로 새 movel을 내리면 서비스 응답은
        success=True로 오지만(명령 자체는 접수됨) 로봇이 물리적으로 전혀 안 움직이는
        경우가 있었음(Doosan 드라이버는 move_stop/move_resume이 쌍으로 설계돼 있는데
        move_resume을 호출하는 곳이 코드베이스에 없었음). None이면(기본값) 호출하지
        않음 - 하위 호환용. 주의: resume_motion() 자체가 실측으로 실패 응답을 준
        사례가 있었음(서비스는 찾았지만 move_resume()이 false 반환) - "재개할 게
        없어서"인지 다른 이유인지 아직 불확실해서, lift가 여전히 안 움직이는 원인을
        더 좁히기 위해 아래 describe_robot_state를 같이 씀.

    describe_robot_state: 넘겨주면(예: safe_motion.get_controller_robot_state +
        ROBOT_STATE_NAMES를 조합한 callable, () -> str), lift 재시도가 실패할
        때마다 컨트롤러가 실제로 보고하는 로봇 상태(STANDBY/MOVING/SAFE_STOP 등)를
        로그에 남김. resume_motion 호출이 실패해도(응답은 왔지만 success=False)
        movel(lift) 자체는 서비스 응답 성공을 받으면서도 z가 전혀 안 바뀌는 현상이
        재현됐는데, 그 순간 컨트롤러 상태가 뭐였는지가 다음 원인 파악의 핵심 단서라
        추가함. None이면(기본값) 조회하지 않음 - 하위 호환용.

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
    2) 그 다음 수직으로만 lift_mm(기본 200mm)만큼 들어올림. HOME/원위치/
       CAMERA로 바로 이동하면, 멈춘 위치가 벽/테이블에 가까울 때(예: 박스
       웨이포인트 근처) 그 경로 도중 충돌할 수 있어서, 다른 어떤 이동보다
       먼저 이 위치에서 거침.

    매 movel/movej 호출 뒤에는 반드시 _wait_until_fully_stopped()로 확실히
    멈춘 걸 확인한 다음에만 다음 이동으로 넘어감 - 고정 시간 대기만으로는
    위치오차가 아직 다 끝나기도 전에 다음 이동이 스케줄되어(블렌딩) 다음과
    의도치 않은 방향으로 움직이다 충돌하는 문제가 있었음.
    """
    node.get_logger().info(f"[E-STOP] 복구 시작: {plan.action}")

    if resume_motion is not None:
        # E-STOP 시 stop_motion()이 걸어둔 Hold 상태를 풀어줌 - 이걸 안 하면
        # 아래 movel들이 서비스 응답은 성공으로 받으면서도 실제로는 움직이지
        # 않는 문제가 있었음(위 resume_motion 인자 설명 참고).
        resume_motion()

    if describe_robot_state is not None:
        node.get_logger().info(f"[E-STOP] 복구 시작 시점 컨트롤러 상태: {describe_robot_state()}")

    current_x, current_y, current_z, current_rx, current_ry, current_rz = get_current_pos()

    flush_pos = posx_factory(current_x, current_y, current_z, current_rx, current_ry, current_rz)
    movel(flush_pos)
    _wait_until_fully_stopped(check_motion, wait, settle_sec, node=node, label="flush")

    # 실측(E-STOP 로그)으로 확인된 문제: 절대좌표(ABS) 목표로 lift를 계산해서
    # movel을 내리면, 서비스 응답은 성공(ret=0)으로 오고 컨트롤러 상태도 정상
    # (STANDBY)으로 보고되는데도 로봇이 전혀 안 움직이는 경우가 반복 재현됐음 -
    # stop_mode(HOLD/QSTOP)를 바꿔봐도 동일해서 정지 방식 문제가 아니라, 그 자세에서
    # orientation을 고정한 채 그 절대 목표가 IK/작업공간상 도달 불가능해서 조용히
    # 무시된 것으로 보임. 그래서 절대좌표 대신 상대이동(REL)으로 바꿈 - 상대이동은
    # 현재 관절해에서 연속적으로 풀리는 경우가 많아 이 문제를 우회할 수 있음.
    # movel(pos, mod)는 하드웨어 의존이 없는 이 모듈 기준으로 "양수 lift_mm=상승"
    # 의미만 유지함 - 실제 DSR에 보낼 때 z축 부호를 뒤집어야 하는지는 하드웨어
    # 어댑터(box_sequence_test.py의 _recovery_movel)가 처리할 책임임.
    rel_lift_pos = posx_factory(0, 0, lift_mm, 0, 0, 0)
    for attempt in range(1, LIFT_RETRY_COUNT + 1):
        cz = get_current_pos()[2]
        node.get_logger().info(
            f"[E-STOP] 충돌 회피를 위해 {lift_mm}mm 상승합니다(상대이동). "
            f"(현재 z={cz:.1f}, 시도 {attempt}/{LIFT_RETRY_COUNT})"
        )
        movel(rel_lift_pos, mod=MOVE_MOD_REL)
        _wait_until_fully_stopped(check_motion, wait, settle_sec, node=node, label="lift")

        actual_z_after_lift = get_current_pos()[2]
        achieved_mm = actual_z_after_lift - cz
        node.get_logger().info(
            f"[E-STOP][lift] 실제 이동 후 z={actual_z_after_lift:.1f} "
            f"(기대값={cz + lift_mm:.1f}, 실제 상승량={achieved_mm:.1f}mm / 목표 {lift_mm}mm)"
        )
        if achieved_mm >= lift_mm * LIFT_MIN_ACHIEVED_FRAC:
            break
        node.get_logger().error(
            f"[E-STOP][lift] 상승이 실행되지 않은 것으로 보입니다(실제 상승량 {achieved_mm:.1f}mm) - 재시도합니다."
        )
        if describe_robot_state is not None:
            node.get_logger().error(f"[E-STOP][lift] 실패 시점 컨트롤러 상태: {describe_robot_state()}")
    else:
        raise RuntimeError(
            f"[E-STOP] {LIFT_RETRY_COUNT}회 재시도에도 상승(lift)이 실행되지 않았습니다 - "
            "충돌 위험이 있어 나머지 복구 이동(HOME/원위치/CAMERA)을 중단합니다. "
            "하드웨어/컨트롤러 상태를 확인하세요."
        )

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
    movel: Callable[..., None],
    movej: Callable[[Any], None],
    wait: Callable[[float], None],
    check_motion: Callable[[], Any],
    gripper_open: Callable[[], bool],
    home_pos,
    camera_pos,
    get_current_pos: Callable[[], Any],
    posx_factory: Callable[..., Any],
    check_hw_safety_stop: Optional[Callable[[], "tuple"]] = None,
    resume_motion: Optional[Callable[[], Any]] = None,
    describe_robot_state: Optional[Callable[[], str]] = None,
    wait_for_resume: Optional[Callable[[], None]] = None,
) -> bool:
    """
    emergency_stop_event(threading.Event)가 걸려 있으면 복구를 수행하고 True를 반환.
    호출부(box_sequence_test.py의 메인 루프)는 True가 반환되면 지금 처리 중이던
    사과를 중단하고 루프 top(다시 decision 대기)으로 돌아가야 함.

    status_bus는 set_state(state, msg=None)를 제공하는 객체(StatusBus)면 됨.

    check_hw_safety_stop: 넘겨주면(예: safe_motion.is_controller_in_hardware_safety_stop을
        노드에 바인딩한 callable), 실제 복구 이동을 내보내기 전에 컨트롤러가
        하드웨어 레벨로(SAFE_STOP/EMERGENCY_STOP 등) 안전정지 상태인지 먼저 확인함.
        () -> (is_blocked: bool, state_name: str)를 반환해야 함. 펜던트/안전펜스의
        물리 비상정지 버튼이 눌린 경우 emergency_stop_event(소프트웨어 플래그)만
        보고 복구 이동을 내보내면 명령이 조용히 무시되거나 예상치 못하게 동작할 수
        있어서, 하드웨어가 안전정지 중이면 복구를 시작하지 않고 emergency_stop_event를
        그대로 걸어둔 채 True만 반환함(호출부는 다음 루프에서 다시 확인하게 됨).
        None이면(기본값) 이 확인을 건너뜀 - 하위 호환용.

    resume_motion: execute_recovery()로 그대로 전달됨 - safe_motion.resume_motion 참고.
    describe_robot_state: execute_recovery()로 그대로 전달됨 - execute_recovery의
        describe_robot_state 설명 참고.

    wait_for_resume: 넘겨주면(예: threading.Event.wait을 바인딩한 callable, 인자
        없이 블록킹 호출), 물리 복구(상승->홈->[원위치]->카메라)가 다 끝난 뒤에도
        바로 READY로 전환해 다음 사이클(재탐지)을 재개하지 않고, 이 함수가 리턴할
        때까지 기다림 - 프론트엔드의 "재개" 버튼이 눌려야만(백엔드의 RESUME
        명령이 도착해야만) 계속 진행되게 하기 위함. 이전에는 무조건 자동으로
        READY까지 갔는데, 실제 안전정지 상황에서는 운영자가 현장을 확인하고
        명시적으로 재개시켜야 한다는 요구사항 때문에 추가함. None이면(기본값)
        기다리지 않고 기존처럼 즉시 READY로 전환함 - 하위 호환용.

    복구 후에는 emergency_stop_event를 clear()해서 다음 루프에서 다시 걸리지 않게
    하고, (wait_for_resume이 없으면) "무조건 CAMERA로 보내서 재개" 정책에 따라
    별도의 재가 명령 없이 바로 다음 decision을 받을 수 있는 상태로 돌린다.
    """
    if not emergency_stop_event.is_set():
        return False

    if check_hw_safety_stop is not None:
        is_blocked, state_name = check_hw_safety_stop()
        if is_blocked:
            msg = (
                f"컨트롤러가 하드웨어 안전정지 상태({state_name})입니다 - 현장 안전을 "
                "확인한 뒤 펜던트/컨트롤러에서 안전정지를 해제해야 소프트웨어 복구를 "
                "시작할 수 있습니다."
            )
            node.get_logger().error(f"[E-STOP] {msg}")
            status_bus.set_state("ERROR", f"controller_safety_stop:{state_name}")
            status_bus.publish_safety("ERR_SYSTEM", msg)
            # emergency_stop_event는 그대로 걸어둔 채 리턴 - 호출부가 다시 decision을
            # 받지 않고 다음 루프에서 이 함수를 재호출해 상태를 다시 확인하게 함.
            wait(1.0)
            return True

    node.get_logger().error("EMERGENCY_STOP 감지 - 복구 절차를 시작합니다.")
    status_bus.set_state("ERROR", "emergency_stop")

    plan = tracker.get_recovery_plan()
    execute_recovery(
        node, plan, movel=movel, movej=movej, wait=wait, check_motion=check_motion,
        gripper_open=gripper_open,
        home_pos=home_pos, camera_pos=camera_pos,
        get_current_pos=get_current_pos, posx_factory=posx_factory,
        resume_motion=resume_motion,
        describe_robot_state=describe_robot_state,
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

    if wait_for_resume is not None:
        # 물리적으로는 이미 안전한 위치(카메라)에 서 있지만, 운영자가 현장을
        # 확인하고 프론트엔드에서 재개 버튼을 누르기 전까지는 READY로 넘어가지
        # 않음 - READY가 아니면 백엔드가 새 Vision 판단을 하지 않으므로(main.py의
        # _vla_consumer_loop 게이팅) 이 상태로 있는 동안은 안전하게 대기만 함.
        node.get_logger().warn("[E-STOP] 물리 복구 완료 - 재개(RESUME) 신호를 기다립니다.")
        status_bus.set_state("ERROR", "waiting_for_resume")
        wait_for_resume()
        node.get_logger().info("[E-STOP] 재개(RESUME) 신호를 받아 사이클을 계속합니다.")

    status_bus.set_state("READY")
    return True
