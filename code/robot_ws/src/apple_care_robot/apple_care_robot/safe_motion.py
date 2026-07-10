"""
safe_motion.py
===============
"비상정지를 감시하면서 안전하게 로봇을 움직이는 것"만 전담하는 모듈.
(estop_handler.py는 "정지 후 어디로 복구할지 결정/실행"을 담당 - 서로 관심사가 다름)

핵심 아이디어 (동기 vs 비동기):
    - movel()/movej()는 "동기(sync)" 함수라서, 로봇이 목적지에 도착할 때까지
      파이썬 코드가 그 자리에서 멈춰 기다림. 기다리는 동안은 다른 코드가 전혀
      실행되지 않아서 "비상정지 눌렸나?"조차 확인할 수가 없음.
    - amovel()/amovej()는 "비동기(async)" 함수라서, 로봇에게 이동 명령만 던지고
      파이썬 코드는 즉시 다시 오로 넘어감. 그래서 로봇이 움직이는 동안에도
      우리 코드가 반복문을 돌면서 "비상정지 눌렸나?"를 아주 짧은 간격(10ms)으로
      계속 확인할 수 있음.
    - 다만 비동기로 바꾼다고 로봇이 저절로 멈추는 건 아님. 실제로 멈추려면
      아래 stop_motion()이 하드웨어 컨트롤러에게 "지금 멈춰!"라는 별도 명령
      (motion/move_stop 서비스)을 직접 보내야 함.

전체 흐름 (safe_movel 기준):
    1) amovel(target)로 이동을 "시작만" 시킴 (바로 리턴됨, 로봇은 백그라운드에서 움직임)
    2) while check_motion(): 아직 이동 중이면 반복
    3)     raise_if_emergency_stop(...): 비상정지 이벤트가 걸려 있으면
           stop_motion()으로 실제 정지 명령을 보내고 EmergencyStopError를 던짐
    4)     wait(0.01): 10ms 재고 다시 2)번으로

이 예외(EmergencyStopError)는 일부러 여기서 잡지 않고 그대로 위로 전파해.
그래야 pick_apple()/force_controlled_place()처럼 safe_movel을 여러 겹 안쪽에서
호출하는 함수들을 수정하는 고치지 않아도, 맨 위(box_sequence_test.py의
메인 루프)에서 try/except 한 번으로 전체를 잡아서 복구(estop_handler.py의
check_and_recover)로 넘어갈 수 있음.
"""

import time as _time

from dsr_msgs2.srv import MoveStop, MoveResume, GetRobotState

# MoveStop.srv 기준 stop_mode 값 (참고용 전체 목록):
#   0 = DR_QSTOP_STO (Quick stop, 안전 토크 차단 - 보통 재가동하려면 파트/원점 재설정 필요)
#   1 = DR_QSTOP     (Quick stop, 토크 차단 없음)
#   2 = DR_SSTO       (Soft stop)
#   3 = DR_HOLD       (Hold stop - 서보 유지, 소프트웨어에서 바로 재가동 가능 "이라는
#                       문서상 설명과 달리, 실측으로는 move_stop(HOLD) 이후 movel이
#                       서비스 응답 성공을 받으면서도 실제로는 전혀 안 움직이는 문제가
#                       반복 재현됨. Doosan은 HOLD 해제용으로 move_resume()을 따로
#                       제공하는데, 그걸 호출해도(motion/move_resume) 재가동이 안 됨)
DR_QSTOP_STO = 0
DR_QSTOP = 1
DR_SSTOP = 2
DR_HOLD = 3

# 자매 프로젝트 auto_dump_robot_pkg(motion_runtime.py의 stop()/request_motion_stop())는
# HOLD를 아예 쓰지 않고 DR_QSTOP을 기본으로, 실패 시 DR_QSTOP_STO로 폴백하는 방식으로
# 이 문제를 겪지 않고 정지 직후 movel/movej가 정상 동작함이 실기로 검증돼 있음 - 그
# 방식을 그대로 채택함 ("정지 후 CAMERA로 다시 재가동" 정책은 QSTOP으로도 동일하게
# 충족됨 - QSTOP은 토크를 차단하지 않으므로 소프트웨어에서 바로 새 움직임을 낼 수 있음).
STOP_MODE_HOLD = DR_HOLD  # 하위 호환용 이름 - 더 이상 기본값으로 쓰지 않음
STOP_MODE_DEFAULT = DR_QSTOP

# 두산 컨트롤러가 system/get_robot_state 서비스로 반환하는 상태값 이름표.
ROBOT_STATE_NAMES = {
    0: "INITIALIZING",
    1: "STANDBY",
    2: "MOVING",
    3: "SAFE_OFF",
    4: "TEACHING",
    5: "SAFE_STOP",
    6: "EMERGENCY_STOP",
    7: "HOMMING",
    8: "RECOVERY",
    9: "SAFE_STOP2",
    10: "SAFE_OFF2",
    15: "NOT_READY",
}

# 이 상태들에서는 컨트롤러가 물리적으로 안전정지 중이라 movel/movej 명령이 먹지
# 않거나 무시될 수 있으므로, 이 상태를 확인하지 않고 소프트웨어 복구 이동을
# 내보내면 안 됨.
CONTROLLER_SAFETY_STATES = {3, 5, 6, 8, 9, 10, 15}


def get_controller_robot_state(node):
    """
    dsr_controller2가 제공하는 system/get_robot_state 서비스를 호출해서
    두산 컨트롤러의 현재 로봇 상태(int)를 조회한다.

    이 값은 /robot/command로 들어오는 "소프트웨어" EMERGENCY_STOP 명령과는
    완전히 별개임 - 펜던트/안전펜스의 물리 비상정지 버튼이 눌리면
    emergency_stop_event(threading.Event)는 걸리지 않으면서 컨트롤러만 이 값을
    통해 하드웨어 레벨로 SAFE_STOP/EMERGENCY_STOP 상태가 됨.

    Args:
        node: 서비스 클라이언트를 만들 rclpy 노드 (stop_motion()과 동일하게
            보통 comm_node를 넘겨야 함).

    Returns:
        int 또는 None: 조회 성공 시 로봇 상태 코드, 실패(서비스 없음/타임아웃/
        응답 실패)하면 None.
    """
    # 서비스 이름 후보를 여러 개 순서대로 시도하는 이유: stop_motion()/resume_motion()과
    # 동일함(Jazzy/Humble 드라이버 버전에 따라 실제 서비스 경로가 다름). 실측으로 확인된
    # 문제: 이 함수만 "dsr_controller2" 세그먼트가 포함된 Jazzy 경로 하나만 시도하다가,
    # Humble 드라이버(dsr_controller2 세그먼트 없이 /dsr01/system/get_robot_state로 등록됨)
    # 환경에서는 항상 "서비스가 준비되지 않았습니다" 경고만 반복 출력하고 하드웨어 안전정지
    # 감지 자체가 죽어있었음.
    service_names = (
        f"/{node.get_namespace().strip('/')}/dsr_controller2/system/get_robot_state",  # 최신(Jazzy) 드라이버
        f"/{node.get_namespace().strip('/')}/system/get_robot_state",                  # 기존(Humble) 드라이버
        "system/get_robot_state",                                                       # 노드 네임스페이스 기준 상대 경로
    )

    import rclpy  # 순환 의존 방지를 위해 필요한 지점에서만 import

    for service_name in service_names:
        client = node.create_client(GetRobotState, service_name)
        if not client.wait_for_service(timeout_sec=0.2):
            continue

        future = client.call_async(GetRobotState.Request())
        rclpy.spin_until_future_complete(node, future, timeout_sec=0.8)
        if not future.done():
            node.get_logger().warning(f"로봇 상태 조회 시간 초과: service={service_name}")
            return None

        result = future.result()
        if result is None or not result.success:
            node.get_logger().warning(f"로봇 상태 조회 실패: service={service_name}")
            return None
        return int(result.robot_state)

    node.get_logger().warning(f"로봇 상태 조회 서비스를 찾지 못했습니다: {service_names}")
    return None


def is_controller_in_hardware_safety_stop(node):
    """
    컨트롤러가 지금 실제로(하드웨어 레벨로) SAFE_STOP/EMERGENCY_STOP 등
    안전정지 상태인지 확인한다.

    estop_handler.check_and_recover()가 복구 이동(movel/movej)을 내보내기
    직전에 반드시 이 함수로 먼저 확인해야 함 - 펜던트의 물리 비상정지 버튼이
    눌린 상태에서는 emergency_stop_event가 안 걸려 있어도(또는 이미 clear
    되었어도) 컨트롤러가 여전히 하드웨어 안전정지 중일 수 있고, 그 상태로
    복구 이동 명령을 내보내면 조용히 무시되거나 예상치 못하게 동작할 수 있음.

    Returns:
        (is_blocked, state_name): 조회 자체가 실패하면 (False, "") 반환함 -
        상태를 모른다고 복구를 무한정 막을 수는 없으므로 보수적으로 "차단하지
        않음"으로 처리함 (auto_dump_robot_pkg의 block_reset_if_controller_safety_stop
        과 동일한 원칙).
    """
    robot_state = get_controller_robot_state(node)
    if robot_state is None:
        return False, ""

    state_name = ROBOT_STATE_NAMES.get(robot_state, f"UNKNOWN({robot_state})")
    return robot_state in CONTROLLER_SAFETY_STATES, state_name


class EmergencyStopError(RuntimeError):
    """비상정지가 감지되어 진행 중이던 동작을 중단했음을 알리는 예외.

    safe_movel/safe_movej 안에서 발생하며, 일부러 그 자리에서 잡지 않고
    호출부(예: box_sequence_test.py의 메인 루프)까지 그대로 전파시켜서
    한 곳에서만 복구 로직을 태우면 되게 만든다.
    """


def stop_motion(node, stop_mode: int = STOP_MODE_DEFAULT) -> bool:
    """
    dsr_controller2가 제공하는 motion/move_stop 서비스를 호출해서 로봇의
    현재 동작을 실제로(하드웨어 레벨에서) 멈춘다.

    주의: DSR_ROBOT2.py에는 이 서비스를 부르는 함수가 나타나 있지 않아서,
    여기서 dsr_msgs2.srv.MoveStop으로 직접 서비스 클라이언트를 만들어서 호출한다.

    서비스 이름 후보를 여러 개 순서대로 시도하는 이유: 두산 드라이버 버전/설정에
    따라 실제 서비스 이름이 조금씩 다를 수 있어서(예: 예전엔 컨트롤러 노드
    이름이 경로에 안 들어감), 하나만 하드코딩하면 환경이 바뀔 때 통째로
    조용히 먹통이 되는 위험이 있음.

    Args:
        node: 서비스 클라이언트를 만들 rclpy 노드. box_sequence_test.py에서는
            DSR 제어용 메인 node가 아니라, 이미 별도 스레드에서 spin 중인
            comm_node를 넘겨야 함 (메인 node를 여기서 또 spin하면 충돌 위험).
        stop_mode: 위 STOP_MODE_* 상수 중 하나.

    Returns:
        bool: 정지 명령이 성공적으로 전달됐으면 True.

    주의 - "찾았지만 거절당함" vs "아예 못 찾음" 구분:
        dsr_controller2.cpp의 move_stop 콜백은 `res->success = Drfl->stop(...)`인데,
        Drfl->stop()은 "지금 멈출 동작이 아무것도 없으면" false를 반환한다 - 즉
        SAFE_STOP/SAFE_OFF처럼 하드웨어가 이미 로봇을 멈춰놓은 상태에서 뒤늦게
        move_stop을 호출하면, 서비스는 정상적으로 찾고 호출도 됐는데 "멈출 대상이
        없어서" success=False로 응답하는 게 정상이다(로봇은 이미 안전하게 멈춰있음).
        이 경우 실제로 정확한 서비스를 찾은 것이므로, 응답이 실패여도 나머지
        이름 후보를 계속 시도할 필요가 없다(다른 후보는 애초에 존재하지 않는
        이름이라 시간만 버림) - future.result()가 왔다는 것 자체가 "이 이름이
        맞다"는 증거이기 때문. 그래서 응답을 받은 순간(성공/실패 무관) 루프를
        끝내고, 응답 자체를 못 받은 경우(서비스가 진짜 없거나 타임아웃)만 다음
        후보로 넘어간다.
    """
    service_names = (
        f"/{node.get_namespace().strip('/')}/dsr_controller2/motion/move_stop",  # 최신 드라이버
        f"/{node.get_namespace().strip('/')}/motion/move_stop",                  # 기존(Humble) 드라이버
        "motion/move_stop",                                                       # 노드 네임스페이스 기준 상대 경로
    )

    for service_name in service_names:
        client = node.create_client(MoveStop, service_name)
        # timeout_sec을 짧게 두는 이유: 없는 서비스 이름을 오래 기다리면 그만큼
        # "정지"가 늦어짐. 후보 3개를 다 합쳐도 최대 1초 이내로 끝나도록 짧게 잡음.
        if not client.wait_for_service(timeout_sec=0.2):
            continue

        request = MoveStop.Request()
        request.stop_mode = int(stop_mode)
        future = client.call_async(request)

        import rclpy  # 순환 의존 방지를 위해 필요한 지점에서만 import
        rclpy.spin_until_future_complete(node, future, timeout_sec=0.8)

        result = future.result() if future.done() else None
        if result is None:
            # 이 이름으로는 응답 자체를 못 받음(타임아웃 등) - 진짜로 이 후보가
            # 아닐 수 있으니 다음 후보를 시도.
            continue

        if result.success:
            node.get_logger().warning(
                f"[safe_motion] 정지 성공: service={service_name}, stop_mode={stop_mode}"
            )
        else:
            # 응답을 받았다는 것 자체가 이 서비스 이름이 맞다는 뜻 - 실패는 보통
            # "이미 멈출 동작이 없음"(하드웨어 안전정지가 먼저 걸린 경우 등)이라
            # 위험 신호가 아니므로 error가 아닌 warning으로 남김.
            node.get_logger().warning(
                f"[safe_motion] 정지 명령이 거절됨(service={service_name}, "
                f"stop_mode={stop_mode}) - 이미 멈출 동작이 없는 상태(하드웨어 "
                "안전정지가 먼저 걸린 경우 등)일 가능성이 높아 무시해도 됨."
            )
        return result.success

    node.get_logger().error(
        f"[safe_motion] motion/move_stop 서비스를 찾지 못했습니다: {service_names}"
    )
    return False


def resume_motion(node) -> bool:
    """
    dsr_controller2가 제공하는 motion/move_resume 서비스를 호출해서, stop_motion()의
    move_stop(stop_mode=DR_HOLD)으로 걸린 "일시정지(Hold)" 상태를 실제로 풀어준다.

    실측(E-STOP 로그)으로 확인된 문제: move_stop(HOLD) 이후 곧바로 새 movel/amovel을
    내려도 서비스 응답은 success=True로 오는데(명령 자체는 접수됨) 로봇이 물리적으로
    전혀 안 움직이는 사례가 반복됐음. Doosan 드라이버(dsr_controller2.cpp의
    move_resume_cb -> Drfl->move_resume())를 보면 move_stop과 move_resume이 쌍으로
    설계되어 있는데, 이 코드베이스 어디에도 move_resume을 호출하는 곳이 없었음 -
    stop_motion()과 마찬가지로 DSR_ROBOT2.py도 이 서비스를 파이썬 함수로 감싸주지
    않아서 여기서 직접 서비스 클라이언트를 만들어 호출한다.

    Args:
        node: 서비스 클라이언트를 만들 rclpy 노드 (stop_motion()과 동일하게 보통 comm_node).

    Returns:
        bool: 재개 명령이 성공적으로 전달됐으면 True.
    """
    service_names = (
        f"/{node.get_namespace().strip('/')}/dsr_controller2/motion/move_resume",
        f"/{node.get_namespace().strip('/')}/motion/move_resume",
        "motion/move_resume",
    )

    for service_name in service_names:
        client = node.create_client(MoveResume, service_name)
        if not client.wait_for_service(timeout_sec=0.2):
            continue

        future = client.call_async(MoveResume.Request())

        import rclpy  # 순환 의존 방지를 위해 필요한 지점에서만 import
        rclpy.spin_until_future_complete(node, future, timeout_sec=0.8)

        result = future.result() if future.done() else None
        if result is not None and result.success:
            node.get_logger().warning(f"[safe_motion] 재개 성공: service={service_name}")
            return True

        node.get_logger().error(f"[safe_motion] 재개 응답 실패: service={service_name}")

    node.get_logger().error(
        f"[safe_motion] motion/move_resume 서비스를 찾지 못했거나 호출에 실패했습니다: {service_names}"
    )
    return False


def make_hw_safety_watcher(
    node,
    emergency_stop_event,
    check_hw_safety_stop,
    interval_sec: float = 0.5,
    stop_motion_fn=stop_motion,
    now_fn=_time.monotonic,
):
    """
    "이동 중"뿐 아니라 "힘제어(파지/내려놓기) 중"에도 실제 펜던트/안전펜스의
    물리 비상정지 버튼을 감지하기 위한 감시자(closure)를 만든다.

    배경: raise_if_emergency_stop()은 /robot/command로 들어오는 "소프트웨어"
    EMERGENCY_STOP 명령(emergency_stop_event)만 봄. 누군가 소프트웨어 명령
    없이 펜던트의 물리 버튼만 누르면 emergency_stop_event가 전혀 걸리지 않아서,
    check_and_recover() 쪽의 하드웨어 상태 확인(is_controller_in_hardware_safety_stop)
    으로는 못 잡음 - 그건 "복구를 시작하기 전"에만 확인하기 때문. 이 물리
    비상정지를 이동/힘제어 진행 중에도 잡으려면 컨트롤러 상태를 폴링 루프
    안에서도 확인해야 함.

    check_hw_safety_stop(예: is_controller_in_hardware_safety_stop을 노드에
    바인딩한 callable)은 ROS 서비스 호출이라 10ms 간격의 빠른 폴링 루프
    안에서 매번 부르면 과부하가 됨. 그래서 반환된 watch() 함수는 스스로
    interval_sec(기본 0.5초)마다 한 번씩만 실제 조회를 하도록 쓰로틀링함
    (그 사이 호출은 그냥 조용히 리턴).

    감지되면: emergency_stop_event.set() + stop_motion_fn 호출 +
    EmergencyStopError를 던짐 - raise_if_emergency_stop과 동일한 결과라서
    호출부는 기존의 단일 "except EmergencyStopError"에 그대로 합류하고,
    이후 estop_handler.check_and_recover()가 이 emergency_stop_event를 보고
    평소와 동일하게 복구를 수행함.

    Returns:
        watch: 인자 없는 callable. check_hw_safety_stop이 None이면 아무것도
        안 하는 함수를 반환함(하위 호환).
    """
    if check_hw_safety_stop is None:
        return lambda: None

    state = {"last_check": now_fn()}

    def watch() -> None:
        now = now_fn()
        if now - state["last_check"] < interval_sec:
            return
        state["last_check"] = now

        is_blocked, state_name = check_hw_safety_stop()
        if not is_blocked:
            return

        msg = f"컨트롤러 하드웨어 안전정지 감지({state_name}) - 즉시 정지합니다."
        node.get_logger().error(f"[safe_motion] {msg}")
        emergency_stop_event.set()
        stop_motion_fn(node, STOP_MODE_DEFAULT)
        raise EmergencyStopError(msg)

    return watch


def raise_if_emergency_stop(node, emergency_stop_event, stop_motion_fn=stop_motion) -> None:
    """
    emergency_stop_event(threading.Event)가 걸려 있으면:
        1) stop_motion_fn으로 실제 하드웨어 정지 명령을 보내고
        2) EmergencyStopError를 던져서 지금 진행 중이던 동작을 중단시킴

    stop_motion_fn을 인자로 받는 이유: 실제 ROS 서비스 호출(stop_motion) 없이도
    "정지 감지 -> 정지 요청 -> 예외 발생" 흐름 자체를 가짜 함수로 테스트하기 위함.
    """
    if not emergency_stop_event.is_set():
        return

    node.get_logger().error("[safe_motion] 비상정지 감지 - 모션 정지를 요청합니다.")

    stop_motion_fn(node, STOP_MODE_DEFAULT)
    raise EmergencyStopError("비상정지 요청으로 동작이 중단되었습니다.")


def _wait_until_motion_done(
    node, emergency_stop_event, *, check_motion, wait, stop_motion_fn=stop_motion, hw_safety_watch=None,
) -> None:
    """amovel/amovej로 시작한 비동기 이동이 끝날 때까지, 짧은 주기로 비상정지를
    감시하며 동기화하는 공용 폴링 루프 (safe_movel/safe_movej가 함께 사용).

    amovel/amovej 호출 직후 곧바로 check_motion()을 확인하면, 컨트롤러 내부
    상태가 아직 "이동 중"으로 갱신되기 전이라 옛 상태(정지)가 잠깐 그대로
    읽혀서 while 루프가 0번 만에 빠져나가버릴 수 있음 - 그러면 실제로는 아직
    한참 진행 중인 이동인데 이 함수가 "끝났다"고 착각하고 바로 리턴해서, 호출부가
    곧바로 다음 이동을 시작해 진행 중이던 이번 이동과 겹치는(블렌딩) 문제가
    있었음 (estop_handler.py의 복구 이동에서 실제로 겪은 문제와 동일한 원인).
    그래서 폴링을 시작하기 전에 짧게 먼저 대기해서 "이동 중" 상태가 확실히
    반영된 뒤부터 확인하게 함 (force_place.py가 amovel 직후 time.sleep(0.1)을
    두는 것과 동일한 이유).

    hw_safety_watch: 넘겨주면(인자 없는 callable, make_hw_safety_watcher 참고)
        매 폴링마다 raise_if_emergency_stop 직후에 호출해서 컨트롤러의 하드웨어
        안전정지 상태(물리 비상정지 버튼)도 이동 중에 감시함. None이면(기본값)
        감시하지 않음 - 하위 호환용.
    """
    wait(0.1)
    while check_motion():
        raise_if_emergency_stop(node, emergency_stop_event, stop_motion_fn=stop_motion_fn)
        if hw_safety_watch is not None:
            hw_safety_watch()
        wait(0.01)


def safe_movel(
    node,
    target,
    emergency_stop_event,
    *,
    amovel,
    check_motion,
    wait,
    stop_motion_fn=stop_motion,
    vel=None,
    acc=None,
    ref=None,
    check_hw_safety_stop=None,
    hw_safety_check_interval_sec=0.5,
) -> None:
    """
    movel(target)과 같은 목적지로 이동하되, 이동 중에는 비상정지를 감시함.
    비상정지가 걸리면 EmergencyStopError를 던지고(로봇은 이미 멈춘 상태),
    정상적으로 도착하면 조용히 리턴함.

    amovel/check_motion/wait을 인자로 주입받는 이유: 실제 DSR_ROBOT2 하드웨어
    없이도(가짜 함수로) 이 폴링/예외 로직 자체를 테스트할 수 있게 하기 위함
    (실제 사용 시에는 box_sequence_test.py가 DSR_ROBOT2.amovel/check_motion/wait을
    그대로 넘겨줌).

    check_hw_safety_stop: 넘겨주면(예: safe_motion.is_controller_in_hardware_safety_stop을
        노드에 바인딩한 callable) 이동 중에도 컨트롤러의 하드웨어 안전정지
        상태(펜던트 물리 비상정지 버튼)를 감시함 - make_hw_safety_watcher 참고.
        None이면(기본값) 감시하지 않음.
    """
    amovel(target, vel=vel, acc=acc, ref=ref)
    hw_safety_watch = make_hw_safety_watcher(
        node, emergency_stop_event, check_hw_safety_stop, hw_safety_check_interval_sec, stop_motion_fn,
    )
    _wait_until_motion_done(
        node, emergency_stop_event, check_motion=check_motion, wait=wait, stop_motion_fn=stop_motion_fn,
        hw_safety_watch=hw_safety_watch,
    )


def safe_movej(
    node,
    target,
    emergency_stop_event,
    *,
    amovej,
    check_motion,
    wait,
    stop_motion_fn=stop_motion,
    vel=None,
    acc=None,
    check_hw_safety_stop=None,
    hw_safety_check_interval_sec=0.5,
) -> None:
    """movej(target)의 안전(비상정지 감시) 버전. safe_movel과 동일한 원리,
    관절 이동(amovej)에 대해서만 적용. check_hw_safety_stop도 safe_movel과 동일하게 동작."""
    amovej(target, vel=vel, acc=acc)
    hw_safety_watch = make_hw_safety_watcher(
        node, emergency_stop_event, check_hw_safety_stop, hw_safety_check_interval_sec, stop_motion_fn,
    )
    _wait_until_motion_done(
        node, emergency_stop_event, check_motion=check_motion, wait=wait, stop_motion_fn=stop_motion_fn,
        hw_safety_watch=hw_safety_watch,
    )
