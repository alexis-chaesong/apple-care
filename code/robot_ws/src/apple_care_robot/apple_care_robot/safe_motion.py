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
호출하는 함수들을 수정하는 고치지 않아도, 맨 위(apple_sorting_cycle.py의
메인 루프)에서 try/except 한 번으로 전체를 잡아서 복구(estop_handler.py의
check_and_recover)로 넘어갈 수 있음.
"""

from dsr_msgs2.srv import MoveStop

# MoveStop.srv 기준 stop_mode 값 (참고용 전체 목록):
#   0 = DR_QSTOP_STO (Quick stop, 안전 토크 차단 - 보통 재가동하려면 파트/원점 재설정 필요)
#   1 = DR_QSTOP     (Quick stop, 토크 차단 없음)
#   2 = DR_SSTO       (Soft stop)
#   3 = DR_HOLD       (Hold stop - 서보 유지, 소프트웨어에서 바로 재가동 가능)
# 우리는 "정지 후 CAMERA로 다시 재가동"이 정책이므로 HOLD를 기본으로 씀.
STOP_MODE_HOLD = 3


class EmergencyStopError(RuntimeError):
    """비상정지가 감지되어 진행 중이던 동작을 중단했음을 알리는 예외.

    safe_movel/safe_movej 안에서 발생하며, 일부러 그 자리에서 잡지 않고
    호출부(예: apple_sorting_cycle.py의 메인 루프)까지 그대로 전파시켜서
    한 곳에서만 복구 로직을 태우면 되게 만든다.
    """


def stop_motion(node, stop_mode: int = STOP_MODE_HOLD) -> bool:
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
        node: 서비스 클라이언트를 만들 rclpy 노드. apple_sorting_cycle.py에서는
            DSR 제어용 메인 node가 아니라, 이미 별도 스레드에서 spin 중인
            comm_node를 넘겨야 함 (메인 node를 여기서 또 spin하면 충돌 위험).
        stop_mode: 위 STOP_MODE_* 상수 중 하나.

    Returns:
        bool: 정지 명령이 성공적으로 전달됐으면 True.
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
        if result is not None and result.success:
            node.get_logger().warning(
                f"[safe_motion] 정지 성공: service={service_name}, stop_mode={stop_mode}"
            )
            return True

        node.get_logger().error(
            f"[safe_motion] 정지 응답 실패: service={service_name}, stop_mode={stop_mode}"
        )

    node.get_logger().error(
        f"[safe_motion] motion/move_stop 서비스를 찾지 못했거나 호출에 실패했습니다: {service_names}"
    )
    return False


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

    stop_motion_fn(node, STOP_MODE_HOLD)
    raise EmergencyStopError("비상정지 요청으로 동작이 중단되었습니다.")


def _wait_until_motion_done(node, emergency_stop_event, *, check_motion, wait, stop_motion_fn=stop_motion) -> None:
    """amovel/amovej로 시작한 비동기 이동이 끝날 때까지, 짧은 주기로 비상정지를
    감시하며 동기화하는 공용 폴링 루프 (safe_movel/safe_movej가 함께 사용)."""
    while check_motion():
        raise_if_emergency_stop(node, emergency_stop_event, stop_motion_fn=stop_motion_fn)
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
) -> None:
    """
    movel(target)과 같은 목적지로 이동하되, 이동 중에는 비상정지를 감시함.
    비상정지가 걸리면 EmergencyStopError를 던지고(로봇은 이미 멈춘 상태),
    정상적으로 도착하면 조용히 리턴함.

    amovel/check_motion/wait을 인자로 주입받는 이유: 실제 DSR_ROBOT2 하드웨어
    없이도(가짜 함수로) 이 폴링/예외 로직 자체를 테스트할 수 있게 하기 위함
    (실제 사용 시에는 apple_sorting_cycle.py가 DSR_ROBOT2.amovel/check_motion/wait을
    그대로 넘겨줌).
    """
    amovel(target, vel=vel, acc=acc, ref=ref)
    _wait_until_motion_done(
        node, emergency_stop_event, check_motion=check_motion, wait=wait, stop_motion_fn=stop_motion_fn
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
) -> None:
    """movej(target)의 안전(비상정지 감시) 버전. safe_movel과 동일한 원리,
    관절 이동(amovej)에 대해서만 적용."""
    amovej(target, vel=vel, acc=acc)
    _wait_until_motion_done(
        node, emergency_stop_event, check_motion=check_motion, wait=wait, stop_motion_fn=stop_motion_fn
    )
