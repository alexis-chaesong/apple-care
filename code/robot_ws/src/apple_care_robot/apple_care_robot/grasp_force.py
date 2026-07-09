"""
Force-Feedback Grasping
=========================

목적:
    사과 크기가 다 달라서, 그리퍼를 미리 정해둔 고정 힘으로만 닫으면
    작은 사과는 헐겁게 잡히고 큰 사과는 너무 세게 눌려 손상될 수 있음.
    두산 로봇 손목의 힘 센서(get_tool_force)로 파지 중 실시간 반발력을
    보면서, "이제 막 눌러서 저항이 생기기 시작한다" 싶은 시점까지만 힘을
    조금씩 올려서 사과마다 필요한 만큼만 잠그는 방식.

    force_place.py가 박스로 "내려놓을 때" 접촉을 힘으로 감지하는 것과 같은
    원리를, 사과를 "집을 때"(그리퍼 클로즈)에도 적용한 것.

주의:
    - 이 함수는 로봇이 이미 사과 위치(pick_pos)에 도달해 있고, 그리퍼가
      열려있는 상태에서 호출해야 함 (pick_helpers.pick_apple 참고).
    - openclose.py가 OnRobot Compute Box에 Modbus/TCP로 [force, width, 16]
      레지스터를 직접 쓰는 구조라, 목표 force 값을 매번 절대값으로 바로
      지정해서 닫을 수 있음 (gripper_close_with_force). 그래서 과거
      "/onrobot/sendCommand" 서비스 기반 코드처럼 'i'/'d' 증감 명령을 여러 번
      보내며 내부 상태를 별도로 추적(calibrate 등)할 필요가 없음.
    - 실시간으로 [grasp] 로그가 계속 찍히는데, 이는 "화면에 진행 상황을
      보여주는" 역할도 겸함. 나중에 Tkinter HMI와 연결하고 싶다면 이 로그
      대신 HMI 쪽 위젯을 갱신하는 방식으로 바꾸면 됨.
"""

import time

from apple_care_robot.openclose import gripper_close_with_force, GRIPPER_MAX_FORCE
from apple_care_robot.safe_motion import raise_if_emergency_stop, make_hw_safety_watcher

FORCE_STEP = 50                 # 매 단계 힘 증가량 (openclose.py 레지스터 단위, 0.1N = 5N)
GRASP_FORCE_THRESHOLD = 3.0     # 손목 반발력 변화량(N)이 이 값을 넘으면 "제대로 눌렀다/잡았다"고 판단
MAX_GRASP_STEPS = 6             # 무한 증가 방지 안전 장치 (6단계 * 50 = 300까지)

# 접촉을 감지한 힘 그대로 이동하면 관성/미끄러짐으로 놓칠 수 있어서,
# 감지 즉시 멈추는 대신 안전 여유분만큼 한 번 더 힘을 얹어줌.
SAFETY_MARGIN_STEPS = 2         # 안전 여유분 단계 수 (2단계 = +100, FORCE_STEP 기준)

# ---------------------------------------------------------------------------
# 겹쳐 있는 사과 접근 중 반발력 감지 시 후퇴
# ---------------------------------------------------------------------------
# 실측으로 확인된 문제: 겹쳐 있는 사과 두 개를 인식해도, 그 중 하나(pick_pos)로
# 그리퍼가 열린 채 내려가는 도중에 다른 하나를 옆에서 눌러버려 하드웨어 안전정지가
# 걸리는 사례가 있었음. grasp_apple_with_force_feedback()은 이미 pick_pos에
# 도착해 그리퍼를 "닫는" 단계의 힘만 감시하므로 이 구간은 다루지 못함 - 그래서
# "그리퍼가 열린 채 pick_pos로 내려가는 하강 구간" 자체를 별도로 감시해야 함.
REACTIVE_CONTACT_FORCE_THRESHOLD_N = 10.0  # 하강 중 이 값(N) 이상 반발력이 튀면 "의도치 않은 접촉"으로 판단
RETREAT_LIFT_MM = 30            # 접촉 감지 시 베이스 기준 +Z로 후퇴하는 거리(mm) - 사용자 지정값
MAX_DESCENT_RETRIES = 2         # 후퇴 후 재시도 최대 횟수 (다 써도 실패하면 호출부의 그립 재시도에 맡김)
DESCENT_POLL_SEC = 0.02


def descend_to_pick_with_reaction_guard(
    node, pick_pos, *, emergency_stop_event=None, stop_node=None, check_hw_safety_stop=None,
    vel=None, acc=None,
):
    """
    pick_pos로 하강하되, 도착 전에 반발력이 튀면(=옆에 겹쳐 있던 다른 사과를
    밀어붙인 것으로 추정) 하드웨어 안전정지로 이어지기 전에 먼저 멈추고 베이스
    기준 +Z로 RETREAT_LIFT_MM만큼 후퇴한 뒤 재시도한다.

    pick_helpers.pick_apple()이 hover_pos 이후 마지막 pick_pos 접근 구간에서
    safe_movel_fn 대신 이 함수를 호출해야 함 - 그래야 그리퍼가 아직 열려 있는
    이 구간에서 예상 밖의 저항을 별도로 감시할 수 있음.

    Args:
        node: rclpy 노드 (로그 출력용).
        pick_pos: 최종 grasp 목표 위치 (posx 값).
        emergency_stop_event/stop_node/check_hw_safety_stop: safe_motion.safe_movel과
            동일한 의미 - 하강 중에도 소프트웨어/하드웨어 비상정지를 감시함.
        vel/acc: 하강 속도/가속도 (None이면 컨트롤러 기본값).

    Returns:
        bool: True면 pick_pos에 정상 도착. False면 재시도(MAX_DESCENT_RETRIES회)를
        다 썼는데도 계속 접촉이 감지되어 포기함 - 호출부는 이 경우를 파지 실패로
        취급해 기존 그립 재시도 루프(box_sequence_test.py의 MAX_PICK_ATTEMPTS)로
        넘어가야 함.

    Raises:
        EmergencyStopError: 소프트웨어/하드웨어 비상정지가 감지되면 그대로 전파됨
        (safe_movel과 동일한 계약 - 호출부의 단일 except에서 처리됨).
    """
    import time as _time

    from DSR_ROBOT2 import (
        amovel, movel, posx, get_current_posx, get_tool_force, check_motion,
        DR_BASE, DR_MV_MOD_REL,
    )
    from apple_care_robot.safe_motion import raise_if_emergency_stop, make_hw_safety_watcher

    hw_safety_watch = make_hw_safety_watcher(
        stop_node or node, emergency_stop_event,
        check_hw_safety_stop if emergency_stop_event is not None else None,
    )

    for attempt in range(1, MAX_DESCENT_RETRIES + 2):  # 최초 시도 1회 + 재시도 MAX_DESCENT_RETRIES회
        baseline = get_tool_force(DR_BASE)
        baseline_fz = baseline[2] if isinstance(baseline, list) and len(baseline) >= 3 else 0.0

        amovel(pick_pos, vel=vel, acc=acc, ref=DR_BASE)
        _time.sleep(0.1)  # 출발 직후 관성 노이즈 스킵 (force_place.py와 동일한 이유)

        contact_detected = False
        while check_motion():
            if emergency_stop_event is not None:
                raise_if_emergency_stop(stop_node or node, emergency_stop_event)
            hw_safety_watch()

            force = get_tool_force(DR_BASE)
            if isinstance(force, list) and len(force) >= 3:
                deviation = abs(force[2] - baseline_fz)
                if deviation >= REACTIVE_CONTACT_FORCE_THRESHOLD_N:
                    contact_detected = True
                    node.get_logger().warn(
                        f'[pick] 하강 중 예상 밖 반발력 감지(변동량={deviation:.2f}N) - '
                        '겹친 사과를 눌렀을 가능성 - 하강을 중단하고 후퇴합니다.'
                    )
                    break
            _time.sleep(DESCENT_POLL_SEC)

        if not contact_detected:
            return True  # 정상 도착

        # 남은 하강 목표를 취소(현재 위치로 재지정) - force_place.py의 취소 기법과 동일.
        # 이걸 안 하면 후퇴 이동을 내려도 컨트롤러 내부에 남아있는 원래 하강 목표가
        # 다시 끼어들어 후퇴가 상쇄될 수 있음.
        stop_pos, _ = get_current_posx(DR_BASE)
        if stop_pos is not None:
            amovel(stop_pos, ref=DR_BASE)
            _time.sleep(0.1)

        if attempt > MAX_DESCENT_RETRIES:
            node.get_logger().error(
                f'[pick] 반발력 회피 재시도({MAX_DESCENT_RETRIES}회)를 다 썼는데도 계속 '
                '접촉이 감지되어 이번 pick_pos 접근을 포기합니다.'
            )
            return False

        node.get_logger().info(
            f'[pick] 베이스 기준 +Z {RETREAT_LIFT_MM}mm 후퇴 후 재시도합니다 '
            f'({attempt}/{MAX_DESCENT_RETRIES}).'
        )
        retreat_pos = posx(0, 0, RETREAT_LIFT_MM, 0, 0, 0)
        movel(retreat_pos, ref=DR_BASE, mod=DR_MV_MOD_REL)

    return False


def grasp_apple_with_force_feedback(
    node, initial_force=100, *, emergency_stop_event=None, stop_node=None, check_hw_safety_stop=None,
):
    """
    힘을 조금씩 올려가며, 손목 힘 센서로 감지된 반발력 기준으로
    딱 필요한 만큼만 사과를 잠그는 함수.

    Args:
        node: rclpy 노드 (로그 출력용)
        initial_force: 첫 시도 힘 (너무 낮으면 첫 시도에서는 못 잠글 수 있음)
        emergency_stop_event: 넘겨주면(threading.Event), 매 단계 사이에 비상정지를
            감시함 - 걸려 있으면 즉시 safe_motion.EmergencyStopError를 던짐
            (호출부가 잡아서 복구해야 함). None이면(기본값) 감시하지 않음 - 이
            호출 경로에 /robot/command 연동이 없는 호출부에서의 하위 호환을 위함.
            force_controlled_place()가 하강 중 접촉을 힘으로 감지하는 것과 마찬가지로,
            여기서는 그리퍼가 닫히는 중(팔 자체는 안 움직이지만 힘을 계속 올리는
            구간)에도 감시가 없으면 EMERGENCY_STOP이 이 함수가 끝날 때까지
            반영되지 않는 문제가 있었음 (실제로 겪은 문제).
        stop_node: emergency_stop_event를 쓸 때, motion/move_stop 서비스 클라이언트를
            만들 rclpy 노드 (보통 comm_node). 안 주면 node를 그대로 사용.
        check_hw_safety_stop: 넘겨주면(예: safe_motion.is_controller_in_hardware_safety_stop을
            노드에 바인딩한 callable) 파지 중에도 컨트롤러의 하드웨어 안전정지
            상태(펜던트 물리 비상정지 버튼)를 감시함 - /robot/command로 소프트웨어
            명령이 안 들어와도 이 구간에서 물리 버튼이 눌리면 감지됨
            (safe_motion.make_hw_safety_watcher 참고). None이면(기본값) 감시하지 않음.

    Returns:
        tuple[int, bool]: (최종 적용된 힘 값 0~GRIPPER_MAX_FORCE, 파지 성공 여부)

    Raises:
        EmergencyStopError: emergency_stop_event가 파지 도중 걸리거나(또는
            check_hw_safety_stop이 하드웨어 안전정지를 감지하면) 발생.
    """
    from DSR_ROBOT2 import get_tool_force, DR_BASE

    node.get_logger().info('실시간 힘 감지 방식으로 사과 파지 시작')

    # emergency_stop_event가 없으면(하위 호환 호출부) check_hw_safety_stop이 있어도
    # 걸어둘 이벤트 자체가 없으므로 감시하지 않음.
    hw_safety_watch = make_hw_safety_watcher(
        stop_node or node, emergency_stop_event,
        check_hw_safety_stop if emergency_stop_event is not None else None,
    )

    current_force = initial_force
    gripper_close_with_force(current_force)

    baseline = get_tool_force(DR_BASE)
    baseline_fz = baseline[2] if isinstance(baseline, list) and len(baseline) >= 3 else 0.0
    node.get_logger().info(f'[grasp] 기준 반발력: {baseline_fz:.2f}N (초기 힘={current_force})')

    for _ in range(MAX_GRASP_STEPS):
        if emergency_stop_event is not None:
            raise_if_emergency_stop(stop_node or node, emergency_stop_event)
        # 소프트웨어 명령 없이 펜던트 물리 비상정지 버튼만 눌린 경우도 잡기 위해,
        # 쓰로틀링된 하드웨어 상태 감시도 매 단계 같이 확인함.
        hw_safety_watch()

        force = get_tool_force(DR_BASE)

        if isinstance(force, list) and len(force) >= 3:
            fz = force[2]
            deviation = abs(fz - baseline_fz)

            node.get_logger().info(
                f'[grasp] 적용 힘={current_force} | 반발력 Fz={fz:.2f}N (변동량={deviation:.2f}N)'
            )

            if deviation >= GRASP_FORCE_THRESHOLD:
                node.get_logger().info(
                    f'[grasp] 파지 확인됨 (힘={current_force}). '
                    f'미끄러짐 방지를 위해 안전 여유분을 추가로 얹습니다.'
                )
                current_force = min(GRIPPER_MAX_FORCE, current_force + FORCE_STEP * SAFETY_MARGIN_STEPS)
                gripper_close_with_force(current_force)
                node.get_logger().info(f'[grasp] 최종 적용 힘 (안전 여유분 포함): {current_force}')
                return current_force, True
        else:
            node.get_logger().warn('[grasp] get_tool_force 읽기 실패 - 힘 값을 가져오지 못했습니다.')

        current_force = min(GRIPPER_MAX_FORCE, current_force + FORCE_STEP)
        gripper_close_with_force(current_force)

    node.get_logger().warn(
        f'[grasp] 최대 힘까지 다 올렸지만({current_force}) 뚜렷한 파지 신호를 감지하지 못했습니다. '
        f'사과가 없거나 위치가 어긋났을 수 있습니다.'
    )
    return current_force, False
