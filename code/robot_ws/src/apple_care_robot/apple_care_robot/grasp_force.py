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
from apple_care_robot.safe_motion import raise_if_emergency_stop

FORCE_STEP = 50                 # 매 단계 힘 증가량 (openclose.py 레지스터 단위, 0.1N = 5N)
GRASP_FORCE_THRESHOLD = 3.0     # 손목 반발력 변화량(N)이 이 값을 넘으면 "제대로 눌렀다/잡았다"고 판단
MAX_GRASP_STEPS = 6             # 무한 증가 방지 안전 장치 (6단계 * 50 = 300까지)

# 접촉을 감지한 힘 그대로 이동하면 관성/미끄러짐으로 놓칠 수 있어서,
# 감지 즉시 멈추는 대신 안전 여유분만큼 한 번 더 힘을 얹어줌.
SAFETY_MARGIN_STEPS = 2         # 안전 여유분 단계 수 (2단계 = +100, FORCE_STEP 기준)


def grasp_apple_with_force_feedback(
    node, initial_force=100, *, emergency_stop_event=None, stop_node=None,
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

    Returns:
        tuple[int, bool]: (최종 적용된 힘 값 0~GRIPPER_MAX_FORCE, 파지 성공 여부)

    Raises:
        EmergencyStopError: emergency_stop_event가 파지 도중 걸리면 발생.
    """
    from DSR_ROBOT2 import get_tool_force, DR_BASE

    node.get_logger().info('실시간 힘 감지 방식으로 사과 파지 시작')

    current_force = initial_force
    gripper_close_with_force(current_force)

    baseline = get_tool_force(DR_BASE)
    baseline_fz = baseline[2] if isinstance(baseline, list) and len(baseline) >= 3 else 0.0
    node.get_logger().info(f'[grasp] 기준 반발력: {baseline_fz:.2f}N (초기 힘={current_force})')

    for _ in range(MAX_GRASP_STEPS):
        if emergency_stop_event is not None:
            raise_if_emergency_stop(stop_node or node, emergency_stop_event)

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
