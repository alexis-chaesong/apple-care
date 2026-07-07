"""
Force-based Placing
=====================

목적:
    박스(b1~b4) 안에 사과가 몇 개 쌓여있는지 매번 다르기 때문에,
    "정확히 z=00.0mm까지 내려가라" 같은 고정 좌표로는 안전하게 놓을 수 없다.
    대신 그리퍼를 천천히 내리다가, 반발력이 일정 값을 넘는 순간(=뭔가에 닿음)
    멈추고 그 자리에서 놓는 방식(Force Control / Compliance Control)을 사용한다.

주의 (중요):
    - task_compliance_ctrl / set_desired_force / get_tool_force /
      release_force / release_compliance_ctrl 이 5개 함수 이름과 파라미터 형식은
      두산 SDK(dsr_common2) 버전에 따라 조금씩 다를 수 있습니다.
      실제로 맞는지 아래 방법으로 꼭 먼저 확인하세요:

          python3 -c "from DSR_ROBOT2 import *; help(task_compliance_ctrl)"

    - 처음 테스트할 땐 반드시:
        1) 로봇 팔 아래에 손/물건 치우기
        2) 힘 임계값(FORCE_THRESHOLD)을 낮게 잡고 시작
        3) 비상정지 버튼 손 닿는 곳에 두기
"""

import time

DOWN_SPEED_VEL = 20        # 내려가는 속도 (mm/s) - 처음엔 느리게
DOWN_SPEED_ACC = 20        # 가속도
MAX_DOWN_DISTANCE = 100    # 접촉을 못 찾았을 때 최대 몇 mm까지 내려갈지 (안전장치)
FORCE_THRESHOLD = 10       # 이 값(N) 이상 반발력이 느껴지면 "닿았다"고 판단 - 처음엔 작게 시작
TIMEOUT_SEC = 5.0          # 접촉을 끝까지 못 찾으면 포기하는 시간 (안전장치)


def force_controlled_place(node, current_pos):
    """
    현재 위치(사과를 쥔 채로 박스 위 hover 지점)에서 수직으로 천천히 내려가다가,
    바닥/사과더미에 닿는 순간을 감지해서 멈추고 그 자리에서 놓을 준비를 하는 함수.

    Args:
        node: rclpy 노드 (로그 출력용)
        current_pos: posx(x, y, z, rx, ry, rz) - 하강을 시작할 현재 위치 (보통 박스 hover 지점)

    Returns:
        bool: True면 접촉 감지 성공, False면 타임아웃/최대거리까지 못 닿음
    """
    from DSR_ROBOT2 import (
        amovel, posx,
        task_compliance_ctrl, set_desired_force, get_tool_force,
        release_force, release_compliance_ctrl,
        DR_FC_MOD_REL, DR_BASE,
    )

    node.get_logger().info('Entering compliance mode (부드러운 접촉 감지 모드로 전환)')

    # z축(위아래)만 부드럽게, 나머지는 평소처럼 뻣뻣하게
    # TODO: stiffness 숫자는 예시값입니다. 로봇/그리퍼 무게에 맞게 조정 필요.
    task_compliance_ctrl(stx=[3000, 3000, 500, 3000, 3000, 3000])
    time.sleep(0.1)

    set_desired_force(
        fd=[0, 0, -FORCE_THRESHOLD, 0, 0, 0],
        dir=[0, 0, 1, 0, 0, 0],
        mod=DR_FC_MOD_REL,
    )

    target_x, target_y, target_z, rx, ry, rz = current_pos
    target_down = posx(target_x, target_y, target_z - MAX_DOWN_DISTANCE, rx, ry, rz)
    amovel(target_down, vel=DOWN_SPEED_VEL, acc=DOWN_SPEED_ACC, ref=DR_BASE)

    start_time = time.time()
    contact_detected = False
    last_log_time = start_time
    baseline_fz = None  # 그리퍼/사과 무게로 인한 정지 상태 오프셋 (접촉 전 기준값)

    # 주의: check_force_condition()은 DSR_ROBOT2 관례상 "성공 시 0, 실패 시 -1"을
    # 반환하는 C 스타일 리턴코드라서 `if check_force_condition(...)`처럼 바로 진리값으로
    # 쓰면 0(성공)이 파이썬에서는 False로 취급되어 항상 타임아웃까지 못 빠져나오는 문제가
    # 있었음. get_tool_force로 직접 힘을 읽어 파이썬에서 비교하는 방식으로 변경.
    while time.time() - start_time < TIMEOUT_SEC:
        force = get_tool_force(DR_BASE)

        if isinstance(force, list) and len(force) >= 3:
            fz = force[2]
            if baseline_fz is None:
                baseline_fz = fz

            if abs(fz - baseline_fz) >= FORCE_THRESHOLD:
                contact_detected = True
                node.get_logger().info(f'접촉 감지됨 (contact detected), Fz={fz:.2f}N')
                break

            now = time.time()
            if now - last_log_time >= 0.5:
                node.get_logger().info(f'[force debug] Fz={fz:.2f}N (baseline={baseline_fz:.2f}N)')
                last_log_time = now
        else:
            node.get_logger().warn('get_tool_force 호출 실패 - 힘 값을 읽지 못했습니다.')

        time.sleep(0.01)

    release_force()
    release_compliance_ctrl()

    if not contact_detected:
        node.get_logger().warn(
            f'{TIMEOUT_SEC}초 동안 접촉을 감지하지 못했습니다. '
            f'박스가 예상보다 훨씬 비어있거나 힘 임계값/방향 설정을 확인해보세요.'
        )

    return contact_detected