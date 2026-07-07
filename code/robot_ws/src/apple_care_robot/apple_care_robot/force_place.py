"""
Force-based Placing (최종 통합본 - 노션 백업용)
==================================
목적:
    하강하기 전 공중(Hover) 상태에서 그리퍼와 사과의 순수 무게(Baseline)를 먼저 측정합니다.
    이후 천천히 하강하면서 박스나 사과더미에 닿아 힘 변동량이 임계값(FORCE_THRESHOLD)을
    넘어서는 순간 즉시 로봇을 멈추고(`mwait(0)`), 안전하게 모션을 중단합니다.
"""

import time

DOWN_SPEED_VEL = 20        # 내려가는 속도 (mm/s)
DOWN_SPEED_ACC = 20        # 가속도
MAX_DOWN_DISTANCE = 150    # 접촉을 못 찾았을 때 최대 몇 mm까지 내려갈지 (안전 마진 확보)
FORCE_THRESHOLD = 4        # 힘 변화량 임계값 (N) - 필요시 8~15 사이로 튜닝
TIMEOUT_SEC = 10.0         # 힘 제어 타임아웃 시간을 10초로 여유 있게 설정


def force_controlled_place(node, current_pos, open_gripper_func=None):
    """
    현재 위치에서 수직으로 내려가다가 바닥 감지 시 멈추는 함수

    Args:
        node: rclpy 노드 (로그 출력용)
        current_pos: 하강을 시작할 현재 위치 (posx 타입)
        open_gripper_func: (하위 호환성 유지용) 메인에서 직접 처리하므로 내부 호출 최소화

    Returns:
        bool: True면 접촉 성공, False면 타임아웃 또는 실패
    """
    from DSR_ROBOT2 import (
        amovel, posx, mwait,
        task_compliance_ctrl, set_desired_force, get_tool_force,
        release_force, release_compliance_ctrl,
        DR_FC_MOD_REL, DR_BASE,
    )

    node.get_logger().info('Entering compliance mode (부드러운 접촉 감지 모드로 전환)')

    # 1) z축(위아래)만 부드럽게 설정 (Compliance 모드 진입)
    task_compliance_ctrl(stx=[3000, 3000, 500, 3000, 3000, 3000])
    time.sleep(0.2)  # 모드 전환 후 센서 및 로봇 안정화 대기

    set_desired_force(
        fd=[0, 0, -10, 0, 0, 0],
        dir=[0, 0, 1, 0, 0, 0],
        mod=DR_FC_MOD_REL,
    )

    # 2) 하강 시작 전, 정지 상태에서 깨끗한 베이스라인 무게(Fz) 측정
    node.get_logger().info('하강 전 정적 기준 무게(Baseline)를 샘플링합니다.')
    baseline_samples = []
    for _ in range(10):
        force = get_tool_force(DR_BASE)
        if isinstance(force, list) and len(force) >= 3:
            baseline_samples.append(force[2])
        time.sleep(0.01)

    if len(baseline_samples) > 0:
        baseline_fz = sum(baseline_samples) / len(baseline_samples)
        node.get_logger().info(f'측정된 기준 무게 Baseline Fz: {baseline_fz:.2f}N')
    else:
        baseline_fz = 0.0
        node.get_logger().warn('센서 값을 읽지 못해 Baseline을 0.0N으로 임시 설정합니다.')

    # 3) 목표 하강 위치 계산 및 비동기(amovel) 명령 하강 시작
    target_x, target_y, target_z, rx, ry, rz = current_pos
    target_down = posx(target_x, target_y, target_z - MAX_DOWN_DISTANCE, rx, ry, rz)
    
    amovel(target_down, vel=DOWN_SPEED_VEL, acc=DOWN_SPEED_ACC, ref=DR_BASE)
    time.sleep(0.1)  # 출발 직후 순간적인 움직임 관성 노이즈 스킵용 대기

    start_time = time.time()
    contact_detected = False
    last_log_time = start_time

    # 4) 실시간 힘 모니터링 루프
    while time.time() - start_time < TIMEOUT_SEC:
        force = get_tool_force(DR_BASE)

        if isinstance(force, list) and len(force) >= 3:
            fz = force[2]
            
            # 정지 상태 기준값(baseline_fz) 대비 힘이 얼마나 변했는지 계산
            force_deviation = abs(fz - baseline_fz)

            # 변화량이 설정한 임계값을 넘었을 때 = 물체에 부딪힘
            if force_deviation >= FORCE_THRESHOLD:
                contact_detected = True
                node.get_logger().info(f'접촉 감지 성공! 변동량: {force_deviation:.2f}N (현재 Fz={fz:.2f}N)')
                
                # [안전장치] 더 이상 짓누르지 않도록 로봇 하강 모션 즉시 인터럽트 중단
                mwait(0) 
                break

            # 0.5초 간격 디버깅 로그 출력
            now = time.time()
            if now - last_log_time >= 0.5:
                node.get_logger().info(f'[디버그] 현재 Fz={fz:.2f}N | 변동량={force_deviation:.2f}N (기준={baseline_fz:.2f}N)')
                last_log_time = now
        else:
            node.get_logger().warn('get_tool_force 호출 실패 - 데이터를 읽을 수 없습니다.')

        time.sleep(0.01)

    # 5) 안전을 위해 컴플라이언스 및 힘 제어 모드 해제
    release_force()
    release_compliance_ctrl()
    time.sleep(0.1)

    # 6) 접촉 결과 반환
    if not contact_detected:
        node.get_logger().warn(f'{TIMEOUT_SEC}초 동안 박스 접촉에 실패했습니다. 하강을 강제 중단합니다.')
        mwait(0)

    return contact_detected