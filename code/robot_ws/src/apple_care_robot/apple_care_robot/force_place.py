"""
Force-based Placing (최종 통합본 - 노션 백업용)
==================================
목적:
    하강하기 전 공중(Hover) 상태에서 그리퍼와 사과의 순수 무게(Baseline)를 먼저 측정합니다.
    이후 천천히 하강하면서 박스나 사과더미에 닿아 힘 변동량이 임계값(FORCE_THRESHOLD)을
    넘어서는 순간, 현재 위치로 재지정(amovel)해서 남은 하강 목표를 취소한 뒤
    안전하게 컴플라이언스/힘 제어 모드를 해제합니다.
"""

import time

from apple_care_robot.safe_motion import raise_if_emergency_stop, make_hw_safety_watcher

DOWN_SPEED_VEL = 30        # 내려가는 속도 (mm/s)
DOWN_SPEED_ACC = 20        # 가속도
MAX_DOWN_DISTANCE = 150    # 접촉을 못 찾았을 때 최대 몇 mm까지 내려갈지 (안전 마진 확보)
FORCE_THRESHOLD = 4        # 힘 변화량 임계값 (N) - 필요시 8~15 사이로 튜닝

# 이미 사과가 있는 박스는 새 사과가 기존 사과에 닿을 때 변동량이 더 작게 나와서
# (실측 약 3N) 기본 FORCE_THRESHOLD(4N)로는 접촉 감지에 실패했음. 그래서 이미
# 사과가 있는 박스 전용으로 더 낮은 임계값을 따로 둠.
EXISTING_APPLE_FORCE_THRESHOLD = 3  # N

# 이미 사과가 있다고 알려진 박스에 접근할 때 쓰는 속도 (평소보다 훨씬 느리게).
CAREFUL_APPROACH_VEL = 15  # mm/s
CAREFUL_APPROACH_ACC = 15

# 박스에 이미 사과가 있으면, 원래 hover 좌표(box_pos, 빈 박스 기준으로 잡은 높이)보다
# 사과 더미 꼭대기가 더 높이 나와 있을 수 있음. 그 상태로 box_pos까지 블라인드로
# movel 해버리면 힘제어가 걸리기도 전에 그대로 부딪힘. 그래서 이미 사과가 있는
# 박스는 box_pos보다 이만큼 더 높은 위치까지만 movel로 접근하고, 그 지점부터
# force_controlled_place로 힘제어 하강을 시작해서 사과 더미와의 접촉도 힘으로 감지함.
EXISTING_APPLE_HOVER_CLEARANCE = 40  # mm

# 타임아웃은 항상 "최대 하강 시간"보다 여유 있게 잡아야 함.
# (DOWN_SPEED_VEL/MAX_DOWN_DISTANCE를 바꿔도 자동으로 맞춰지도록 계산식으로 둠)
_TRAVEL_TIME = MAX_DOWN_DISTANCE / DOWN_SPEED_VEL
TIMEOUT_SEC = _TRAVEL_TIME + 2.5

# 실제로는 바닥/사과에 닿아서 더 이상 못 내려가고 있는데도, 힘 변화량이
# force_threshold를 못 넘어서(사과가 물러서 충격이 완만하거나 센서 노이즈에
# 묻히는 경우) 계속 대기하다 결국 타임아웃으로 실패 처리되는 문제가 있었음.
# 그래서 힘 조건과 별개로 "z가 이만큼(STALL_TIME_SEC) 안 움직이면 그냥 접촉된
# 것으로 본다"는 조건을 추가함 - 힘제어 하강 중에 z가 안 움직인다는 것 자체가
# 이미 뭔가에 막혀 있다는 뜻이기 때문.
STALL_TIME_SEC = 2.0        # z가 이 시간(초) 이상 거의 안 움직이면 접촉으로 간주
STALL_Z_EPSILON_MM = 0.5    # 이 이하의 z 변화는 "안 움직임"으로 취급 (컨트롤러/센서 노이즈 감안)

# release_force()/release_compliance_ctrl()도 amovel()처럼 ret(0=성공/-1=실패)을
# 반환하는데 지금까지 이 값을 버렸음 - E-STOP 직후처럼 컨트롤러가 막 정지 명령을
# 받은 시점에 이 해제 호출이 실패하면, task_compliance_ctrl(stx=[...,500,...])로
# 걸어둔 Z축 컴플라이언스(강성 500, 다른 축의 1/6)가 안 풀린 채로 남을 수 있음 -
# 그 상태에서 나가는 movel(Z 방향 이동)은 물렁한 Z축에 흡수되어 명령상 성공
# (ret=0)으로 와도 실제로는 거의 안 움직이는 문제로 이어질 수 있어서(E-STOP 복구의
# lift 단계가 반복 재현한 증상과 일치) 재시도함.
RELEASE_RETRY_COUNT = 3


def force_controlled_place(
    node, current_pos, force_threshold=FORCE_THRESHOLD,
    *, emergency_stop_event=None, stop_node=None, check_hw_safety_stop=None,
):
    """
    현재 위치에서 수직으로 내려가다가 바닥 감지 시 멈추는 함수

    Args:
        node: rclpy 노드 (로그 출력용)
        current_pos: 하강을 시작할 현재 위치 (posx 타입)
        force_threshold: 접촉으로 판단할 힘 변화량 임계값 (N).
            이미 사과가 있는 박스는 EXISTING_APPLE_FORCE_THRESHOLD를 넘겨서 호출.
        emergency_stop_event: 넘겨주면(threading.Event), 하강 힘제어 루프 중에도
            비상정지를 감시함 - 걸려 있으면 즉시 하드웨어 정지 후
            safe_motion.EmergencyStopError를 던짐 (호출부가 잡아서 복구해야 함).
            None이면(기본값) 기존과 동일하게 비상정지를 감시하지 않음 - 이 호출
            경로에 /robot/command 연동이 없는 호출부에서의 하위 호환을 위함.
        stop_node: emergency_stop_event를 쓸 때, motion/move_stop 서비스 클라이언트를
            만들 rclpy 노드 (보통 comm_node). 안 주면 node를 그대로 사용.
        check_hw_safety_stop: 넘겨주면(예: safe_motion.is_controller_in_hardware_safety_stop을
            노드에 바인딩한 callable) 힘제어 하강 중에도 컨트롤러의 하드웨어
            안전정지 상태(펜던트 물리 비상정지 버튼)를 감시함 - /robot/command로
            소프트웨어 명령이 안 들어와도 이 구간에서 물리 버튼이 눌리면 감지됨
            (safe_motion.make_hw_safety_watcher 참고). None이면(기본값) 감시하지 않음.

    Returns:
        bool: True면 접촉 성공, False면 타임아웃 또는 실패

    Raises:
        EmergencyStopError: emergency_stop_event가 힘제어 도중 걸리거나(또는
            check_hw_safety_stop이 하드웨어 안전정지를 감지하면) 발생. 아래
            finally 블록의 예외 안전 처리는 그대로 실행되므로, 하강 목표
            취소(amovel)와 컴플라이언스 모드 해제 둘 다 이 경우에도 정상 수행됨.
    """
    from DSR_ROBOT2 import (
        amovel, posx, get_current_posx,
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

    # emergency_stop_event가 없으면(하위 호환 호출부) check_hw_safety_stop이 있어도
    # 걸어둘 이벤트 자체가 없으므로 감시하지 않음 - make_hw_safety_watcher는 감지 시
    # emergency_stop_event.set()을 호출하므로 None이면 AttributeError가 남.
    hw_safety_watch = make_hw_safety_watcher(
        stop_node or node, emergency_stop_event,
        check_hw_safety_stop if emergency_stop_event is not None else None,
    )

    start_time = time.time()
    contact_detected = False
    last_log_time = start_time

    # z-스톨(안 움직임) 감지용 기준값. 처음엔 아직 기준 z를 못 정했으니 None으로
    # 시작하고, 루프 첫 iteration에서 바로 채워짐.
    stall_ref_z = None
    stall_ref_time = start_time

    # 이 while 루프는 반드시 try/finally로 감싸야 함: 도중에 예외/Ctrl+C로
    # 빠져나가도 release_force()/release_compliance_ctrl()이 호출되지 않으면
    # 로봇 컨트롤러가 힘/컴플라이언스 제어 상태로 남아있게 되는 안전 문제가 있음.
    try:
        # 4) 실시간 힘 모니터링 루프
        while time.time() - start_time < TIMEOUT_SEC:
            force = get_tool_force(DR_BASE)

            if isinstance(force, list) and len(force) >= 3:
                fz = force[2]

                # 정지 상태 기준값(baseline_fz) 대비 힘이 얼마나 변했는지 계산
                force_deviation = abs(fz - baseline_fz)

                # 변화량이 설정한 임계값을 넘었을 때 = 물체에 부딪힘
                if force_deviation >= force_threshold:
                    contact_detected = True
                    node.get_logger().info(f'접촉 감지 성공! 변동량: {force_deviation:.2f}N (현재 Fz={fz:.2f}N)')
                    break

                # 0.5초 간격 디버깅 로그 출력
                now = time.time()
                if now - last_log_time >= 0.5:
                    node.get_logger().info(f'[디버그] 현재 Fz={fz:.2f}N | 변동량={force_deviation:.2f}N (기준={baseline_fz:.2f}N)')
                    last_log_time = now
            else:
                node.get_logger().warn('get_tool_force 호출 실패 - 데이터를 읽을 수 없습니다.')

            # 힘 임계값과 별개로, z 자체가 STALL_TIME_SEC 이상 거의 안 움직이면
            # (=뭔가에 막혀서 더 이상 못 내려가는 중) 접촉으로 간주하고 종료.
            # get_current_posx()는 컨트롤러가 순간적으로 빈 응답을 주면 내부적으로
            # IndexError를 던지는 결함이 있음(Humble/Jazzy 브랜치 동일) - 이 tick의
            # 스톨 체크만 건너뛰도록 방어.
            try:
                current_pos_now, _ = get_current_posx(DR_BASE)
            except Exception as e:
                node.get_logger().warn(f'get_current_posx 호출 실패 (이번 tick 스톨 체크 건너뜀): {e}')
                current_pos_now = None
            if current_pos_now is not None:
                current_z = current_pos_now[2]
                now = time.time()
                if stall_ref_z is None or abs(current_z - stall_ref_z) > STALL_Z_EPSILON_MM:
                    # z가 움직였으니(또는 첫 샘플이니) 스톨 기준을 지금 위치/시각으로 갱신
                    stall_ref_z = current_z
                    stall_ref_time = now
                elif now - stall_ref_time >= STALL_TIME_SEC:
                    contact_detected = True
                    node.get_logger().info(
                        f'힘 변동량은 임계값 미달이지만 z가 {STALL_TIME_SEC:.1f}초 이상 '
                        f'안 움직여 접촉으로 간주합니다 (z={current_z:.2f}mm).'
                    )
                    break

            # 하강 도중에도 비상정지를 감시함 (emergency_stop_event가 넘어온
            # 경우에만). 걸리면 raise_if_emergency_stop이 실제 하드웨어 정지를
            # 요청하고 EmergencyStopError를 던짐 - 아래 finally가 그래도 실행되어
            # 하강 목표 취소/모드 해제는 정상적으로 이뤄짐.
            if emergency_stop_event is not None:
                raise_if_emergency_stop(stop_node or node, emergency_stop_event)
            # 소프트웨어 명령 없이 펜던트 물리 비상정지 버튼만 눌린 경우도
            # 잡기 위해, 쓰로틀링된 하드웨어 상태 감시도 같이 확인함.
            hw_safety_watch()

            time.sleep(0.01)
    finally:
        # 주의: 접촉 감지든 타임아웃이든, 위의 amovel(target_down, ...)은 여전히
        # MAX_DOWN_DISTANCE 하강을 목표로 백그라운드에서 실행 중인 상태임.
        # 이 상태로 바로 release_force/release_compliance_ctrl을 부르면 Z축이
        # 다시 뻣뻣한 위치 제어로 돌아가면서 이미 멈췄어야 할 지점을 넘어 원래
        # 목표(MAX_DOWN_DISTANCE 하강)까지 마저 가려고 해서 충돌/급정지로 이어질 수 있음.
        # 그래서 release 전에 "지금 있는 자리로 가라"는 짧은 모션을 먼저 던져서
        # 남아있던 하강 목표를 덮어씀(취소).
        try:
            stop_pos, _ = get_current_posx(DR_BASE)
            if stop_pos is not None:
                amovel(stop_pos, vel=DOWN_SPEED_VEL, acc=DOWN_SPEED_ACC, ref=DR_BASE)
                time.sleep(0.1)
        except Exception as e:
            node.get_logger().warn(f'현재 위치 취소용 get_current_posx/amovel 실패 (무시하고 release 진행): {e}')

        # 5) 안전을 위해 컴플라이언스 및 힘 제어 모드 해제 - 실패하면 재시도
        # (RELEASE_RETRY_COUNT 위 설명 참고).
        for attempt in range(1, RELEASE_RETRY_COUNT + 1):
            force_ret = release_force(0.2)
            compliance_ret = release_compliance_ctrl()
            if force_ret == 0 and compliance_ret == 0:
                break
            node.get_logger().error(
                f'컴플라이언스/힘 제어 해제 실패(시도 {attempt}/{RELEASE_RETRY_COUNT}): '
                f'release_force={force_ret}, release_compliance_ctrl={compliance_ret} - 재시도합니다.'
            )
            time.sleep(0.1)
        else:
            node.get_logger().error(
                '컴플라이언스/힘 제어 해제가 끝내 실패했습니다 - 로봇이 여전히 Z축 '
                '컴플라이언스 상태로 남아있을 수 있습니다. 이후 이동이 명령상 성공해도 '
                '실제로는 거의 안 움직이면 이게 원인일 가능성이 높습니다.'
            )
        time.sleep(0.1)

    # 6) 접촉 결과 반환
    if not contact_detected:
        node.get_logger().warn(f'{TIMEOUT_SEC:.1f}초 동안 박스 접촉에 실패했습니다. 하강을 강제 중단합니다.')

    return contact_detected