"""
pick_helpers.py
==================
box_sequence_test.py(유일한 실행 진입점)가 가져다 쓰는 사과 집기 공용 함수 모듈.

원래 파일명은 apple_sorting_cycle.py였고 자체 main()으로 백엔드(/decision/result,
/robot/command) 연동 사이클을 돌렸으나, box_sequence_test.py가 이미 하드웨어
검증용으로 갖고 있던 기능(그립 재시도, 박스 점유 확인, E-STOP 복구, 벽 회피)과
겹치는 ~500줄짜리 루프를 두 파일이 따로 유지하다가, 머지할 때마다 코드가 꼬이는
문제가 반복됨(실제로 겪은 문제 - PR #18 머지 후 두 파일 다 깨져서 실행 자체가 안
됐음). main()을 없애고 box_sequence_test.py 하나로 실행 진입점을 통합하면서,
이 파일도 실제로 남은 내용(사과를 집는 함수들)에 맞춰 이름을 바꿈.

이 파일에는 두 파일이 공유하는 순수 함수만 남김:
    - pick_apple(): 사과를 집는 물리 동작 (벽 회피 + 힘 감지 파지)
    - _depth_offset_for_condition(): condition("small"/"unknown"/그 외)에 따른
      depth 보정값 선택
"""

from apple_care_robot.vision_transform import (
    SMALL_APPLE_DEPTH_OFFSET_MM, UNKNOWN_FRUIT_DEPTH_OFFSET_MM,
)

# Vision이 주는 pose는 [x, y, z] 위치만 포함하고 그리퍼 방향(rx,ry,rz)은 안 주므로,
# 사과를 향해 내려가는 고정 방향을 기본값으로 사용함.
DEFAULT_PICK_ORIENTATION = (128.8, -174.74, -139.09)


def _depth_offset_for_condition(condition):
    """
    condition("small"/"unknown"/그 외)에 따라 camera_to_base()에 넘길
    depth_offset_mm을 고름. 작은 사과와 미확인 과일은 서로 다른 상수를 쓰므로
    (크기를 전혀 모르는 unknown을 작은 사과와 같은 값으로 묶어두지 않고, 나중에
    실측으로 각자 독립적으로 튜닝할 수 있게 함) 항상 이 함수를 통해서만 결정함.
    """
    if condition == "small":
        return SMALL_APPLE_DEPTH_OFFSET_MM
    if condition == "unknown":
        return UNKNOWN_FRUIT_DEPTH_OFFSET_MM
    return None


def pick_apple(
    node, pick_pos, safe_movel_fn=None, *, condition=None,
    emergency_stop_event=None, stop_node=None, check_hw_safety_stop=None,
):
    """
    사과를 집는 함수.
    1) (트레이 벽 근처면) 대각선 경유점을 먼저 거쳐서 벽과의 충돌 없이 접근
    2) 사과 위치로 이동
    3) 실시간 힘 감지로 그리퍼 닫기 (사과 크기에 맞춰 힘 자동 조절, grasp_force.py 참고)

    집은 뒤 별도의 퇴피 동작은 하지 않음 - 호출하는 쪽(메인 루프)에서 바로
    CAMERA 위치로 이동하는 것으로 대체함.

    Args:
        condition: vision/decision이 판별한 사과 상태("small"/"unknown"/그 외).
            grasp_apple_with_force_feedback()으로 그대로 전달되어, "small"이면
            첫 파지 힘/증가 단위를 낮춰 첫 닫기부터 세게 조여 작은 사과가 튕겨
            나가는 문제를 피함 (grasp_force.py의 SMALL_APPLE_INITIAL_FORCE 참고).
        safe_movel_fn: 호출부(box_sequence_test.py의 main())가 만들어서 넘겨주는,
            comm_node/emergency_stop 이벤트가 이미 묶여있는 이동 함수
            (safe_motion.safe_movel 참고). 비상정지가 걸리면 EmergencyStopError를
            던지므로, 이 함수는 그 예외를 잡지 않고 그대로 호출부(메인 루프의
            try/except)까지 전파해야 함.
            None이면(기본값) /robot/command 연동이 없는 호출부에서의 하위 호환을
            위해 평범한(비상정지 감시 없는) movel을 씀.
        emergency_stop_event/stop_node: 그리퍼로 파지하는 구간(팔 자체는 안
            움직이지만 힘을 계속 올리는 구간)에도 비상정지를 감시하려면 넘김 -
            grasp_apple_with_force_feedback()으로 그대로 전달됨. 안 넘기면(기본값)
            이 구간에서는 감시하지 않음 (safe_movel_fn이 이동 중에는 이미 감시하므로
            완전한 사각지대는 아니지만, 파지 구간 자체는 별도로 챙겨야 함).
        check_hw_safety_stop: grasp_apple_with_force_feedback()으로 그대로 전달됨 -
            파지 구간에서도 소프트웨어 명령 없이 펜던트 물리 비상정지 버튼이
            눌리는 경우를 감지함 (safe_motion.make_hw_safety_watcher 참고).

    Returns:
        bool: 파지 성공 여부 (손목 힘 센서로 접촉/파지가 확인됐는지)
    """
    from apple_care_robot.grasp_force import (
        grasp_apple_with_force_feedback, descend_to_pick_with_reaction_guard,
    )
    from apple_care_robot.wall_avoidance import compute_wall_aware_approach

    if safe_movel_fn is None:
        from DSR_ROBOT2 import movel as safe_movel_fn
    from DSR_ROBOT2 import posx

    # pick_pos가 트레이 벽에 가까우면(wall_avoidance.py 참고) 벽 반대쪽에서
    # 비스듬히 내려가는 경유점을 먼저 거침 - 벽에서 멀면 hover_pos는 None이라
    # 기존과 동일하게 바로 pick_pos로 감.
    hover_pos, pick_pos = compute_wall_aware_approach(pick_pos, posx)
    if hover_pos is not None:
        node.get_logger().info(f'사과가 트레이 벽 근처로 판단되어 대각선 경유점을 먼저 거칩니다: {hover_pos}')
        safe_movel_fn(hover_pos)

    # 마지막 pick_pos 접근은 safe_movel_fn(위치 제어만)이 아니라 반발력을 함께
    # 감시하는 descend_to_pick_with_reaction_guard를 씀 - 실측으로 확인된 문제:
    # 겹쳐 있는 사과 중 하나로 그리퍼가 열린 채 내려가다 다른 하나를 눌러서
    # 하드웨어 안전정지가 걸리는 사례가 있었음. 이 함수가 반발력을 미리 감지해
    # 베이스 기준 +Z로 후퇴 후 재시도하므로, 여기서 실패(False)하면 그리퍼가
    # 아직 열려 있는 상태이므로 아래 grasp_apple_with_force_feedback을 부를
    # 필요 없이 곧바로 파지 실패로 취급하면 됨 (호출부의 그립 재시도 루프로 위임).
    node.get_logger().info(f'사과 집기 위치로 이동: {pick_pos}')
    reached = descend_to_pick_with_reaction_guard(
        node, pick_pos,
        emergency_stop_event=emergency_stop_event, stop_node=stop_node,
        check_hw_safety_stop=check_hw_safety_stop,
    )
    if not reached:
        return False

    node.get_logger().info('실시간 힘 감지로 그리퍼 닫기 (사과 크기에 맞춰 힘 자동 조절)')
    applied_force, picked_ok = grasp_apple_with_force_feedback(
        node, condition=condition,
        emergency_stop_event=emergency_stop_event, stop_node=stop_node,
        check_hw_safety_stop=check_hw_safety_stop,
    )
    node.get_logger().info(f'최종 적용된 파지 힘: {applied_force} (파지 성공 여부: {picked_ok})')
    if not picked_ok:
        node.get_logger().error('파지 힘 감지 실패 - 사과를 못 집었을 수 있습니다.')

    return picked_ok
