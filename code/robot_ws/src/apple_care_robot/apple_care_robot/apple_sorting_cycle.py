"""
Apple Sorting Cycle
=====================

목적:
    "카메라로 사과 감지 -> 분류 결과에 따라 b1~b4로 이동 -> 집기 -> 힘제어 하강으로
    내려놓기 -> 경유점 거쳐 복귀" 전체 사이클을 구현.

백엔드 연동 (apple-care/code/backend 참고):
    - /decision/result (String/JSON) 구독: 백엔드의 decision_planner.decide()가
      action="execute"로 판단한 결과가 여기로 들어옴.
      형식: {"fruit": str, "destination": "normal_box"|"processing_box"|
             "discard_box"|"ugly_box", "pose": [x,y,z] 또는 None, "reason": str}
      Vision이 아직 판정을 못 내렸거나 확신이 낮은 사과는 ask_human 경로로 빠져서
      여기로 아예 들어오지 않음 (백엔드 HITL 흐름, 로봇 쪽은 신경 쓸 필요 없음).

      ⚠️ pose는 로봇 베이스 좌표계가 아니라 "카메라 좌표계" 좌표임
      (obj_detection/depth_utils.py의 _pixel_to_camera_coords가 계산한 값이
      vision_bridge.py -> decision_planner.py를 그대로 통과해서 옴). 그래서 바로
      posx()에 넣으면 안 되고, vision_transform.camera_to_base()로 로봇 베이스
      좌표계로 변환한 뒤 써야 함 (cobot2_ws/pick_and_place_voice의
      robot_control.py에 있는 transform_to_base와 동일한 원리 - 자세한 설명은
      vision_transform.py 참고). 변환에는 "카메라가 그 사과를 봤을 때의 로봇
      pose"가 필요한데, 로봇이 CAMERA 위치에 서 있는 동안 vision이 감지하고
      decision이 내려오므로 여기서는 decision을 받은 시점의 get_current_posx()를
      그대로 그 값으로 사용함.
    - /robot/command (String/JSON) 구독: {"command": "START"|"EMERGENCY_STOP"|...}
      START가 와야 사이클을 시작하고, EMERGENCY_STOP이 오면 사이클을 중단함.
      HOLD/RESUME/MANUAL_PAUSE로 사이클 도중에 멈췄다가 재개하는 것은 이번
      연동 범위에 포함하지 않음 (수신은 하되 로그만 남기고 무시).
    - StatusBus로 /robot/process_state, /robot/motion_status, /gripper/status,
      /robot/safety_event를 발행해서 백엔드/HMI가 현재 상태를 알 수 있게 함.

    이 스크립트는 이제 정해진 개수만 처리하고 끝나는 게 아니라, /decision/result가
    들어오는 대로 계속 처리하는 상시 서비스 루프로 동작함 (EMERGENCY_STOP 전까지).

박스 매핑 (destination -> 실제 박스, TODO: 실제 배치와 일치하는지 반드시 확인 필요):
    normal_box     -> b1 (경유점: way1)
    processing_box -> b2 (경유점: way1)
    ugly_box       -> b3 (경유점: way1)
    discard_box    -> b4 (경유점: way2)

동작 순서 / HOME 관련:
    - HOME은 사이클 전체에서 맨 처음(1회)과 맨 마지막(1회)에만 거침.
      그 사이에는 매 사과마다 CAMERA -> (집기) -> CAMERA -> 경유점 -> 박스
      -> 경유점 -> CAMERA 순으로만 돌고, 박스로 가기 전/집은 뒤에 HOME을
      다시 거치지 않음.
    - 집은 직후 별도의 퇴피 동작 없이 바로 CAMERA 위치로 이동함.

안전 원칙:
    - 박스가 비어있어도 항상 힘제어(force_controlled_place)로 하강함.
      (힘제어 없는 고속 하강은 비상정지를 유발해서 제외함)
    - 이미 사과가 있다고 알려진 박스(현재 b1/normal_box)는, 사과 더미 꼭대기가 원래
      hover 좌표(box_pos)보다 높이 나와 있을 수 있어서 box_pos까지 블라인드로
      내려가지 않음. 대신 box_pos보다 EXISTING_APPLE_HOVER_CLEARANCE만큼 더 높은
      위치까지만 조심히(CAREFUL_APPROACH_VEL/ACC) movel로 접근하고, 그 지점부터
      force_controlled_place로 힘제어 하강을 시작해서 사과 더미와의 접촉도 힘으로 감지함.
    - 박스에 놓은 뒤(성공/실패 무관)에는 매번 웨이포인트 -> 카메라 위치 순서로 복귀함.

TODO:
    - DESTINATION_TO_BOX의 박스 배치(b1~b4가 실제로 어느 destination에 대응하는지)
      확인 필요 - 지금은 임시로 위 순서대로 매핑해둠.
    - DESTINATION_TO_BOX의 has_existing_apple 플래그: 카메라가 박스 내부를 보고
      "이미 사과가 있는지" 판단한 결과로 교체 (지금은 b1만 하드코딩으로 True)
    - pose가 없을 때 쓰는 FALLBACK_PICK_POS_XYZ / DEFAULT_PICK_ORIENTATION:
      실제 pick 좌표/그리퍼 방향으로 교체 필요 (지금은 placeholder)
"""

import json
import queue
import threading

import rclpy
import DR_init
from std_msgs.msg import String

from apple_care_robot.force_place import (
    force_controlled_place,
    CAREFUL_APPROACH_VEL, CAREFUL_APPROACH_ACC,
    EXISTING_APPLE_HOVER_CLEARANCE, EXISTING_APPLE_FORCE_THRESHOLD,
)
from apple_care_robot.openclose import gripper_open
from apple_care_robot.status_bus import StatusBus
from apple_care_robot.vision_transform import camera_to_base

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

TOPIC_DECISION_RESULT = "/decision/result"
TOPIC_ROBOT_COMMAND = "/robot/command"

# Vision이 주는 pose는 [x, y, z] 위치만 포함하고 그리퍼 방향(rx,ry,rz)은 안 주므로,
# 사과를 향해 내려가는 고정 방향을 기본값으로 사용함.
DEFAULT_PICK_ORIENTATION = (128.8, -174.74, -139.09)

# decision.pose가 None으로 온 경우(비전이 좌표를 못 준 경우)에 쓰는 임시 pick 좌표.
FALLBACK_PICK_POS_XYZ = (492.0, 13.48, 23.39)

DESTINATION_TO_BOX = {}  # main()에서 채움: destination -> (box_name, box_pos, way_pos, has_existing_apple)


def pick_apple(node, pick_pos):
    """
    사과를 집는 함수.
    1) 사과 위치로 이동
    2) 실시간 힘 감지로 그리퍼 닫기 (사과 크기에 맞춰 힘 자동 조절, grasp_force.py 참고)

    집은 뒤 별도의 퇴피 동작은 하지 않음 - 호출하는 쪽(메인 루프)에서 바로
    CAMERA 위치로 이동하는 것으로 대체함.

    Returns:
        bool: 파지 성공 여부 (손목 힘 센서로 접촉/파지가 확인됐는지)
    """
    from DSR_ROBOT2 import movel
    from apple_care_robot.grasp_force import grasp_apple_with_force_feedback

    node.get_logger().info(f'사과 집기 위치로 이동: {pick_pos}')
    movel(pick_pos)

    node.get_logger().info('실시간 힘 감지로 그리퍼 닫기 (사과 크기에 맞춰 힘 자동 조절)')
    applied_force, picked_ok = grasp_apple_with_force_feedback(node)
    node.get_logger().info(f'최종 적용된 파지 힘: {applied_force} (파지 성공 여부: {picked_ok})')
    if not picked_ok:
        node.get_logger().error('파지 힘 감지 실패 - 사과를 못 집었을 수 있습니다.')

    return picked_ok


def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node("apple_sorting_cycle", namespace=ROBOT_ID)
    DR_init.__dsr__node = node

    # DSR_ROBOT2의 movel/wait 등은 내부적으로 이 node를 대상으로
    # rclpy.spin_until_future_complete(node, future)를 계속 호출함. 같은 node를
    # 백그라운드 스레드에서 또 rclpy.spin()으로 돌리면 동일 노드를 두 곳에서
    # 동시에 spin하게 되어 충돌/누락이 생길 수 있음. 그래서 /decision/result,
    # /robot/command 구독과 StatusBus 발행은 완전히 별개인 comm_node로 분리함.
    comm_node = rclpy.create_node("apple_sorting_cycle_comm", namespace=ROBOT_ID)

    status_bus = StatusBus(comm_node)
    status_bus.set_state("INIT")

    from DSR_ROBOT2 import (
        movel, movej, posx, posj, wait,
        set_velx, set_accx, set_velj, set_accj, get_current_posx,
    )

    # 좌표 정의
    HOME = posj(0, 0, 90, 0, 90, 0)
    CAMERA = posx(458.01, -112.234, 246.969, 158.276, 176.847, 157.433)

    WAY1 = posx(446.54, 244.89, 206.64, 79.38, 177.62, 172.06)
    WAY2 = posx(740.15, 189.44, 182.73, 170.37, -162.55, -103.95)

    B1 = posx(271.57, 248.16, 47.72, 90.88, -177.84, 89.87)
    B2 = posx(466.13, 239.03, 38.71, 39.87, -175.85, 55.2)
    B3 = posx(635.23, 246.09, 34.71, 169.25, -179.16, -169.02)
    B4 = posx(786.27, 185.37, 36.23, 8.77, 158.32, 91.18)

    # destination -> (박스 이름, 박스 좌표, 경유점, 기존 사과 존재 여부)
    global DESTINATION_TO_BOX
    DESTINATION_TO_BOX = {
        "normal_box":     ("b1", B1, WAY1, True),   # 이미 사과가 있다고 가정 -> hover 진입을 조심히
        "processing_box": ("b2", B2, WAY1, False),
        "ugly_box":       ("b3", B3, WAY1, False),
        "discard_box":    ("b4", B4, WAY2, False),
    }

    set_velx(45, 30)
    set_accx(45, 30)
    set_velj(30)
    set_accj(30)

    # ------------------------------------------------------------------
    # 백엔드 연동: /decision/result 구독, /robot/command 구독
    # ------------------------------------------------------------------
    decision_queue: "queue.Queue[dict]" = queue.Queue()
    started = threading.Event()
    emergency_stop = threading.Event()

    def decision_result_callback(msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            node.get_logger().error(f"{TOPIC_DECISION_RESULT} JSON 파싱 실패: {msg.data}")
            return
        decision_queue.put(data)

    def robot_command_callback(msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            node.get_logger().error(f"{TOPIC_ROBOT_COMMAND} JSON 파싱 실패: {msg.data}")
            return

        command = data.get("command")
        if command == "START":
            node.get_logger().info(f"[{TOPIC_ROBOT_COMMAND}] START 수신")
            started.set()
        elif command == "EMERGENCY_STOP":
            node.get_logger().error(f"[{TOPIC_ROBOT_COMMAND}] EMERGENCY_STOP 수신")
            emergency_stop.set()
        else:
            # HOLD/RESUME/MANUAL_PAUSE 등 사이클 도중 정지/재개는 이번 연동 범위 밖.
            node.get_logger().warn(f"[{TOPIC_ROBOT_COMMAND}] '{command}' 명령은 아직 처리하지 않습니다.")

    comm_node.create_subscription(String, TOPIC_DECISION_RESULT, decision_result_callback, 10)
    comm_node.create_subscription(String, TOPIC_ROBOT_COMMAND, robot_command_callback, 10)

    # movel/wait이 메인 스레드를 블로킹하는 동안에도 위 구독 콜백이 처리되도록
    # comm_node를 별도 스레드에서 spin (DSR 제어용 node는 건드리지 않음 - 위 주석 참고).
    spin_thread = threading.Thread(target=rclpy.spin, args=(comm_node,), daemon=True)
    spin_thread.start()

    status_bus.set_state("READY")
    node.get_logger().info("=== Apple sorting cycle: /robot/command의 START 대기 중 ===")
    started.wait()

    node.get_logger().info("=== Apple sorting cycle start ===")
    status_bus.set_state("MOVING")

    # HOME은 사이클 전체에서 맨 처음(여기)과 맨 마지막(루프를 빠져나간 뒤)에만 거침.
    # 그 사이에는 CAMERA <-> WAY <-> BOX 사이만 순환함.
    node.get_logger().info("Move to HOME (최초 1회)")
    movej(HOME)
    wait(0.3)

    node.get_logger().info("Move to CAMERA position")
    movel(CAMERA)
    wait(0.3)

    while rclpy.ok():
        if emergency_stop.is_set():
            node.get_logger().error("EMERGENCY_STOP 상태 - 사이클을 중단합니다.")
            status_bus.set_state("ERROR", "emergency_stop")
            break

        try:
            decision = decision_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        fruit = decision.get("fruit")
        destination = decision.get("destination")
        pose = decision.get("pose")

        if destination not in DESTINATION_TO_BOX:
            node.get_logger().error(f"알 수 없는 destination: {destination} - 이 사과는 건너뜁니다.")
            continue

        box_name, box_pos, way_pos, has_existing_apple = DESTINATION_TO_BOX[destination]

        if pose:
            # pose는 카메라 좌표계 -> 로봇이 현재 서 있는(=CAMERA에서 감지된
            # 순간의) pose를 기준으로 로봇 베이스 좌표계로 변환해야 movel에 그대로
            # 쓸 수 있음. 자세한 설명은 vision_transform.py 상단 docstring 참고.
            camera_pose_at_detection = get_current_posx()[0]
            bx, by, bz = camera_to_base(pose, camera_pose_at_detection)
            node.get_logger().info(
                f"Vision 좌표 변환: camera={pose} -> base=({bx:.2f}, {by:.2f}, {bz:.2f})"
            )
            pick_pos = posx(bx, by, bz, *DEFAULT_PICK_ORIENTATION)
        else:
            node.get_logger().warn("decision.pose가 없어 임시 pick 좌표를 사용합니다.")
            pick_pos = posx(*FALLBACK_PICK_POS_XYZ, *DEFAULT_PICK_ORIENTATION)

        node.get_logger().info(f"--- 사과: fruit={fruit} destination={destination} -> {box_name} ---")

        # 1) 사과 집기 (이미 CAMERA 위치에 있는 상태 - 최초 진입 전 또는 이전
        #    사이클의 마지막 단계에서 CAMERA로 이동해둔 상태에서 바로 이어짐)
        status_bus.set_motion("PICKING", fruit)
        picked_ok = pick_apple(node, pick_pos)
        status_bus.publish_gripper_status(picked_ok)
        if not picked_ok:
            status_bus.publish_safety("ERR_PICK", f"{fruit} 파지(힘 감지) 실패")
        wait(0.3)

        # 2) 집은 직후 별도 퇴피 없이 바로 CAMERA 위치로 복귀
        node.get_logger().info("Move to CAMERA position (집은 직후 바로 복귀)")
        movel(CAMERA)
        wait(0.3)

        # 3) 목적지 박스로 가기 전, 반드시 해당 박스의 경유점을 거침
        node.get_logger().info(f"Move to way point before {box_name}")
        movel(way_pos)
        wait(0.3)

        status_bus.set_motion("PLACING", box_name)

        if has_existing_apple:
            # 이미 사과가 있으면 사과 더미 꼭대기가 원래 hover 좌표(box_pos)보다
            # 높이 나와 있을 수 있어, box_pos까지 블라인드로 내려가면 힘제어가
            # 걸리기도 전에 부딪힐 수 있음. 그래서 box_pos보다 더 높은 위치까지만
            # 조심히 movel로 접근하고, 그 지점부터 힘제어로 하강을 시작함.
            hx, hy, hz, hrx, hry, hrz = box_pos
            safe_hover_pos = posx(hx, hy, hz + EXISTING_APPLE_HOVER_CLEARANCE, hrx, hry, hrz)
            node.get_logger().info(
                f"{box_name}: 이미 사과가 있다고 가정 - 원래 hover보다 "
                f"{EXISTING_APPLE_HOVER_CLEARANCE}mm 위에서부터 조심히 접근합니다."
            )
            movel(safe_hover_pos, vel=CAREFUL_APPROACH_VEL, acc=CAREFUL_APPROACH_ACC)
            wait(0.3)
            contact_ok = force_controlled_place(
                node, safe_hover_pos, force_threshold=EXISTING_APPLE_FORCE_THRESHOLD
            )
        else:
            node.get_logger().info(f"Move to {box_name} (hover above box)")
            movel(box_pos)
            wait(0.3)
            contact_ok = force_controlled_place(node, box_pos)

        if contact_ok:
            node.get_logger().info(f"Apple contact confirmed at {box_name}. Opening gripper.")
            opened_ok = gripper_open()
            status_bus.publish_gripper_status(not opened_ok)
            if not opened_ok:
                node.get_logger().error(
                    f"{box_name}: 그리퍼 오픈 실패 - 그립을 유지한 채로 복귀합니다."
                )
                status_bus.publish_safety("ERR_DROP", f"{box_name} 그리퍼 오픈 실패")
            wait(0.3)

            retreat_x, retreat_y, retreat_z, rx, ry, rz = box_pos
            retreat_pos = posx(retreat_x, retreat_y, retreat_z + 50, rx, ry, rz)
            movel(retreat_pos)
            wait(0.2)
        else:
            node.get_logger().error(
                f"{box_name}에서 접촉 실패 - 그립 유지한 채로 원위치 복귀. "
                f"이 사과는 놓지 못한 상태로 다음 단계 진행됨."
            )
            status_bus.publish_safety("ERR_DROP", f"{box_name} 접촉 실패")

        # 4) 성공/실패 무관하게 매번 웨이포인트 -> 카메라 순으로 복귀 (홈은 거치지 않음).
        #    다음 사이클은 이 CAMERA 위치에서 바로 이어서 사과를 집음.
        node.get_logger().info(f"Return to way point after {box_name}")
        movel(way_pos)
        wait(0.3)

        node.get_logger().info("Return to CAMERA position")
        movel(CAMERA)
        wait(0.3)

        status_bus.set_state("MOVING")

    node.get_logger().info("=== Apple sorting cycle done ===")

    node.get_logger().info("Move to HOME (종료)")
    movej(HOME)
    wait(0.3)

    comm_node.destroy_node()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
