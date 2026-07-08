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

박스 매핑 (destination -> 실제 박스, 실제 배치로 확정됨):
    processing_box -> b1 (경유점: way1)
    ugly_box       -> b2 (경유점: way1)
    normal_box     -> b3 (경유점: way1)
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
    - 이미 사과가 있다고 알려진 박스(현재 b1/processing_box)는, 사과 더미 꼭대기가 원래
      hover 좌표(box_pos)보다 높이 나와 있을 수 있어서 box_pos까지 블라인드로
      내려가지 않음. 대신 box_pos보다 EXISTING_APPLE_HOVER_CLEARANCE만큼 더 높은
      위치까지만 조심히(CAREFUL_APPROACH_VEL/ACC) movel로 접근하고, 그 지점부터
      force_controlled_place로 힘제어 하강을 시작해서 사과 더미와의 접촉도 힘으로 감지함.
    - 박스에 놓은 뒤(성공/실패 무관)에는 매번 웨이포인트 -> 카메라 위치 순서로 복귀함.

TODO:
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
from apple_care_msgs.srv import SrvAppleStatus

from apple_care_robot.force_place import (
    force_controlled_place,
    CAREFUL_APPROACH_VEL, CAREFUL_APPROACH_ACC,
    EXISTING_APPLE_HOVER_CLEARANCE, EXISTING_APPLE_FORCE_THRESHOLD,
)
from apple_care_robot.openclose import gripper_open
from apple_care_robot.status_bus import StatusBus
from apple_care_robot.vision_transform import camera_to_base, SMALL_APPLE_DEPTH_OFFSET_MM

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

# box_sequence_test.py에서 포팅: 그립 자체가 실패하는 경우(사과를 놓침 -
# grasp_apple_with_force_feedback이 picked_ok=False)를 대비한 재시도 횟수. 매번
# CAMERA로 돌아가서 위치를 다시 받아온 뒤 집기를 재시도함.
MAX_PICK_ATTEMPTS = 3

VISION_SERVICE_NAME = "/get_apple_status"

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
    CAMERA = posx(493.30, -104.81, 247.48, 26.09, 175.22, 23.18)

    WAY1 = posx(446.54, 244.89, 206.64, 79.38, 177.62, 172.06)
    WAY2 = posx(740.15, 189.44, 182.73, 170.37, -162.55, -103.95)

    B1 = posx(271.57, 248.16, 47.72, 90.88, -177.84, 89.87)
    B2 = posx(466.13, 239.03, 38.71, 39.87, -175.85, 55.2)
    B3 = posx(635.23, 246.09, 34.71, 169.25, -179.16, -169.02)
    B4 = posx(786.27, 185.37, 36.23, 8.77, 158.32, 91.18)

    # destination -> (박스 이름, 박스 좌표, 경유점, 기존 사과 존재 여부)
    # 실제 물리 배치 확정: b1=가공용, b2=못난이, b3=정상, b4=폐기
    global DESTINATION_TO_BOX
    DESTINATION_TO_BOX = {
        "processing_box": ("b1", B1, WAY1, True),   # 이미 사과가 있다고 가정 -> hover 진입을 조심히
        "ugly_box":       ("b2", B2, WAY1, False),
        "normal_box":     ("b3", B3, WAY1, False),
        "discard_box":    ("b4", B4, WAY2, False),
    }

    set_velx(45, 30)
    set_accx(45, 30)
    set_velj(30)
    set_accj(30)

    # box_sequence_test.py에서 포팅: 그립 실패 재시도 시 CAMERA에서 사과 위치를
    # 다시 받아오기 위한 전용 서비스 클라이언트 (백엔드의 decision은 최초 감지
    # 시점의 위치만 담고 있어서, 그립을 놓친 뒤 재시도할 땐 최신 위치가 필요함).
    vision_client = node.create_client(SrvAppleStatus, VISION_SERVICE_NAME)
    while not vision_client.wait_for_service(timeout_sec=3.0):
        node.get_logger().info(f"Waiting for {VISION_SERVICE_NAME} service...")

    def refresh_pick_pos(condition):
        """
        그립 실패 후 재시도 직전에 호출. CAMERA 위치에 서 있는 상태에서
        get_apple_status를 다시 호출해 최신 위치로 pick_pos를 재계산함
        (사과가 그립 실패 중 살짝 밀렸을 수 있음). 실패(서비스 오류/depth 측정
        실패)하면 None을 반환하고, 호출부는 이전 pick_pos로 그대로 재시도함.
        """
        future = vision_client.call_async(SrvAppleStatus.Request())
        rclpy.spin_until_future_complete(node, future)
        response = future.result()
        if response is None:
            node.get_logger().error(f"{VISION_SERVICE_NAME} 서비스 호출 실패 (위치 재획득)")
            return None

        position = list(response.position or [])
        if not position or all(v == 0.0 for v in position):
            node.get_logger().warn("위치 재획득 실패 - depth 측정 실패(position이 [0,0,0])")
            return None

        camera_pose_at_detection = get_current_posx()[0]
        depth_offset = SMALL_APPLE_DEPTH_OFFSET_MM if condition == "small" else None
        bx, by, bz = camera_to_base(position, camera_pose_at_detection, depth_offset_mm=depth_offset)
        node.get_logger().info(
            f"위치 재획득: camera={position} -> base=({bx:.2f}, {by:.2f}, {bz:.2f})"
        )
        return posx(bx, by, bz, *DEFAULT_PICK_ORIENTATION)

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
    #
    # rclpy.spin(comm_node)처럼 executor를 명시하지 않으면 프로세스 전역(암묵적)
    # executor를 쓰는데, movej/movel(DSR_ROBOT2 내부)도 같은 컨텍스트에서
    # rclpy.spin_until_future_complete(g_node, future)로 똑같이 암묵적 executor를
    # 쓰려고 해서 "RuntimeError: Executor is already spinning"이 남 (실제로 겪은
    # 문제 - object_detection의 img_node 중첩 executor 충돌과 동일한 원인,
    # python310_to_312_voice_changes.md 19.2번 항목 참고). comm_node 전용
    # SingleThreadedExecutor를 명시적으로 만들어서 완전히 분리해야 함.
    comm_executor = rclpy.executors.SingleThreadedExecutor()
    comm_executor.add_node(comm_node)
    spin_thread = threading.Thread(target=comm_executor.spin, daemon=True)
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

    # CAMERA 위치에 도착 + 그리퍼가 비어있는 이 시점부터가 백엔드 입장에서
    # "지금 보이는 게 진짜 새 사과"라고 신뢰할 수 있는 유일한 구간임.
    # /robot/process_state="READY"를 여기서 명시적으로 발행해야, 백엔드의
    # _vla_consumer_loop가 이 시점에 들어온 Vision 감지만 판단(자동 실행/사람에게
    # 질문)하고, 그 외 구간(집기/이동/내려놓기 중)에 카메라가 우연히 잡는 잔상은
    # 무시하도록 게이팅할 수 있음.
    status_bus.set_state("READY")

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
        condition = decision.get("condition")

        if destination not in DESTINATION_TO_BOX:
            node.get_logger().error(f"알 수 없는 destination: {destination} - 이 사과는 건너뜁니다.")
            continue

        box_name, box_pos, way_pos, has_existing_apple = DESTINATION_TO_BOX[destination]

        # 작은 사과는 몸통이 낮아서 일반 오프셋만큼 내려가면 트레이/받침대와
        # 충돌하므로, condition="small"일 때만 덜 내려가는 전용 오프셋을 씀
        # (box_sequence_test.py에서 포팅).
        depth_offset = SMALL_APPLE_DEPTH_OFFSET_MM if condition == "small" else None

        if pose:
            # pose는 카메라 좌표계 -> 로봇이 현재 서 있는(=CAMERA에서 감지된
            # 순간의) pose를 기준으로 로봇 베이스 좌표계로 변환해야 movel에 그대로
            # 쓸 수 있음. 자세한 설명은 vision_transform.py 상단 docstring 참고.
            camera_pose_at_detection = get_current_posx()[0]
            bx, by, bz = camera_to_base(pose, camera_pose_at_detection, depth_offset_mm=depth_offset)
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
        # 이 decision을 처리하는 순간부터 그리퍼가 다시 CAMERA로 돌아와 열릴 때까지는
        # "새 사과를 보는 중"이 아니므로 READY를 벗어남을 명시적으로 알림
        # (백엔드가 이 값이 READY가 아닐 때는 Vision 감지를 무시하도록 게이팅함)
        #
        # 그립 실패(힘 감지로 사과를 놓친 것으로 판단) 시 pick_pos 하나로 계속
        # 헛손질하지 않도록, CAMERA로 돌아가 위치를 다시 받아온 뒤 MAX_PICK_ATTEMPTS
        # 번까지 재시도함 (box_sequence_test.py의 try_pick_apple에서 포팅).
        picked_ok = False
        for attempt in range(1, MAX_PICK_ATTEMPTS + 1):
            status_bus.set_state("MOVING")
            status_bus.set_motion("PICKING", fruit)
            picked_ok = pick_apple(node, pick_pos)
            status_bus.publish_gripper_status(picked_ok)
            if picked_ok:
                break

            status_bus.publish_safety(
                "ERR_PICK", f"{fruit} 파지(힘 감지) 실패 ({attempt}/{MAX_PICK_ATTEMPTS})"
            )
            if attempt == MAX_PICK_ATTEMPTS:
                break

            node.get_logger().warn(
                f"그립 실패(사과를 놓친 것으로 판단) - CAMERA로 돌아가 위치를 "
                f"다시 받아서 재시도합니다. ({attempt}/{MAX_PICK_ATTEMPTS})"
            )
            # 다음 시도 전에 그리퍼를 확실히 열어둠 (실패한 그립이 반쯤 닫힌
            # 채로 남아있으면 다음 파지 힘 감지 기준선이 틀어질 수 있음).
            gripper_open()
            movel(CAMERA)
            wait(0.3)
            refreshed_pos = refresh_pick_pos(condition)
            if refreshed_pos is not None:
                pick_pos = refreshed_pos
            # 재획득 실패 시 이전 pick_pos로 그대로 재시도함
        wait(0.3)

        if not picked_ok:
            node.get_logger().error(
                f"{MAX_PICK_ATTEMPTS}번 시도해도 그립에 실패해 이번 사과는 포기합니다."
            )
            node.get_logger().info("Move to CAMERA position (집기 포기 후 복귀)")
            movel(CAMERA)
            wait(0.3)
            status_bus.set_state("READY")
            continue

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

        # CAMERA 위치 도착 + 그리퍼는 이미 위에서 열림(gripper_open()) -> 다음 사과를
        # 볼 준비가 된 상태이므로 READY로 되돌림 (예전엔 여기서 계속 "MOVING"으로
        # 남아있어서 백엔드가 "카메라 위치에서 대기 중"과 "이동 중"을 구분 못했음)
        status_bus.set_state("READY")

    node.get_logger().info("=== Apple sorting cycle done ===")

    node.get_logger().info("Move to HOME (종료)")
    movej(HOME)
    wait(0.3)

    comm_executor.shutdown()
    comm_node.destroy_node()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
