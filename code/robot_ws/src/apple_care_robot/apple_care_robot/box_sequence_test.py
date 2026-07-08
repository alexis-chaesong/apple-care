"""
Box Sequence Test Script (OnRobot 그리퍼는 openclose.py를 통해 Modbus/TCP로 직접 제어)
=======================================================================================

사과 집는 위치(pick pos)는 더 이상 하드코딩된 좌표가 아니라, obj_detection의
get_apple_status 서비스를 CAMERA 위치에서 직접 호출해서 얻은 실측 위치를 씀
(apple_sorting_cycle.py가 /decision/result로 받은 pose를 처리하는 것과 동일한
원리 - vision_transform.camera_to_base()로 카메라 좌표계 -> 로봇 베이스 좌표계
변환. 자세한 설명은 vision_transform.py 참고).

박스 매핑도 더 이상 고정 순서(b1->b2->b3->b4)가 아니라 vision이 인식한 사과의
status로 정해짐 (STATUS_TO_BOX_NAME 참고): apple_small -> b1, apple_damaged -> b2,
apple_normal -> b3, apple_rotten(폐기 대상) -> b4. "unknown"처럼 확실하게 분류되지
않은 경우는 집지 않고 재시도함 (wait_for_confident_detection).
박스 좌표(B1~B4)/웨이포인트는 이번 연동 범위 밖이라 여전히 하드코딩임.
"""

import rclpy
import DR_init
from apple_care_msgs.srv import SrvAppleStatus
from apple_care_robot.openclose import gripper_open
from apple_care_robot.apple_sorting_cycle import pick_apple
from apple_care_robot.vision_transform import (
    camera_to_base, DEPTH_OFFSET_MM, SMALL_APPLE_DEPTH_OFFSET_MM,
)

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

# 반드시 앞에 '/'를 붙인 절대 이름이어야 함. 이 노드가 namespace="dsr01"로 생성되기
# 때문에("box_sequence_test" 아래 참고), 상대 이름("get_apple_status")을 쓰면
# rclpy가 자동으로 "/dsr01/get_apple_status"로 리졸브해버려서 obj_detection이
# 네임스페이스 없이 여는 실제 서비스(/get_apple_status)를 영영 못 찾음
# (pick_and_place_voice/robot_control.py의 "/get_3d_position"과 동일한 이유).
VISION_SERVICE_NAME = "/get_apple_status"

# Vision(get_apple_status)이 주는 position은 [x,y,z]뿐이라 그리퍼 방향(rx,ry,rz)이
# 없음 - apple_sorting_cycle.py의 DEFAULT_PICK_ORIENTATION과 동일한 고정값 사용.
PICK_ORIENTATION = (128.8, -174.74, -139.09)

# vision이 판정한 status -> 어느 박스로 보낼지. yolo.py의 CLASS_NAMES 4개 전부
# 대응시킴 (apple_rotten은 폐기 대상이라 B4로 보냄). "unknown"/"empty"는 여기 없어서
# 자동으로 "확실하게 분류 안 됨" 취급됨 (get_apple_detection 참고).
STATUS_TO_BOX_NAME = {
    "apple_small": "b1",
    "apple_damaged": "b2",
    "apple_normal": "b3",
    "apple_rotten": "b4",
}

# CAMERA 위치에서 재시도할 때 쓰는 값 - 확실하게 분류된(=STATUS_TO_BOX_NAME에 있는)
# 사과가 나올 때까지 이만큼 반복해서 다시 물어봄.
MAX_DETECTION_ATTEMPTS = 5
DETECTION_RETRY_WAIT_SEC = 1.0

# 그립 자체가 실패하는 경우(사과를 놓침 - grasp_apple_with_force_feedback이
# picked_ok=False)를 대비한 재시도 횟수. 매번 CAMERA로 돌아가서 위치를 다시
# 받아온 뒤 집기를 재시도함 (try_pick_apple 참고).
MAX_PICK_ATTEMPTS = 3


def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node("box_sequence_test", namespace=ROBOT_ID)
    DR_init.__dsr__node = node

    from DSR_ROBOT2 import (
        movel, movej, posx, posj, wait,
        set_velx, set_accx, set_velj, set_accj, get_current_posx,
    )
    from apple_care_robot.force_place import (
        force_controlled_place,
        CAREFUL_APPROACH_VEL, CAREFUL_APPROACH_ACC,
        EXISTING_APPLE_HOVER_CLEARANCE, EXISTING_APPLE_FORCE_THRESHOLD,
    )

    vision_client = node.create_client(SrvAppleStatus, VISION_SERVICE_NAME)
    while not vision_client.wait_for_service(timeout_sec=3.0):
        node.get_logger().info(f"Waiting for {VISION_SERVICE_NAME} service...")

    def get_apple_detection():
        """
        CAMERA 위치에 서 있는 상태에서 get_apple_status를 한 번 호출해서
        (status, pick_pos)를 반환. status는 STATUS_TO_BOX_NAME의 키 중 하나일 때만
        "확실하게 분류됨"으로 보고, 그 외(예: unknown/empty, depth 측정 실패로
        position이 [0,0,0], 서비스 호출 실패 등)는 전부 (None, None)을 반환함 -
        호출하는 쪽(wait_for_confident_detection)이 이 경우 재시도를 판단함.
        """
        future = vision_client.call_async(SrvAppleStatus.Request())
        rclpy.spin_until_future_complete(node, future)
        response = future.result()

        if response is None:
            node.get_logger().error(f"{VISION_SERVICE_NAME} 서비스 호출 실패")
            return None, None

        node.get_logger().info(
            f"Vision 응답: status={response.status} confidence={response.confidence:.2f} "
            f"position={list(response.position)}"
        )

        if response.status not in STATUS_TO_BOX_NAME:
            # "unknown"/"empty"뿐 아니라 STATUS_TO_BOX_NAME에 없는 라벨이면 전부
            # "아직 확실하게 분류되지 않음"으로 취급 - 애매한 걸 아무 박스에나
            # 넣지 않기 위함.
            node.get_logger().warn(
                f"status='{response.status}'는 확실하게 분류된 게 아니라 이번 감지는 건너뜁니다."
            )
            return None, None

        position = list(response.position)
        # detection.py의 _compute_position은 depth 측정 실패("cz == 0") 시에도
        # 빈 리스트가 아니라 [0.0, 0.0, 0.0]을 반환함. 파이썬에서 [0.0,0.0,0.0]은
        # 원소가 3개라 `not position`으로는 안 걸러지므로(길이만 보고 참/거짓을
        # 판단하기 때문), "전부 0"인지 별도로 체크해야 depth 측정 실패를
        # "카메라 렌즈 위치(진짜 (0,0,0))"로 착각해서 그쪽으로 이동하는 걸 막을 수 있음.
        if not position or all(v == 0.0 for v in position):
            node.get_logger().warn(
                f"status='{response.status}'는 확실하지만 depth 측정 실패로 위치를 "
                "못 얻어서 이번 감지는 건너뜁니다 (position이 [0,0,0])."
            )
            return None, None

        # get_apple_status를 호출한 지금 이 순간 로봇이 CAMERA 위치에 서 있으므로,
        # 이 시점의 get_current_posx()가 곧 "카메라가 이 사과를 본 시점의 로봇 pose".
        camera_pose_at_detection = get_current_posx()[0]
        # 작은 사과는 몸통이 낮아서 일반 오프셋만큼 내려가면 트레이/받침대와
        # 충돌하므로, apple_small일 때만 덜 내려가는 전용 오프셋을 씀.
        depth_offset = (
            SMALL_APPLE_DEPTH_OFFSET_MM if response.status == "apple_small" else DEPTH_OFFSET_MM
        )
        bx, by, bz = camera_to_base(
            position, camera_pose_at_detection, depth_offset_mm=depth_offset
        )
        node.get_logger().info(
            f"Vision 좌표 변환: camera={position} -> "
            f"base=({bx:.2f}, {by:.2f}, {bz:.2f})"
        )
        pick_pos = posx(bx, by, bz, *PICK_ORIENTATION)
        return response.status, pick_pos

    def wait_for_confident_detection():
        """
        get_apple_detection()이 확실한 분류를 줄 때까지 MAX_DETECTION_ATTEMPTS번까지
        재시도. (로봇은 이미 CAMERA 위치에 서 있는 상태 - 재시도 사이에는 다시
        움직이지 않고 그냥 짧게 대기 후 다시 물어봄.)

        Returns:
            (status, pick_pos) 또는 끝까지 확실한 분류가 안 나오면 (None, None).
        """
        for attempt in range(1, MAX_DETECTION_ATTEMPTS + 1):
            status, pick_pos = get_apple_detection()
            if status is not None:
                return status, pick_pos
            node.get_logger().info(
                f"확실한 감지 대기 중... ({attempt}/{MAX_DETECTION_ATTEMPTS})"
            )
            wait(DETECTION_RETRY_WAIT_SEC)
        return None, None

    def try_pick_apple():
        """
        (로봇이 CAMERA 위치에 서 있다고 가정) 확실하게 분류된 사과를 감지해서
        집기를 시도하고, 그립 성공 여부(force feedback 기반)까지 확인한다.
        가끔 힘 감지 파지가 실패해서 사과를 놓치는 경우가 있는데, 그럴 때
        pick_pos 하나로 계속 헛손질하지 않도록 - 그립 실패 시 CAMERA로 돌아가서
        위치를 다시 받아오고(사과가 살짝 밀렸거나 처음 위치가 부정확했을 수 있음)
        MAX_PICK_ATTEMPTS번까지 재시도한다.

        Returns:
            (box_name, True) - 감지 + 집기 + 그립 확인까지 성공.
            (None, False) - 확실한 감지 자체가 안 나왔거나, 재시도 끝에도
                계속 그립에 실패한 경우.
        """
        for attempt in range(1, MAX_PICK_ATTEMPTS + 1):
            status, pick_pos = wait_for_confident_detection()
            if status is None:
                # 확실한 분류 자체가 안 나옴 - 그립 문제가 아니라 애초에 집을
                # 대상이 없다는 뜻이라 여기서 더 재시도할 이유가 없음.
                return None, False

            box_name = STATUS_TO_BOX_NAME[status]
            node.get_logger().info(
                f"감지 결과: status={status} -> {box_name}로 분류 "
                f"(집기 시도 {attempt}/{MAX_PICK_ATTEMPTS})"
            )

            picked_ok = pick_apple(node, pick_pos)
            if picked_ok:
                return box_name, True

            node.get_logger().warn(
                f"그립 실패(사과를 놓친 것으로 판단) - CAMERA로 돌아가 위치를 "
                f"다시 받아서 재시도합니다. ({attempt}/{MAX_PICK_ATTEMPTS})"
            )
            # 다음 시도 전에 그리퍼를 확실히 열어둠 (실패한 그립이 반쯤 닫힌
            # 채로 남아있으면 다음 파지 힘 감지 기준선이 틀어질 수 있음).
            gripper_open()
            movel(CAMERA)
            wait(0.3)

        node.get_logger().error(
            f"{MAX_PICK_ATTEMPTS}번 시도해도 그립에 실패해 이번 사과는 포기합니다."
        )
        return None, False

    # 좌표 정의
    HOME = posj(0, 0, 90, 0, 90, 0)
    CAMERA = posx(493.30, -104.81, 247.48, 26.09, 175.22, 23.18)
    WAY1 = posx(446.54, 244.89, 206.64, 79.38, 177.62, 172.06)
    WAY2 = posx(740.15, 189.44, 182.73, 170.37, -162.55, -103.95)

    B1 = posx(271.57, 248.16, 40.72, 90.88, -177.84, 89.87)
    B2 = posx(466.13, 239.03, 40.71, 39.87, -175.85, 55.2)
    B3 = posx(635.23, 246.09, 40.71, 169.25, -179.16, -169.02)
    B4 = posx(786.27, 185.37, 40.23, 8.77, 158.32, 91.18)

    # 비전 미연동 - "박스에 이미 사과가 있는지"를 여기서 하드코딩으로 표시.
    # 나중에 비전이 박스 안을 직접 인식한 결과로 이 값을 대체하면 됨.
    # 이름(b1~b4) -> (박스 좌표, 경유점, 기존 사과 존재 여부). STATUS_TO_BOX_NAME이
    # 감지된 사과의 status로 이 중 하나를 골라줌 (더 이상 고정 순서로 안 돎).
    BOXES_BY_NAME = {
        "b1": (B1, WAY1, True),    # 이미 사과가 있다고 가정 -> hover 진입을 조심히
        "b2": (B2, WAY1, False),
        "b3": (B3, WAY1, False),
        "b4": (B4, WAY2, False),
    }

    set_velx(45, 30)
    set_accx(45, 30)
    set_velj(30)
    set_accj(30)

    def place_apple_in_box(name, box_pos, way_pos, has_existing_apple):
        """
        try_pick_apple()이 이미 집어서 그립까지 확인해둔 사과를 목적지 박스에
        놓는 사이클 (집기 자체는 여기서 하지 않음).
        홈(안전 경유) -> 웨이포인트 -> 박스(hover) -> 힘제어 하강 -> 그리퍼 오픈 ->
        후퇴 -> 웨이포인트 -> 카메라 복귀
        (박스가 비어있어도 항상 힘제어로 하강함 - 고속/생략 없음.
         단, 이미 사과가 있다고 알려진 박스는 hover 위치 진입 자체를 조심히 함)
        """
        node.get_logger().info(f"--- Sorting to {name} ---")

        # 집자마자 대각선으로 박스로 가지 않고, 반드시 HOME을 먼저 거침
        #      (지름길로 가면 테이블/카메라 장비와 충돌 위험이 있음)
        node.get_logger().info("Move to HOME before heading to box (안전 경유)")
        movej(HOME)
        wait(0.3)

        node.get_logger().info(f"Move to way point before {name}")
        movel(way_pos)
        wait(0.3)

        if has_existing_apple:
            # 이미 사과가 있으면 사과 더미 꼭대기가 원래 hover 좌표(box_pos)보다
            # 높이 나와 있을 수 있어, box_pos까지 블라인드로 내려가면 힘제어가
            # 걸리기도 전에 부딪힐 수 있음. 그래서 box_pos보다 더 높은 위치까지만
            # 조심히 movel로 접근하고, 그 지점부터 힘제어로 하강을 시작함.
            hx, hy, hz, hrx, hry, hrz = box_pos
            safe_hover_pos = posx(hx, hy, hz + EXISTING_APPLE_HOVER_CLEARANCE, hrx, hry, hrz)
            node.get_logger().info(
                f"{name}: 이미 사과가 있다고 가정 - 원래 hover보다 "
                f"{EXISTING_APPLE_HOVER_CLEARANCE}mm 위에서부터 조심히 접근합니다."
            )
            movel(safe_hover_pos, vel=CAREFUL_APPROACH_VEL, acc=CAREFUL_APPROACH_ACC)
            wait(0.3)
            contact_ok = force_controlled_place(
                node, safe_hover_pos, force_threshold=EXISTING_APPLE_FORCE_THRESHOLD
            )
        else:
            node.get_logger().info(f"Move to {name} (hover above box)")
            movel(box_pos)
            wait(0.3)
            contact_ok = force_controlled_place(node, box_pos)

        if contact_ok:
            node.get_logger().info(f"Apple contact confirmed at {name}. Opening gripper.")

            # 메인 루프 컴플라이언스가 완전히 풀린 클린한 상태에서 그리퍼 오픈 호출
            gripper_opened = gripper_open()

            if not gripper_opened:
                node.get_logger().error(
                    f"{name}: 그리퍼 오픈 실패 - 그립을 유지한 채로 복귀합니다."
                )

            # 두산 로봇 멈춤 상태 유지 (그리퍼 물리적 동작 완료 대기)
            wait(1.0)

            # 안전하게 z축으로 50mm 퇴적 후 이동
            retreat_x, retreat_y, retreat_z, rx, ry, rz = box_pos
            retreat_pos = posx(retreat_x, retreat_y, retreat_z + 50, rx, ry, rz)
            movel(retreat_pos)
            wait(0.2)
        else:
            node.get_logger().error(f"{name}에서 접촉 실패 - 그립 유지한 채로 복귀")

        # 사과를 놓았든 실패했든, 매 박스마다 웨이포인트 -> 카메라 순으로 복귀하고
        # 여기서 멈춤 (홈은 안 거침 - 다음 사이클이 바로 이 CAMERA 위치에서
        # 이어서 다음 사과를 감지함. run_one_cycle/전체 루프 뒤의 마지막 HOME
        # 복귀만 예외).
        node.get_logger().info(f"Return to way point after {name}")
        movel(way_pos)
        wait(0.3)

        node.get_logger().info("Return to CAMERA position")
        movel(CAMERA)
        wait(0.3)

        return contact_ok

    def run_one_cycle():
        """
        (로봇이 이미 CAMERA 위치에 서 있다고 가정하고) 확실하게 분류된 사과를
        감지해서 집고(그립 실패 시 CAMERA에서 재시도 - try_pick_apple 참고),
        그 status에 맞는 박스로 분류해서 놓음.
        확실한 분류/그립이 끝내 안 되면 홈으로 돌아가지 않고 CAMERA에 그대로
        머무름 - 다음 사이클이 바로 이어서 재시도하도록.

        Returns:
            성공해서 실제로 놓기까지 시도했으면 박스 이름(str), 이번 사이클을
            통째로 건너뛰었으면 None.
        """
        box_name, picked_ok = try_pick_apple()
        if not picked_ok:
            node.get_logger().warn(
                "이번 사이클은 건너뜁니다 (CAMERA 위치에 그대로 대기)."
            )
            return None

        box_pos, way_pos, has_existing_apple = BOXES_BY_NAME[box_name]
        place_apple_in_box(box_name, box_pos, way_pos, has_existing_apple)
        return box_name

    node.get_logger().info("=== Box sequence test start ===")

    # 시작하자마자 그리퍼부터 열어둠 - 이전 실행이 그립을 잡은 채로 끝났거나
    # 하드웨어 전원이 새로 들어와서 그리퍼 상태를 모르는 채로 시작할 수 있으므로,
    # 첫 pick 이전에 항상 열린 상태에서 시작하도록 보장.
    node.get_logger().info("Opening gripper before starting")
    if not gripper_open():
        node.get_logger().error("시작 시 그리퍼 오픈 실패 - 하드웨어 연결 상태를 확인하세요.")

    # HOME은 전체 시퀀스에서 맨 처음(여기)과 맨 마지막(루프 끝난 뒤)에만 거침.
    # 그 사이에는 매 사이클마다 CAMERA에서 바로 감지 -> 집기 -> 박스 -> CAMERA로
    # 돌아오는 식으로만 돌고, 사이클 사이에 불필요한 HOME 왕복을 하지 않음.
    node.get_logger().info("Move to HOME (최초 1회)")
    movej(HOME)
    wait(0.3)

    node.get_logger().info("Move to CAMERA position")
    movel(CAMERA)
    wait(0.3)

    # 박스 개수(4개)만큼 사이클을 돎 - 매 사이클마다 실제로 뭘 감지했느냐에 따라
    # 목적지 박스가 달라짐 (고정 순서 b1->b2->b3->b4가 아님).
    for _ in range(len(BOXES_BY_NAME)):
        run_one_cycle()

    node.get_logger().info("Move to HOME (종료)")
    movej(HOME)
    wait(0.3)

    node.get_logger().info("=== Box sequence test done ===")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()