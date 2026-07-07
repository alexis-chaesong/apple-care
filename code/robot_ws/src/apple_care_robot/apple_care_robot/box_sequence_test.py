"""
Box Sequence Test Script (OnRobot 그리퍼는 openclose.py를 통해 Modbus/TCP로 직접 제어)
=======================================================================================
"""

import rclpy
import DR_init
from openclose import gripper_open
from apple_sorting_cycle import pick_apple

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL


def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node("box_sequence_test", namespace=ROBOT_ID)
    DR_init.__dsr__node = node

    from DSR_ROBOT2 import movel, movej, posx, posj, wait, set_velx, set_accx, set_velj, set_accj
    from force_place import (
        force_controlled_place,
        CAREFUL_APPROACH_VEL, CAREFUL_APPROACH_ACC,
        EXISTING_APPLE_HOVER_CLEARANCE, EXISTING_APPLE_FORCE_THRESHOLD,
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

    # 비전 미연동 - 사과를 집을 임시 좌표 (apple_sorting_cycle.py의 더미 pick 위치와 동일)
    APPLE_PICK_POS = posx(492.0, 13.48, 23.39, 128.8, -174.74, -139.09)

    # 비전 미연동 - "박스에 이미 사과가 있는지"를 여기서 하드코딩으로 표시.
    # 나중에 비전이 CAMERA 위치에서 인식한 결과로 이 값을 대체하면 됨.
    BOXES = [
        ("b1", B1, WAY1, True),    # 이미 사과가 있다고 가정 -> hover 진입을 조심히
        ("b2", B2, WAY1, False),
        ("b3", B3, WAY1, False),
        ("b4", B4, WAY2, False),
    ]

    set_velx(45, 30)
    set_accx(45, 30)
    set_velj(30)
    set_accj(30)

    def place_apple_in_box(name, box_pos, way_pos, has_existing_apple):
        """
        사과 한 개를 집어서 하나의 박스에 놓는 전체 사이클 (분기 단위).
        홈 -> 카메라 -> 집기 -> 홈(안전 경유) -> 웨이포인트 -> 박스(hover) ->
        힘제어 하강 -> 그리퍼 오픈 -> 후퇴 -> 웨이포인트 -> 카메라 -> 홈 복귀
        (박스가 비어있어도 항상 힘제어로 하강함 - 고속/생략 없음.
         단, 이미 사과가 있다고 알려진 박스는 hover 위치 진입 자체를 조심히 함)
        """
        node.get_logger().info(f"--- Visiting {name} ---")

        # 1) 사이클 시작은 항상 HOME에서
        node.get_logger().info("Move to HOME")
        movej(HOME)
        wait(0.3)

        # 2) 카메라 위치 경유 (비전이 붙기 전이라 지금은 확인용)
        node.get_logger().info("Move to CAMERA position")
        movel(CAMERA)
        wait(0.3)

        # 3) 사과 집기
        pick_apple(node, APPLE_PICK_POS)
        wait(0.3)

        # 3.5) 집자마자 대각선으로 박스로 가지 않고, 반드시 HOME을 먼저 거침
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

        # 사과를 놓았든 실패했든, 매 박스마다 반드시 웨이포인트 -> 카메라 -> 홈 순으로 복귀
        node.get_logger().info(f"Return to way point after {name}")
        movel(way_pos)
        wait(0.3)

        node.get_logger().info("Return to CAMERA position")
        movel(CAMERA)
        wait(0.3)

        node.get_logger().info("Return to HOME")
        movej(HOME)
        wait(0.3)

        return contact_ok

    node.get_logger().info("=== Box sequence test start ===")

    for name, box_pos, way_pos, has_existing_apple in BOXES:
        place_apple_in_box(name, box_pos, way_pos, has_existing_apple)

    node.get_logger().info("=== Box sequence test done ===")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()