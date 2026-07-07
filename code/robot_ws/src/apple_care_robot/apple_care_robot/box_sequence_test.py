"""
Box Sequence Test Script (OnRobot 그리퍼 Modbus/TCP 직접 제어)
=============================================================
"""

import rclpy
import DR_init
from pymodbus.client.sync import ModbusTcpClient

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

# OnRobot Compute Box 접속 정보 (실제 장비의 IP로 맞춰서 수정)
GRIPPER_IP = "192.168.1.1"
GRIPPER_PORT = 502
GRIPPER_CHANGER_ADDR = 65

# RG2 기준 값
GRIPPER_MAX_FORCE = 400    # 0.1N 단위 (=40N)
GRIPPER_MAX_WIDTH = 1100   # 0.1mm 단위 (=110mm, 완전히 벌린 상태)


def my_real_gripper_open(node):
    """
    OnRobotRGControllerServer ROS2 노드 없이, Modbus/TCP로 그리퍼에 직접 접속해서 여는 함수
    """
    node.get_logger().info("[하드웨어 액션] OnRobot 그리퍼 오픈 (Modbus/TCP 직접 제어)...")

    client = ModbusTcpClient(host=GRIPPER_IP, port=GRIPPER_PORT, timeout=1)
    if not client.connect():
        node.get_logger().error(f"OnRobot 그리퍼({GRIPPER_IP}:{GRIPPER_PORT})에 연결하지 못했습니다!")
        return False

    try:
        # 레지스터 0~2: [목표 힘, 목표 폭(최대=완전 오픈), 컨트롤(16=이동 실행)]
        client.write_registers(
            address=0,
            values=[GRIPPER_MAX_FORCE, GRIPPER_MAX_WIDTH, 16],
            unit=GRIPPER_CHANGER_ADDR,
        )
        node.get_logger().info("[하드웨어 액션] OnRobot 그리퍼 오픈 명령 송신 완료")
        return True
    finally:
        client.close()


def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node("box_sequence_test", namespace=ROBOT_ID)
    DR_init.__dsr__node = node

    from DSR_ROBOT2 import movel, movej, posx, posj, wait, set_velx, set_accx, set_velj, set_accj
    from force_place import force_controlled_place

    # 좌표 정의
    HOME = posj(0, 0, 90, 0, 90, 0)
    CAMERA = posx(510.55, -113.0, 323.08, 163.09, 175.42, 160.62)
    WAY1 = posx(446.54, 244.89, 206.64, 79.38, 177.62, 172.06)   
    WAY2 = posx(740.15, 189.44, 182.73, 170.37, -162.55, -103.95)  

    B1 = posx(271.57, 248.16, 47.72, 90.88, -177.84, 89.87)
    B2 = posx(466.13, 239.03, 38.71, 39.87, -175.85, 55.2)
    B3 = posx(635.23, 246.09, 34.71, 169.25, -179.16, -169.02)
    B4 = posx(786.27, 185.37, 36.23, 8.77, 158.32, 91.18)

    BOXES = [
        ("b1", B1, WAY1),
        ("b2", B2, WAY1),
        ("b3", B3, WAY1),
        ("b4", B4, WAY2),
    ]

    set_velx(60, 40)
    set_accx(60, 40)
    set_velj(40)
    set_accj(40)

    def place_apple_in_box(name, box_pos, way_pos):
        """
        사과 한 개를 하나의 박스에 놓는 전체 사이클 (분기 단위).
        이동 -> 힘제어 접촉 감지 -> 그리퍼 오픈 -> 후퇴 -> 웨이포인트 -> 카메라 위치 -> 홈 복귀
        (비전 연동 전이라 지금은 어느 박스로 갈지가 BOXES 리스트로 고정되어 있음)
        """
        node.get_logger().info(f"--- Visiting {name} ---")

        node.get_logger().info(f"Move to way point before {name}")
        movel(way_pos)
        wait(0.3)
        node.get_logger().info(f"Move to {name} (hover above box)")
        movel(box_pos)
        wait(0.3)

        # 힘제어 하강 수행 (단순 감지만 수행하도록 매개변수 분리)
        contact_ok = force_controlled_place(node, box_pos)

        if contact_ok:
            node.get_logger().info(f"Apple contact confirmed at {name}. Opening gripper.")

            # 메인 루프 컴플라이언스가 완전히 풀린 클린한 상태에서 그리퍼 오픈 호출
            gripper_opened = my_real_gripper_open(node)

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

    node.get_logger().info("Move to HOME")
    movej(HOME)
    wait(0.5)

    node.get_logger().info("Move to CAMERA position")
    movel(CAMERA)
    wait(0.5)

    for name, box_pos, way_pos in BOXES:
        place_apple_in_box(name, box_pos, way_pos)

    node.get_logger().info("=== Box sequence test done ===")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()