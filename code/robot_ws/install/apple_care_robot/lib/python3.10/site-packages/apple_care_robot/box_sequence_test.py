"""
Box Sequence Test Script
==========================

목적:
    Vision/Decision 로직 없이, 지금까지 티칭한 좌표들이 실제로 맞는지
    "홈 -> 카메라 위치 -> 각 박스" 순서로 로봇을 움직여서 눈으로 확인하는
    수동 테스트용 스크립트입니다.

비유:
    자동 배달 로직을 짜기 전에, 먼저 사람이 손으로 지도에 표시한 지점들이
    실제로 맞는 길인지 한 번 직접 운전해보는 것과 같아요.
    (자동화는 그 다음 단계)

전제:
    - doosan_robot2 (ROS2 driver)가 이미 설치되어 있고, DSR_ROBOT2 모듈을
      import 할 수 있는 환경이어야 합니다.
    - 아래 movel / posx / wait 등의 함수명은 설치된 doosan_robot2 버전에 따라
      다를 수 있으니, 안 맞으면 형 환경의 예제 스크립트(예: single_robot_*.py)에서
      실제 import 구문을 확인해서 맞춰주세요.

참고:
    - HOME은 posx(TCP 위치)가 아니라 posj(관절 각도)로 잡혀 있어서
      다른 지점들과 다르게 movej로 이동합니다. (movel과 헷갈리지 않도록 주의)
"""

import rclpy
import DR_init

ROBOT_ID = "dsr01"      # TODO: 실제 로봇 네임스페이스/ID로 확인 후 수정
ROBOT_MODEL = "m0609"

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL


def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node("box_sequence_test", namespace=ROBOT_ID)
    DR_init.__dsr__node = node

    # DR_init 설정 이후에 import 해야 동작함 (doosan_robot2 예제들의 공통 패턴)
    from DSR_ROBOT2 import movel, movej, posx, posj, wait, set_velx, set_accx, set_velj, set_accj
    from force_place import force_controlled_place

    # ------------------------------------------------------------------
    # 좌표 정의 (사진에 있던 System 변수값 그대로 옮김)
    # posx(x, y, z, rx, ry, rz)  -> TCP 위치/자세 (mm / deg), movel에서 사용
    # posj(j1, j2, j3, j4, j5, j6) -> 각 관절 각도 (deg), movej에서 사용
    # ------------------------------------------------------------------
    HOME = posj(0, 0, 90, 0, 90, 0)  # joint 좌표라서 movex가 아니라 movej로 이동해야 함

    CAMERA = posx(510.55, -113.0, 323.08, 163.09, 175.42, 160.62)

    WAY1 = posx(446.54, 244.89, 206.64, 79.38, 177.62, 172.06)   # b1,b2,b3 진입 전 경유점
    WAY2 = posx(740.15, 189.44, 182.73, 170.37, -162.55, -103.95)  # b4 진입 전 경유점

    B1 = posx(271.57, 248.16, 47.72, 90.88, -177.84, 89.87)
    B2 = posx(466.13, 239.03, 38.71, 39.87, -175.85, 55.2)
    B3 = posx(635.23, 246.09, 34.71, 169.25, -179.16, -169.02)
    B4 = posx(786.27, 185.37, 36.23, 8.77, 158.32, 91.18)

    # 박스별로 어느 경유점을 거쳐야 하는지 짝지어둠
    # (b4만 way2, 나머지는 way1)
    BOXES = [
        ("b1", B1, WAY1),
        ("b2", B2, WAY1),
        ("b3", B3, WAY1),
        ("b4", B4, WAY2),
    ]

    # 속도/가속도 설정 (수동 테스트 단계라 일부러 느리게)
    # TODO: 값 두배 올린 상태임. 느리면 더 올리자. 
    set_velx(60, 40)   # movel용: (선속도 mm/s, 각속도 deg/s)
    set_accx(60, 40)
    set_velj(40)       # movej용: 관절 속도 (deg/s)
    set_accj(40)       # movej용: 관절 가속도 (deg/s^2)

    node.get_logger().info("=== Box sequence test start ===")

    # 1) 홈에서 시작 (posj 좌표이므로 movej 사용)
    node.get_logger().info("Move to HOME")
    movej(HOME)
    wait(0.5)

    # 2) 카메라 위치로 이동 (전체 촬영 지점)
    node.get_logger().info("Move to CAMERA position")
    movel(CAMERA)
    wait(0.5)

    # 3) 각 박스를 순서대로 왕복
    for name, box_pos, way_pos in BOXES:
        node.get_logger().info(f"--- Visiting {name} ---")

        node.get_logger().info(f"Move to way point before {name}")
        movel(way_pos)
        wait(0.3)
        node.get_logger().info(f"Move to {name} (hover above box)")
        movel(box_pos)
        wait(0.3)

        # TODO: 여기서 실제로는 box_pos보다 살짝 위(hover)에서 시작해야 안전함
        contact_ok = force_controlled_place(node, box_pos)

        if contact_ok:
            node.get_logger().info(f"[TODO] Gripper open (apple released into {name})")
            # TODO: 실제 그리퍼 open 명령 연결
        else:
            node.get_logger().error(f"{name}에서 접촉 실패 - 그립 유지한 채로 원위치 복귀")
            # TODO: 실패 시 그냥 놓지 말고 way point로 복귀 후 예외 처리 연결
        node.get_logger().info(f"Return to way point after {name}")
        movel(way_pos)
        wait(0.3)

    # 4) 마지막에 다시 홈으로 복귀 (posj 좌표이므로 movej 사용)
    node.get_logger().info("Return to HOME")
    movej(HOME)

    node.get_logger().info("=== Box sequence test done ===")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
