"""
prepare_camera.py
====================

obj_detection의 카메라 인식 노드(object_detection)를 띄우기 전에 먼저 실행하는
준비 스크립트. 로봇 팔이 시야를 가리고 있거나 그리퍼가 닫혀 있는 채로 카메라
노드를 켜면 배경 캘리브레이션(_calibrate_scene)이나 인식 자체가 엉망이 되므로,
카메라 노드를 켜기 전에 로봇을 미리 정리된 자세로 만들어 둔다.

하는 일:
    1) 그리퍼를 완전히 연다 (그리퍼가 닫혀 있으면 그 자체가 시야를 가리거나
       인식 대상으로 오탐될 수 있음).
    2) 로봇을 HOME -> CAMERA 순서로 이동시켜서 카메라가 트레이를 내려다보는
       고정 자세를 만든다.

⚠️ CAMERA 좌표는 실제로 사과 인식/피킹에 쓰는 스크립트(box_sequence_test.py)의
CAMERA 값과 반드시 같아야 한다. vision_transform.
camera_to_base()가 "카메라가 사과를 본 순간의 로봇 pose"를 기준으로 좌표를
변환하므로, 이 스크립트가 로봇을 다른 위치에 세워두면 이후 배경 캘리브레이션
(_calibrate_scene)이 엉뚱한 시야 기준으로 이루어짐. 좌표를 바꿀 일이 있으면
두 파일(prepare_camera.py / box_sequence_test.py)의
CAMERA 값을 같이 맞춰야 한다.

⚠️ 로봇 제어(DSR_ROBOT2)에는 여전히 의존하므로 robot_ws가 빌드/소싱되어 있어야
하지만(실행.txt의 p2() 참고), 그리퍼 제어는 apple_care_robot을 import하지 않고
gripper_control.py에 obj_detection 자체 코드로 들어있음 (openclose.py와 로직
동일 - 자세한 이유는 gripper_control.py 상단 참고).

사용법 (obj_detection 카메라 노드보다 먼저 실행):
    ros2 run obj_detection prepare_camera
    ros2 run obj_detection object_detection   # 준비가 끝난 뒤 카메라 노드 실행
"""

import rclpy
import DR_init
from obj_detection.gripper_control import gripper_open

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL


def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node("prepare_camera", namespace=ROBOT_ID)
    DR_init.__dsr__node = node

    from DSR_ROBOT2 import movel, movej, posx, posj, wait, set_velx, set_accx, set_velj, set_accj

    # box_sequence_test.py의 HOME/CAMERA와 반드시 동일한 값을 써야 함 (docstring 참고)
    HOME = posj(0, 0, 90, 0, 90, 0)
    CAMERA = posx(493.30, -104.81, 247.48, 26.09, 175.22, 23.18)

    set_velx(45, 30)
    set_accx(45, 30)
    set_velj(30)
    set_accj(30)

    node.get_logger().info("=== Prepare camera: 그리퍼 오픈 + CAMERA 위치 이동 ===")

    node.get_logger().info("그리퍼 오픈 (시야 확보)")
    opened = gripper_open()
    if not opened:
        node.get_logger().error("그리퍼 오픈 명령 실패 - 하드웨어 연결 상태를 확인하세요.")

    node.get_logger().info("Move to HOME")
    movej(HOME)
    wait(0.3)

    node.get_logger().info("Move to CAMERA position")
    movel(CAMERA)
    wait(0.3)

    node.get_logger().info(
        "=== 준비 완료: 로봇이 CAMERA 위치에 있고 그리퍼가 열려 있습니다. "
        "이제 'ros2 run obj_detection object_detection'을 실행하세요. ==="
    )

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
