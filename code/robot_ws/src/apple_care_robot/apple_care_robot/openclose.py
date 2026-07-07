"""
Gripper Open/Close (OnRobot RG2, Modbus/TCP 직접 제어)
=========================================================

box_sequence_test.py / apple_sorting_cycle.py 등에서
    from openclose import gripper_open, gripper_close
형태로 가져다 씀.

OnRobotRGControllerServer ROS2 노드나 "/onrobot/sendCommand" 서비스에는
의존하지 않음 (해당 노드가 실물/시뮬레이션 환경에서 죽어있어 응답하지 않는
문제가 있었음). 대신 Compute Box에 Modbus/TCP로 직접 접속해서 레지스터를 씀.
"""

import time

from pymodbus.client.sync import ModbusTcpClient

# OnRobot Compute Box 접속 정보 (실제 장비의 IP로 맞춰서 수정)
GRIPPER_IP = "192.168.1.1"
GRIPPER_PORT = 502
GRIPPER_CHANGER_ADDR = 65

# RG2 기준 값. 다른 그리퍼를 쓰면 이 두 값을 바꿔야 함.
GRIPPER_MAX_FORCE = 400    # 0.1N 단위 (=40N)
GRIPPER_MAX_WIDTH = 1100   # 0.1mm 단위 (=110mm, 완전히 벌린 상태)

# 명령 자체는 즉시 응답하지만, 실제 그리퍼가 움직여서 멈추기까지는
# 시간이 더 걸리므로 호출한 쪽에서 이 정도는 대기해줘야 함.
MOTION_WAIT_SEC = 1.0


def _send_command(force: int, width: int) -> bool:
    """레지스터 0~2: [목표 힘, 목표 폭, 컨트롤(16=이동 실행)]을 그리퍼에 씀."""
    client = ModbusTcpClient(host=GRIPPER_IP, port=GRIPPER_PORT, timeout=1)
    if not client.connect():
        print(f"[openclose] OnRobot 그리퍼({GRIPPER_IP}:{GRIPPER_PORT})에 연결하지 못했습니다!")
        return False

    try:
        client.write_registers(
            address=0,
            values=[force, width, 16],
            unit=GRIPPER_CHANGER_ADDR,
        )
        return True
    finally:
        client.close()


def gripper_open() -> bool:
    """그리퍼를 여는 함수 (사과를 놓을 때 호출). 성공 여부를 bool로 반환."""
    ok = _send_command(GRIPPER_MAX_FORCE, GRIPPER_MAX_WIDTH)
    if not ok:
        print("[openclose] 그리퍼 오픈 명령 전송 실패")
    time.sleep(MOTION_WAIT_SEC)
    return ok


def gripper_close() -> bool:
    """그리퍼를 닫는 함수 (사과를 집을 때 호출). 성공 여부를 bool로 반환."""
    ok = _send_command(GRIPPER_MAX_FORCE, 0)
    if not ok:
        print("[openclose] 그리퍼 클로즈 명령 전송 실패")
    time.sleep(MOTION_WAIT_SEC)
    return ok
