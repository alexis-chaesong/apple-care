"""
Gripper Open/Close (OnRobot RG2, Modbus/TCP 직접 제어)
=========================================================

box_sequence_test.py 등에서
    from apple_care_robot.openclose import gripper_open, gripper_close_with_force
형태로 가져다 씀.

OnRobotRGControllerServer ROS2 노드나 "/onrobot/sendCommand" 서비스에는
의존하지 않음 (해당 노드가 실물/시뮬레이션 환경에서 죽어있어 응답하지 않는
문제가 있었음). 대신 Compute Box에 Modbus/TCP로 직접 접속해서 레지스터를 씀.

pymodbus 버전 호환:
    ROS 2 Humble 환경(구버전 pymodbus, 2.x대)과 Jazzy 환경(pymodbus 3.x대)에서
    각각 apt로 설치되는 python3-pymodbus 버전이 달라 API가 서로 다름.
        - import 경로: pymodbus < 3.0은 `pymodbus.client.sync.ModbusTcpClient`,
          >= 3.0은 `.sync` 서브모듈이 없어지고 `pymodbus.client.ModbusTcpClient`.
        - write_registers()의 슬레이브 ID 인자명: 버전에 따라 `unit`(2.x~3.0대) /
          `slave`(3.x대) / `device_id`(최신 3.7+)로 계속 바뀜.
    아래는 두 경로를 모두 시도해서 import하고, write_registers 실제 시그니처를
    inspect해서 어떤 이름을 쓰는지 런타임에 판별하는 방식으로 두 배포판 모두에서
    코드 수정 없이 그대로 동작하게 함.
"""

import inspect
import time

try:
    from pymodbus.client.sync import ModbusTcpClient  # pymodbus < 3.0 (예: ROS 2 Humble)
except ImportError:
    from pymodbus.client import ModbusTcpClient  # pymodbus >= 3.0 (예: ROS 2 Jazzy)


def _resolve_unit_kwarg() -> str:
    """설치된 pymodbus 버전의 write_registers가 쓰는 슬레이브 ID 인자명을 찾는다."""
    params = inspect.signature(ModbusTcpClient.write_registers).parameters
    for name in ("unit", "slave", "device_id"):
        if name in params:
            return name
    # 위 세 이름 중 아무것도 못 찾으면(알 수 없는 미래 버전) 가장 흔한 이름으로 시도
    return "slave"


_UNIT_KWARG = _resolve_unit_kwarg()

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
            **{_UNIT_KWARG: GRIPPER_CHANGER_ADDR},
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


def gripper_close_with_force(force: int) -> bool:
    """
    지정한 힘(force, 0.1N 단위, 0~GRIPPER_MAX_FORCE)으로 그리퍼를 닫는 함수.

    grasp_force.py의 실시간 힘 감지 파지에서, 사과 크기에 맞춰 힘을 단계적으로
    올려가며 이 함수를 반복 호출함. Modbus 레지스터에 목표 force를 직접 쓰는
    구조라 힘 값을 매번 절대값으로 바로 지정할 수 있음 (증감 명령을 여러 번
    보내며 내부 상태를 따로 추적할 필요 없음).
    """
    force = max(0, min(GRIPPER_MAX_FORCE, force))
    ok = _send_command(force, 0)
    if not ok:
        print(f"[openclose] 그리퍼 클로즈(힘={force}) 명령 전송 실패")
    time.sleep(MOTION_WAIT_SEC)
    return ok
