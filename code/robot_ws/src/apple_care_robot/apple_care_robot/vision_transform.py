"""
vision_transform.py
=====================

Vision(obj_detection)이 주는 좌표는 항상 "카메라 좌표계" 기준이다
(obj_detection/depth_utils.py의 _pixel_to_camera_coords 참고). 이 카메라 좌표를
로봇이 실제로 movel 할 수 있는 "로봇 베이스 좌표계"로 바꿔주는 게 이 파일의 역할.

cobot2_ws/src/pick_and_place_voice/robot_control/robot_control.py의
get_robot_pose_matrix / transform_to_base 로직을 그대로 참고함 (동일한 방식의
eye-in-hand 캘리브레이션: 그리퍼에 카메라가 고정 부착되어 있고, 그 상대 위치가
T_gripper2camera.npy에 저장돼 있음).

좌표 변환 원리:
    카메라가 그리퍼에 고정되어 있으므로, "카메라가 본 물체 위치"를 로봇 베이스
    좌표계로 옮기려면 2단계 변환이 필요함.
        1) base2gripper : 사진을 찍은 그 순간의 로봇 pose(get_current_posx())로
           만든 "베이스 -> 그리퍼" 변환 행렬.
        2) gripper2cam  : 캘리브레이션으로 미리 구해둔 "그리퍼 -> 카메라"
           고정 변환 행렬 (T_gripper2camera.npy).
    이 둘을 곱하면 base2cam(베이스 -> 카메라) 행렬이 나오고, 여기에 카메라가
    본 좌표(동차좌표)를 곱하면 베이스 좌표계 기준 위치가 나온다.

주의:
    robot_posx는 반드시 "카메라가 그 물체를 봤을 때"의 로봇 pose여야 한다.
    apple_sorting_cycle.py에서는 로봇이 CAMERA 위치에 서 있는 동안 vision이
    감지 -> 백엔드가 판정 -> /decision/result로 pose가 내려오는 구조라서,
    decision을 받은 시점에 get_current_posx()를 불러도 (로봇이 그 사이에
    다른 곳으로 움직이지 않았다면) CAMERA 위치와 같다.
"""

import os

import numpy as np
from ament_index_python.packages import get_package_share_directory
from scipy.spatial.transform import Rotation

PACKAGE_NAME = "apple_care_robot"
GRIPPER2CAM_FILENAME = "T_gripper2camera.npy"

# 카메라 depth 계산 원점(렌즈)과 실제로 잡아야 할 지점(사과 표면) 사이의
# 오차를 보정하는 값(mm). z(깊이) 축에 더해짐 - 이는 mm 단위. pick_and_place_voice의
# DEPTH_OFFSET(-18.0)을 그대로 가져온 시작값이며, 실제 그리퍼/카메라 장착
# 위치가 다르면 실측해서 다시 튜닝해야 함.
DEPTH_OFFSET_MM = -80.0

# 작은 사과는 몸통 자체가 낮아서, 일반 사과 기준으로 튜닝된 DEPTH_OFFSET_MM만큼
# 내려가면 사과를 지나쳐 트레이/받침대와 충돌함. 그래서 작은 사과 전용으로 10mm
# 덜 내려가는 오프셋을 따로 둠 (camera_to_base의 depth_offset_mm 인자로 넘겨서 씀).
SMALL_APPLE_DEPTH_OFFSET_MM = DEPTH_OFFSET_MM + 20.0

# 미확인(unknown) 과일은 크기/모양을 전혀 모르는 상태라 일반 DEPTH_OFFSET_MM대로
# 내려가면 트레이/받침대와 충돌할 위험이 있음. SMALL_APPLE_DEPTH_OFFSET_MM을
# 그대로 재사용하지 않고, 나중에 실측으로 독립적으로 튜닝할 수 있도록 별도 상수로
# 둠 (지금은 SMALL_APPLE_DEPTH_OFFSET_MM과 같은 값에서 시작).
UNKNOWN_FRUIT_DEPTH_OFFSET_MM = DEPTH_OFFSET_MM + 20

# 변환 후 z가 이 값보다 낮게 나오면(예: depth 잡음/오탐으로 비정상적으로 얕게
# 나온 경우) 이 값으로 클램프해서 로봇이 테이블을 뚫고 내려가는 것을 방지.
MIN_Z_MM = 2.0

_gripper2cam_matrix = None  # 파일 I/O를 매 호출마다 하지 않도록 캐시


def _load_gripper2cam_matrix() -> np.ndarray:
    global _gripper2cam_matrix
    if _gripper2cam_matrix is None:
        package_path = get_package_share_directory(PACKAGE_NAME)
        npy_path = os.path.join(package_path, "resource", GRIPPER2CAM_FILENAME)
        _gripper2cam_matrix = np.load(npy_path)
    return _gripper2cam_matrix


def _robot_pose_matrix(x, y, z, rx, ry, rz) -> np.ndarray:
    """[x,y,z,rx,ry,rz](posx, ZYZ 오일러) -> 4x4 동차변환행렬."""
    rotation = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = [x, y, z]
    return matrix


def camera_to_base(camera_xyz, robot_posx, apply_offset: bool = True, depth_offset_mm: float = None):
    """
    카메라 좌표계의 3D 위치를 로봇 베이스 좌표계로 변환.

    Args:
        camera_xyz: (x, y, z) mm, vision의 position/pose (카메라 좌표계).
        robot_posx: [x, y, z, rx, ry, rz] mm/deg, 카메라 촬영 시점의 로봇 pose
            (get_current_posx()[0] 결과를 그대로 넘기면 됨).
        apply_offset: True면 z 보정을 적용 (depth_offset_mm + MIN_Z_MM 클램프).
        depth_offset_mm: z에 더할 보정값(mm). None이면 기본값 DEPTH_OFFSET_MM을
            씀. 작은 사과처럼 물리적으로 낮은 물체는 SMALL_APPLE_DEPTH_OFFSET_MM
            처럼 별도 값을 넘겨서 덜/더 내려가게 조정할 수 있음.

    Returns:
        (x, y, z) mm, 로봇 베이스 좌표계 기준 위치.
    """
    gripper2cam = _load_gripper2cam_matrix()
    coord = np.append(np.array(camera_xyz, dtype=float), 1.0)  # 동차좌표

    x, y, z, rx, ry, rz = robot_posx
    base2gripper = _robot_pose_matrix(x, y, z, rx, ry, rz)
    base2cam = base2gripper @ gripper2cam
    base_coord = base2cam @ coord

    bx, by, bz = base_coord[:3]
    if apply_offset:
        offset = DEPTH_OFFSET_MM if depth_offset_mm is None else depth_offset_mm
        bz += offset
        bz = max(bz, MIN_Z_MM)
    return float(bx), float(by), float(bz)
