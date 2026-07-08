"""
Tray Wall Avoidance
=====================

사과가 담긴 트레이(받침대)의 테두리 벽에 가까이 있을 때, pick_pos로 수직 그대로
접근하면 그리퍼(또는 벌어진 손가락)가 벽에 부딪힘. 이를 피하려고 사과가 트레이
가장자리 근처에 있는 경우에만 아래 두 가지를 같이 적용해서 "벽 반대쪽(트레이
중심 방향)에서 비스듬히 내려가는" 접근을 만든다:

    1) 그리퍼 자세(rx,ry,rz)를 트레이 중심 방향으로 WALL_APPROACH_TILT_DEG만큼
       기울임 (수직 하강 대신 살짝 기울어진 자세로 사과에 접근/파지).
    2) pick_pos 도달 전에, pick_pos보다 트레이 중심 쪽으로 더 들어가 있고 더 높은
       위치(경유점)를 먼저 거치게 해서, 경유점 -> pick_pos 구간이 대각선 경로가
       되도록 함.

벽에서 안 먼(=WALL_PROXIMITY_MARGIN_MM보다 먼) 사과는 기존과 동일하게 그냥
바로 pick_pos로 접근함 (이 모듈이 관여하지 않음).

모서리(두 벽에 동시에 가까움)는 벽 하나만 가까운 경우보다 공간이 훨씬 좁아서,
실측 결과 벽 하나 기준으로 튜닝한 tilt/offset 그대로는 그리퍼가 여전히 벽에
스치는 문제가 있었다. 그래서 모서리는 CORNER_APPROACH_* 값(더 큰 tilt/offset)을
따로 쓰고, 최종 grasp 지점(pick_pos)도 CORNER_PICK_INSET_MM만큼 트레이 중심
쪽으로 같이 당겨서 마지막 하강 지점의 여유도 확보한다 (벽 하나만 가까운 경우는
기존처럼 pick_pos를 그대로 둠).

주의 - 트레이 경계값 관련:
    TRAY_X_MIN_MM/TRAY_X_MAX_MM/TRAY_Y_MIN_MM/TRAY_Y_MAX_MM은 실제 작업대
    왼쪽위/오른쪽아래 모서리를 로봇을 직접 움직여 찍은 실측 posx 값
    (왼쪽위 (331.42, 116.98, 13.53, 7.22, -178.1, 2.22), 오른쪽아래
    (607.58, -158.69, 13.3, 90.99, -177.12, 92.46))의 x/y만 가져온 것.
    z/rx/ry/rz는 이 모듈에서 쓰지 않음(트레이 경계는 x/y 평면 기준으로만 판단).

주의 - 기울임 방향 관련:
    _tilt_orientation()의 회전축 부호는 "그리퍼가 거의 수직 아래를 보는 자세
    (PICK_ORIENTATION/DEFAULT_PICK_ORIENTATION)"라는 가정 하에 계산한 것이라,
    실제 로봇에서 처음 테스트할 때 기울어지는 방향(벽 쪽 vs 트레이 중심 쪽)을
    반드시 눈으로 확인할 것. 반대로 기울면 WALL_APPROACH_TILT_DEG 값의 부호만
    뒤집으면 됨.
"""

import numpy as np
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# 실측값 - 작업대 왼쪽위 (331.42, 116.98) / 오른쪽아래 (607.58, -158.69) 모서리의
# x/y (로봇 베이스 좌표계, mm). 왼쪽위가 x_min/y_max, 오른쪽아래가 x_max/y_min.
# ---------------------------------------------------------------------------
TRAY_X_MIN_MM = 331.42
TRAY_X_MAX_MM = 607.58
TRAY_Y_MIN_MM = -158.69
TRAY_Y_MAX_MM = 116.98

# pick_pos가 트레이 벽으로부터 이 거리(mm) 안이면 "벽 근처"로 보고 대각선 접근을 적용.
WALL_PROXIMITY_MARGIN_MM = 40.0

# 벽 근처 접근 시 수직에서 트레이 중심 방향으로 기울이는 각도(도).
WALL_APPROACH_TILT_DEG = 15.0

# 대각선 경유점을 pick_pos 대비 트레이 중심 쪽으로 얼마나 옆으로, 얼마나 높이 띄울지(mm).
WALL_APPROACH_LATERAL_OFFSET_MM = 30.0
WALL_APPROACH_HOVER_CLEARANCE_MM = 40.0

# 모서리(두 벽에 동시에 WALL_PROXIMITY_MARGIN_MM 이내로 가까움)는 벽 하나만 가까운
# 경우보다 그리퍼가 움직일 수 있는 여유 공간이 훨씬 좁다. 실측 결과 벽 하나 기준으로
# 튜닝한 WALL_APPROACH_TILT_DEG/OFFSET 그대로 쓰면 모서리에서 그리퍼가 벽에 스치는
# 문제가 있어서, 모서리 전용으로 더 큰 값을 따로 둔다.
CORNER_APPROACH_TILT_DEG = 22.0
CORNER_APPROACH_LATERAL_OFFSET_MM = 45.0
CORNER_APPROACH_HOVER_CLEARANCE_MM = 50.0

# 모서리에서는 hover 경유점만 우회시키고 최종 pick_pos(x,y)는 원래 사과 위치 그대로
# 두면, 마지막 하강 구간에서 결국 다시 두 벽에 바짝 붙은 지점까지 내려가야 해서
# 그립 순간 손가락이 스칠 수 있다. 사과 자체 반지름(가장 작은 apple_small 기준
# ABSOLUTE_SMALL_DIAMETER_MM/2=30mm)보다 충분히 작은 값만큼 트레이 중심 쪽으로
# 최종 grasp 지점도 같이 당겨서, 사과를 놓치지 않는 선에서 여유를 더 번다.
# 벽 하나만 가까운 경우는 기존처럼 pick_pos를 건드리지 않는다(그동안 문제 없었음).
CORNER_PICK_INSET_MM = 10.0


def _nearest_wall_escape_dir(x, y):
    """
    (x, y)가 트레이 경계 중 어느 벽에라도 WALL_PROXIMITY_MARGIN_MM 이내로
    가까우면, 그 벽에서 트레이 중심 쪽으로 벗어나는 정규화된 방향(ex, ey)과
    "모서리(두 벽에 동시에 가까움)인지" 여부를 반환. 모서리면 두 방향을 합성해
    대각선 방향으로 줌. 벽 근처가 아니면 (None, None, False).
    """
    margins = {
        "x_min": (x - TRAY_X_MIN_MM, (1.0, 0.0)),
        "x_max": (TRAY_X_MAX_MM - x, (-1.0, 0.0)),
        "y_min": (y - TRAY_Y_MIN_MM, (0.0, 1.0)),
        "y_max": (TRAY_Y_MAX_MM - y, (0.0, -1.0)),
    }

    close = [(dist, direction) for dist, direction in margins.values() if dist <= WALL_PROXIMITY_MARGIN_MM]
    if not close:
        return None, None, False

    is_corner = len(close) >= 2
    min_dist = min(dist for dist, _ in close)
    ex = sum(d[0] for _, d in close)
    ey = sum(d[1] for _, d in close)
    norm = (ex ** 2 + ey ** 2) ** 0.5
    if norm < 1e-6:
        return min_dist, None, is_corner
    return min_dist, (ex / norm, ey / norm), is_corner


def _tilt_orientation(base_orientation, escape_dir_xy, tilt_deg):
    """
    base_orientation(rx,ry,rz ZYZ 오일러 - 그리퍼가 거의 수직 아래를 보는 자세)를
    escape_dir_xy(베이스 좌표계 XY 평면, 벽 반대/트레이 중심 방향) 쪽으로
    tilt_deg만큼 기울인 새 (rx,ry,rz)를 반환.

    회전축을 escape_dir_xy와 수직인 수평축(ey, -ex, 0)으로 잡고, 베이스(월드)
    좌표계 기준으로 base_orientation 앞에 이 회전을 곱함 - 그래야 그리퍼가
    "제자리에서 자기 축 기준"이 아니라 "트레이 중심을 향해" 기울어짐.
    """
    ex, ey = escape_dir_xy
    axis = np.array([ey, -ex, 0.0])
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-6:
        return base_orientation
    axis = axis / axis_norm

    r_base = Rotation.from_euler("ZYZ", base_orientation, degrees=True)
    r_tilt = Rotation.from_rotvec(np.radians(tilt_deg) * axis)
    new_rx, new_ry, new_rz = (r_tilt * r_base).as_euler("ZYZ", degrees=True)
    return (float(new_rx), float(new_ry), float(new_rz))


def compute_wall_aware_approach(pick_pos, posx_factory):
    """
    pick_pos가 트레이 벽 근처면 (hover_pos, tilted_pick_pos)를 반환해서
    호출하는 쪽이 hover_pos -> tilted_pick_pos 순으로 movel하게 한다
    (경유점에서 pick_pos로 내려가는 구간이 대각선이 됨).
    벽 근처가 아니면 (None, pick_pos)를 반환 - 기존처럼 바로 pick_pos로 접근.

    Args:
        pick_pos: [x,y,z,rx,ry,rz] (posx 값, 언패킹 가능하면 됨).
        posx_factory: DSR_ROBOT2.posx (numpy/scipy와 DSR_ROBOT2의 import 순서
            제약을 피하려고 호출하는 쪽에서 넘겨받음).

    Returns:
        (hover_pos 또는 None, 실제로 movel할 최종 pick_pos)
    """
    x, y, z, rx, ry, rz = pick_pos
    _, escape_dir, is_corner = _nearest_wall_escape_dir(x, y)
    if escape_dir is None:
        return None, pick_pos

    tilt_deg = CORNER_APPROACH_TILT_DEG if is_corner else WALL_APPROACH_TILT_DEG
    lateral_offset_mm = (
        CORNER_APPROACH_LATERAL_OFFSET_MM if is_corner else WALL_APPROACH_LATERAL_OFFSET_MM
    )
    hover_clearance_mm = (
        CORNER_APPROACH_HOVER_CLEARANCE_MM if is_corner else WALL_APPROACH_HOVER_CLEARANCE_MM
    )
    pick_inset_mm = CORNER_PICK_INSET_MM if is_corner else 0.0

    ex, ey = escape_dir
    new_rx, new_ry, new_rz = _tilt_orientation((rx, ry, rz), escape_dir, tilt_deg)

    hover_x = x + ex * lateral_offset_mm
    hover_y = y + ey * lateral_offset_mm
    hover_z = z + hover_clearance_mm

    pick_x = x + ex * pick_inset_mm
    pick_y = y + ey * pick_inset_mm

    hover_pos = posx_factory(hover_x, hover_y, hover_z, new_rx, new_ry, new_rz)
    tilted_pick_pos = posx_factory(pick_x, pick_y, z, new_rx, new_ry, new_rz)
    return hover_pos, tilted_pick_pos
