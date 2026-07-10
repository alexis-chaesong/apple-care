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
따로 쓰고, 최종 grasp 지점(pick_pos)도 기울인 뒤의 실제 툴 자세 기준 -Y축
방향으로 CORNER_PICK_INSET_MM만큼 TCP를 이동시켜서 마지막 하강 지점의 여유를
조정한다 (벽 하나만 가까운 경우는 기존처럼 pick_pos를 그대로 둠).

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

    실측으로 확인된 문제: DEFAULT_PICK_ORIENTATION의 ry(-174.74)가 ZYZ 오일러
    특이점(ry=180)에 아주 가까워서, 벽마다 다르게 계산한 회전을 다시 (rx,ry,rz)로
    분해할 때 scipy가 원래 자세와 동떨어진 표현(rx/rz가 100~200도씩 튐)을 고르는
    경우가 있었음 - 이 경우 실제 로봇은 명령받은 각도에 정확히 도달하지 못하고
    벽 방향과 무관하게 항상 비슷한 쪽으로 기울어 보이는 문제가 있었음.
    _nearest_zyz_branch()로 원래 자세에 가까운 branch를 고르도록 고쳐서 해결함.

    실측으로 확인된 문제 (2): 위 특이점 문제를 고친 뒤에도, 실기 사진으로 확인한
    결과 왼쪽/오른쪽 벽에서 그리퍼가 기울어지는 방향이 서로 반대로(의도와 정반대로)
    나타남. 원인은 회전축 부호(ey, -ex, 0)가 "그리퍼가 접근하는 방향(팁->손목
    반대쪽) 벡터"를 트레이 중심 쪽으로 기울이도록 잡혀 있었던 것 - pick_pos(사과
    위치, 그립 지점)는 고정된 채로 이 벡터만 회전하므로, 실제로는 손목/몸통이
    벽 쪽으로 더 쏠리는 반대 결과가 됨. 축 부호를 (-ey, ex, 0)으로 뒤집어서
    손목/몸통이 벽 반대쪽(트레이 중심)으로 빠지도록 고침.
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
#
# 실측으로 확인된 문제: 회전 방향 자체는 좌/우/전면/후면/모서리 전부 수학적으로
# 대칭이고 올바르게 계산되는데도(_tilt_orientation 참고), 실기 사진상 전면 벽
# 근처에서는 대각선 오프셋이 거의 안 걸린 것처럼 보이는 사례가 있었음 - 비전이
# 준 pick 좌표가 실제 벽까지의 거리보다 약간 더 크게(예: 40~50mm) 나오는 오차가
# 있으면, 이 판정 자체가 "벽 근처 아님"으로 걸러져서 티트/오프셋이 아예 적용
# 안 됨. 40mm는 오차 여유가 거의 없어서, 여유를 좀 더 두어 비전/실측 오차가
# 있어도 확실히 걸리도록 늘림.
WALL_PROXIMITY_MARGIN_MM = 45.0

# 벽 근처 접근 시 수직에서 트레이 중심 방향으로 기울이는 각도(도).
WALL_APPROACH_TILT_DEG = 10.0

# 대각선 경유점을 pick_pos 대비 트레이 중심 쪽으로 얼마나 옆으로, 얼마나 높이 띄울지(mm).
WALL_APPROACH_LATERAL_OFFSET_MM = 22.0
WALL_APPROACH_HOVER_CLEARANCE_MM = 30.0

# 모서리(두 벽에 동시에 WALL_PROXIMITY_MARGIN_MM 이내로 가까움)는 벽 하나만 가까운
# 경우보다 그리퍼가 움직일 수 있는 여유 공간이 훨씬 좁다. 실측 결과 벽 하나 기준으로
# 튜닝한 WALL_APPROACH_TILT_DEG/OFFSET 그대로 쓰면 모서리에서 그리퍼가 벽에 스치는
# 문제가 있어서, 모서리 전용으로 더 큰 값을 따로 둔다.
CORNER_APPROACH_TILT_DEG = 22.0
CORNER_APPROACH_LATERAL_OFFSET_MM = 45.0
CORNER_APPROACH_HOVER_CLEARANCE_MM = 50.0

# 모서리에서는 hover 경유점만 우회시키고 최종 pick_pos는 원래 사과 위치 그대로
# 두면, 마지막 하강 구간에서 결국 다시 두 벽에 바짝 붙은 지점까지 내려가야 해서
# 그립 순간 손가락이 스칠 수 있다. hover 단계에서 이미 충분히 우회했으므로,
# 최종 grasp 지점은 기울인 뒤의 실제 툴 자세 기준 -Y축 방향으로 이 값(mm)만큼
# TCP를 더 이동시켜서 여유를 조정한다 (베이스 프레임 escape_dir이 아니라 툴
# -Y축 기준 - compute_wall_aware_approach 참고).
CORNER_PICK_INSET_MM = 5.0

# 실측으로 확인된 문제: 벽 하나만 가까워서 기울여 접근하는 경우(모서리 아님),
# 최종 grasp 지점을 pick_pos 그대로 두면 기울인 자세 기준으로 그리퍼가 벽 쪽
# 사과를 살짝 덜 파고들어(그립 위치가 벽 반대쪽으로 조금 치우쳐) 파지가
# 불안정해지는 사례가 있었음. 모서리와 동일한 원리로, 기울인 뒤의 실제 툴 자세
# 기준 -Y축 방향으로 이만큼(mm) TCP를 더 이동시켜 보정한다(실측 튜닝 범위 5~8mm
# 중간값). 기울이지 않는 경우(STRAIGHT_APPROACH_WALLS)는 적용하지 않음 - 그
# 경우는 tilt 자체가 없어 이 보정의 전제(기울인 방향으로 덜 들어감)가 성립하지 않음.
WALL_PICK_INSET_MM = 6.5


def is_within_tray_bounds(x, y) -> bool:
    """(x, y)가 트레이 작업 영역(TRAY_X_MIN_MM~MAX_MM, TRAY_Y_MIN_MM~MAX_MM) 안에
    있는지 확인한다.

    실측으로 확인된 문제: vision이 준 위치가 이 트레이 범위를 한참 벗어난 경우
    (예: x가 TRAY_X_MAX_MM보다 100mm 이상 큼), _nearest_wall_escape_dir()의 거리
    계산이 "경계를 얼마나 벗어났는지"를 나타내는 음수값도 WALL_PROXIMITY_MARGIN_MM
    이내로 오판해서 "벽 근처"로 잘못 처리하는 버그가 있었음(범위 안/벽 근처/범위
    밖 세 경우를 구분하지 못했음). 그 결과 실제로는 존재하지 않는 위치로 접근을
    시도하다 파지가 반복 실패하는 사례가 있었음 - 그래서 애초에 파지 시도 자체를
    이 범위 안으로만 제한하기 위해 별도로 둠(호출부: box_sequence_test.py).
    """
    return TRAY_X_MIN_MM <= x <= TRAY_X_MAX_MM and TRAY_Y_MIN_MM <= y <= TRAY_Y_MAX_MM


# "상단(후면)" 벽(y_max) 하나에만 가까운 경우(모서리 아님)는 실측 결과 다른 세 벽과
# 달리 그리퍼를 기울이면 안 됨 - 이 벽 쪽에는 사과가 여러 개 나란히 놓이는 경우가
# 많아서, tilt 때문에 옆으로 빠지는 hover 경로가 바로 옆 사과와 부딪힐 위험이 있음.
# 그래서 이 벽만 "직선 접근"(기울이지 않고, hover 위치만 트레이 중심 쪽/위로 살짝
# 띄움)으로 예외 처리함. 모서리(다른 벽과 동시에 가까움)는 기존처럼 기울임 -
# 그 경우는 이미 실측으로 문제없이 확인됨.
STRAIGHT_APPROACH_WALLS = frozenset({"y_max"})


def _nearest_wall_escape_dir(x, y):
    """
    (x, y)가 트레이 경계 중 어느 벽에라도 WALL_PROXIMITY_MARGIN_MM 이내로
    가까우면, 그 벽에서 트레이 중심 쪽으로 벗어나는 정규화된 방향(ex, ey)과
    "모서리(두 벽에 동시에 가까움)인지" 여부, 가까운 벽 이름들의 집합을 반환.
    모서리면 두 방향을 합성해 대각선 방향으로 줌. 벽 근처가 아니면
    (None, None, False, frozenset()).

    이 함수는 (x, y)가 트레이 범위 "안"(is_within_tray_bounds)이라는 전제 하에
    호출돼야 함 - 범위를 이미 벗어난 좌표(예: x > TRAY_X_MAX_MM)를 넣으면 해당
    축의 margin이 음수가 되는데, 실측으로 확인된 버그: 예전엔 `dist <=
    WALL_PROXIMITY_MARGIN_MM` 조건이 음수도 그냥 통과시켜서 "벽에 딱 붙어있음"과
    "범위를 한참 벗어남"을 구분 못 했음(둘 다 "벽 근처"로 오판). 그래서 하한을
    0으로 둬서, 이미 경계를 넘어선 축은 "근접"으로 안 치도록 막음. 정상 흐름에서는
    box_sequence_test.py가 pick_apple() 호출 전에 is_within_tray_bounds()로 이미
    걸러주므로 이 하한에 실제로 걸릴 일은 없어야 정상이지만, 이 함수 자체의
    불변조건을 명확히 하기 위해 둠.
    """
    margins = {
        "x_min": (x - TRAY_X_MIN_MM, (1.0, 0.0)),
        "x_max": (TRAY_X_MAX_MM - x, (-1.0, 0.0)),
        "y_min": (y - TRAY_Y_MIN_MM, (0.0, 1.0)),
        "y_max": (TRAY_Y_MAX_MM - y, (0.0, -1.0)),
    }

    close = [
        (name, dist, direction) for name, (dist, direction) in margins.items()
        if 0 <= dist <= WALL_PROXIMITY_MARGIN_MM
    ]
    if not close:
        return None, None, False, frozenset()

    close_walls = frozenset(name for name, _, _ in close)
    is_corner = len(close) >= 2
    min_dist = min(dist for _, dist, _ in close)
    ex = sum(d[0] for _, _, d in close)
    ey = sum(d[1] for _, _, d in close)
    norm = (ex ** 2 + ey ** 2) ** 0.5
    if norm < 1e-6:
        return min_dist, None, is_corner, close_walls
    return min_dist, (ex / norm, ey / norm), is_corner, close_walls


def _wrap_deg(angle_deg):
    """각도를 (-180, 180] 범위로 정규화."""
    return (angle_deg + 180.0) % 360.0 - 180.0


def _nearest_zyz_branch(candidate, reference):
    """
    ZYZ 오일러(proper Euler, 축 반복)는 같은 회전을 나타내는 두 가지 표현이
    있음: (rx,ry,rz)와 (rx+180,-ry,rz+180)(mod 360)는 동일한 회전.

    DEFAULT_PICK_ORIENTATION의 ry(-174.74)가 ZYZ 특이점(ry=180)에 아주 가까워서,
    scipy의 as_euler()가 두 표현 중 reference(원래 자세)와 동떨어진 쪽을 골라주는
    경우가 있었음 - 실측 결과 이 경우 rx/rz가 100~200도씩 튀어서, 벽 방향에 따라
    의도한 만큼만 기울여야 할 자세가 실제로는 큰 폭으로 달라진 손목 자세로
    지시되고, 그 결과 로봇이 명령받은 각도에 정확히 도달하지 못하고 매번 비슷한
    방향으로 수렴하는 문제(모든 벽에서 같은 방향/각도로 기울어 보이는 문제)로
    이어졌음. reference(기존 자세)에 더 가까운 쪽 branch를 골라서, 최소한의
    관절 이동만으로 목표 자세에 도달하게 하여 이 문제를 없앤다.
    """
    rx, ry, rz = candidate
    alt = (_wrap_deg(rx + 180.0), -ry, _wrap_deg(rz + 180.0))

    def _cost(cand):
        return sum(abs(_wrap_deg(c - r)) for c, r in zip(cand, reference))

    return candidate if _cost(candidate) <= _cost(alt) else alt


def _tilt_orientation(base_orientation, escape_dir_xy, tilt_deg):
    """
    base_orientation(rx,ry,rz ZYZ 오일러 - 그리퍼가 거의 수직 아래를 보는 자세)를
    escape_dir_xy(베이스 좌표계 XY 평면, 벽 반대/트레이 중심 방향) 쪽으로
    tilt_deg만큼 기울인 새 (rx,ry,rz)를 반환.

    회전축을 escape_dir_xy와 수직인 수평축(-ey, ex, 0)으로 잡고, 베이스(월드)
    좌표계 기준으로 base_orientation 앞에 이 회전을 곱함 - 그래야 그리퍼가
    "제자리에서 자기 축 기준"이 아니라 "트레이 중심을 향해" 기울어짐.

    실측으로 확인된 문제: 축을 (ey, -ex, 0)으로 두면(이전 구현), pick_pos(사과
    위치, 고정)를 기준으로 회전시키기 때문에 "그리퍼가 접근하는 방향(팁->손목
    반대쪽) 벡터"가 트레이 중심 쪽으로 기우는 대신, 실제로는 손목/몸통 쪽이
    벽 쪽으로 더 쏠리는 반대 결과가 나옴(왼쪽 벽에서 오른쪽 벽에서와 반대로
    동작하는 것처럼 보였음). 축 부호를 (-ey, ex, 0)으로 뒤집어서 손목/몸통이
    벽 반대쪽(트레이 중심)으로 빠지도록 고침.
    """
    ex, ey = escape_dir_xy
    axis = np.array([-ey, ex, 0.0])
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-6:
        return base_orientation
    axis = axis / axis_norm

    r_base = Rotation.from_euler("ZYZ", base_orientation, degrees=True)
    r_tilt = Rotation.from_rotvec(np.radians(tilt_deg) * axis)
    raw = tuple(float(v) for v in (r_tilt * r_base).as_euler("ZYZ", degrees=True))
    new_rx, new_ry, new_rz = _nearest_zyz_branch(raw, base_orientation)
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
    _, escape_dir, is_corner, close_walls = _nearest_wall_escape_dir(x, y)
    if escape_dir is None:
        return None, pick_pos

    tilt_deg = CORNER_APPROACH_TILT_DEG if is_corner else WALL_APPROACH_TILT_DEG
    lateral_offset_mm = (
        CORNER_APPROACH_LATERAL_OFFSET_MM if is_corner else WALL_APPROACH_LATERAL_OFFSET_MM
    )
    hover_clearance_mm = (
        CORNER_APPROACH_HOVER_CLEARANCE_MM if is_corner else WALL_APPROACH_HOVER_CLEARANCE_MM
    )

    # 상단(y_max) 벽 하나에만 가까운 경우는 기울이지 않고 "직선 접근"으로 예외
    # 처리 - STRAIGHT_APPROACH_WALLS 정의 참고 (여러 사과가 나란히 놓이는 벽이라
    # tilt로 인한 옆 이동이 이웃 사과와 부딪힐 위험이 있음). 모서리는 제외(그대로 기울임).
    if not is_corner and close_walls <= STRAIGHT_APPROACH_WALLS:
        tilt_deg = 0.0

    # pick_inset_mm은 반드시 위의 STRAIGHT_APPROACH_WALLS 예외 처리(tilt_deg=0.0)
    # 이후에 계산해야 함 - 기울이지 않는 경우까지 보정을 걸면 안 되므로 최종
    # tilt_deg 값을 기준으로 판단한다.
    if is_corner:
        pick_inset_mm = CORNER_PICK_INSET_MM
    elif tilt_deg > 0:
        pick_inset_mm = WALL_PICK_INSET_MM
    else:
        pick_inset_mm = 0.0

    ex, ey = escape_dir
    new_rx, new_ry, new_rz = _tilt_orientation((rx, ry, rz), escape_dir, tilt_deg)

    hover_x = x + ex * lateral_offset_mm
    hover_y = y + ey * lateral_offset_mm
    hover_z = z + hover_clearance_mm

    # 최종 grasp 지점(pick_inset)은 베이스 프레임 escape_dir이 아니라, 기울인 뒤의
    # 실제 툴 자세(new_rx,new_ry,new_rz) 기준 -Y축 방향으로 pick_inset_mm만큼
    # 이동시킨다 - 즉 "TCP를 툴좌표 -Y로 N mm"라는 요구사항을 그대로 반영함
    # (+Y가 아니라 -Y이므로 벽 쪽으로 더 다가가는 방향).
    tool_y = Rotation.from_euler(
        "ZYZ", (new_rx, new_ry, new_rz), degrees=True
    ).as_matrix()[:, 1]
    pick_x = x - tool_y[0] * pick_inset_mm
    pick_y = y - tool_y[1] * pick_inset_mm
    pick_z = z - tool_y[2] * pick_inset_mm

    # 안전장치: DEFAULT_PICK_ORIENTATION이 ZYZ 오일러 특이점에 가깝다는 게 이미
    # 실측으로 확인된 문제라(docstring 상단 "실측으로 확인된 문제" 참고),
    # _nearest_zyz_branch로 branch를 골라도 특정 벽 각도 조합에서 tool_y 축이
    # 예상과 다른 방향을 가리키면 hover_x/y나 pick_x/y가 트레이 밖 엉뚱한 곳으로
    # 크게 튈 수 있다. 이 조정된 최종 좌표는 box_sequence_test.py의
    # is_within_tray_bounds() 사전 검사(compute_wall_aware_approach 호출 *전*의
    # 원본 vision 좌표 기준)를 이미 지난 뒤 여기서 새로 계산되므로 그 검사를
    # 안 거친 채 그대로 movel 목표가 된다 - 그래서 여기서 한 번 더 확인해서
    # 벗어났으면 이번 tilt/inset 보정 자체를 포기하고 원래(보정 전) pick_pos로
    # 직선 접근하도록 안전하게 되돌린다.
    if not (is_within_tray_bounds(hover_x, hover_y) and is_within_tray_bounds(pick_x, pick_y)):
        return None, pick_pos

    hover_pos = posx_factory(hover_x, hover_y, hover_z, new_rx, new_ry, new_rz)
    tilted_pick_pos = posx_factory(pick_x, pick_y, pick_z, new_rx, new_ry, new_rz)
    return hover_pos, tilted_pick_pos
