import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from apple_care_robot.wall_avoidance import (
    is_within_tray_bounds, _nearest_wall_escape_dir, compute_wall_aware_approach,
    _tilt_orientation, _nearest_zyz_branch,
    TRAY_X_MIN_MM, TRAY_X_MAX_MM, TRAY_Y_MIN_MM, TRAY_Y_MAX_MM,
    WALL_PROXIMITY_MARGIN_MM,
)
from apple_care_robot.pick_helpers import DEFAULT_PICK_ORIENTATION

# ry(-174.74)가 ZYZ 오일러 특이점(ry=180)에 가까워서 branch 선택 버그가 드러나는
# 실제 값이라 DEFAULT_PICK_ORIENTATION을 그대로 씀.
_REAL_PICK_ORIENTATION = DEFAULT_PICK_ORIENTATION


def test_is_within_tray_bounds_true_for_center_of_tray():
    cx = (TRAY_X_MIN_MM + TRAY_X_MAX_MM) / 2
    cy = (TRAY_Y_MIN_MM + TRAY_Y_MAX_MM) / 2
    assert is_within_tray_bounds(cx, cy) is True


def test_is_within_tray_bounds_true_exactly_on_edges():
    # 경계값 자체는 "안"으로 포함해야 함 (부등호가 <=/>=여야 함)
    assert is_within_tray_bounds(TRAY_X_MIN_MM, TRAY_Y_MIN_MM) is True
    assert is_within_tray_bounds(TRAY_X_MAX_MM, TRAY_Y_MAX_MM) is True


def test_is_within_tray_bounds_false_when_x_far_past_max():
    # 실측으로 확인된 문제 재현: x가 TRAY_X_MAX_MM을 100mm 이상 넘어간 경우.
    # _nearest_wall_escape_dir()는 이런 음수 거리도 "벽 근처"로 오판했었음 -
    # is_within_tray_bounds()는 이런 경우를 명확히 "범위 밖"으로 걸러내야 함.
    cy = (TRAY_Y_MIN_MM + TRAY_Y_MAX_MM) / 2
    assert is_within_tray_bounds(TRAY_X_MAX_MM + 130.0, cy) is False


def test_is_within_tray_bounds_false_when_x_past_min():
    cy = (TRAY_Y_MIN_MM + TRAY_Y_MAX_MM) / 2
    assert is_within_tray_bounds(TRAY_X_MIN_MM - 50.0, cy) is False


def test_is_within_tray_bounds_false_when_y_out_of_range():
    cx = (TRAY_X_MIN_MM + TRAY_X_MAX_MM) / 2
    assert is_within_tray_bounds(cx, TRAY_Y_MAX_MM + 50.0) is False
    assert is_within_tray_bounds(cx, TRAY_Y_MIN_MM - 50.0) is False


# ── _nearest_wall_escape_dir: 음수 거리(범위 밖) 오판 회귀 테스트 ─────────────
# 실측으로 확인된 버그: x가 TRAY_X_MAX_MM을 한참 넘어가면(margin이 음수) 예전
# 코드는 이걸 "벽 근처"로 오판해서 대각선 접근을 시도했음. 하한을 0으로 둬서
# 막았는지 확인함.

def test_nearest_wall_escape_dir_detects_normal_wall_proximity():
    # 벽에 실제로 가까운(margin이 0~WALL_PROXIMITY_MARGIN_MM 사이) 정상 케이스는
    # 계속 감지돼야 함 - 회귀로 아예 안 잡히게 되면 안 됨.
    cy = (TRAY_Y_MIN_MM + TRAY_Y_MAX_MM) / 2
    x_near_max_wall = TRAY_X_MAX_MM - (WALL_PROXIMITY_MARGIN_MM / 2)
    min_dist, escape_dir, is_corner, close_walls = _nearest_wall_escape_dir(x_near_max_wall, cy)
    assert escape_dir is not None
    assert is_corner is False


def test_nearest_wall_escape_dir_ignores_position_far_outside_x_max():
    cy = (TRAY_Y_MIN_MM + TRAY_Y_MAX_MM) / 2
    x_far_outside = TRAY_X_MAX_MM + 130.0  # 실제 로그에서 관측된 이탈 폭과 유사
    min_dist, escape_dir, is_corner, close_walls = _nearest_wall_escape_dir(x_far_outside, cy)
    assert escape_dir is None
    assert is_corner is False


def test_nearest_wall_escape_dir_ignores_position_far_outside_y_min():
    cx = (TRAY_X_MIN_MM + TRAY_X_MAX_MM) / 2
    y_far_outside = TRAY_Y_MIN_MM - 130.0
    min_dist, escape_dir, is_corner, close_walls = _nearest_wall_escape_dir(cx, y_far_outside)
    assert escape_dir is None
    assert is_corner is False


def test_compute_wall_aware_approach_does_nothing_for_far_outside_position():
    def _fake_posx(x, y, z, rx, ry, rz):
        return (x, y, z, rx, ry, rz)

    cy = (TRAY_Y_MIN_MM + TRAY_Y_MAX_MM) / 2
    pick_pos = (TRAY_X_MAX_MM + 130.0, cy, 10.0, 0.0, 180.0, 0.0)
    hover_pos, final_pick_pos = compute_wall_aware_approach(pick_pos, _fake_posx)
    assert hover_pos is None
    assert final_pick_pos == pick_pos


# ── 벽 방향별 기울임 방향/각도 회귀 테스트 ─────────────────────────────────
# 실측으로 확인된 버그: DEFAULT_PICK_ORIENTATION의 ry(-174.74)가 ZYZ 오일러
# 특이점(ry=180)에 가까워서, 벽마다 다르게 계산해야 할 기울임이 scipy의
# as_euler() branch 선택에 따라 rx/rz가 100~200도씩 튀는 표현으로 나올 때가
# 있었음 - 실제 로봇에서는 벽 방향과 무관하게 항상 비슷한(오른쪽) 방향으로만
# 기울어지는 것처럼 보이는 원인이었음. _nearest_zyz_branch()가 원래 자세에
# 더 가까운 branch를 고르는지, 그리고 그 결과 네 벽 모두 기울임 방향이
# 서로 달라지는지 확인한다.

def test_nearest_zyz_branch_picks_representation_closer_to_reference():
    reference = _REAL_PICK_ORIENTATION
    # reference와 동일한 회전의 "먼" branch를 후보로 넣어도, 더 가까운 쪽
    # (reference 자기 자신)을 골라야 함.
    far_branch = (
        reference[0] + 180.0 - 360.0,  # wrap 확인용으로 -180 방향으로도 넣어봄
        -reference[1],
        reference[2] + 180.0,
    )
    chosen = _nearest_zyz_branch(far_branch, reference)

    def _ang_dist(a, b):
        d = abs(a - b) % 360.0
        return min(d, 360.0 - d)

    # 핵심 불변조건: 선택된 branch는 항상 두 후보 중 reference에 더 가깝다 -
    # 여기서는 reference 자신이 압도적으로 더 가까우므로 chosen도 reference와
    # 거의 같아야 함(부동소수점 오차 허용).
    cost_chosen = sum(_ang_dist(c, r) for c, r in zip(chosen, reference))
    cost_far = sum(_ang_dist(c, r) for c, r in zip(far_branch, reference))
    assert cost_chosen <= cost_far
    assert cost_chosen < 1e-6


def test_tilt_orientation_stays_close_to_base_near_zyz_singularity():
    # DEFAULT_PICK_ORIENTATION처럼 ry가 180 근처인 자세를 4방향(상하좌우)으로
    # 기울여도, branch 선택 버그가 있던 예전과 달리 rx/rz가 100도 이상
    # 튀어서는 안 됨 (실측: 버그 있던 branch는 141~218도까지 튀었음).
    def _ang_dist(a, b):
        d = abs(a - b) % 360.0
        return min(d, 360.0 - d)

    for escape_dir in [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]:
        new_orientation = _tilt_orientation(_REAL_PICK_ORIENTATION, escape_dir, 15.0)
        for new_angle, base_angle in zip(new_orientation, _REAL_PICK_ORIENTATION):
            assert _ang_dist(new_angle, base_angle) < 100.0, (
                f"escape_dir={escape_dir} tilt={new_orientation} "
                f"base={_REAL_PICK_ORIENTATION}"
            )


def test_tilt_orientation_differs_per_wall_direction():
    # 왼쪽/오른쪽/위/아래 벽 각각에 대해 계산된 기울임 자세가 서로 달라야 함
    # (전부 같은 방향/각도로 기울면 안 된다는 요구사항의 핵심 회귀 테스트).
    orientations = {
        escape_dir: _tilt_orientation(_REAL_PICK_ORIENTATION, escape_dir, 15.0)
        for escape_dir in [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
    }
    values = list(orientations.values())
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            assert values[i] != values[j]


# ── 상단(y_max) 벽 "직선 접근" 예외 회귀 테스트 ─────────────────────────────
# 실측 요구사항: 상단(후면) 벽 근처에는 사과가 여러 개 나란히 놓이는 경우가 많아서,
# 다른 세 벽처럼 기울여 접근하면 hover 경유점이 옆으로 빠지면서 이웃 사과와 부딪힐
# 위험이 있음 - 그래서 이 벽 하나에만 가까운 경우(모서리 아님)는 기울이지 않고
# 자세를 그대로 유지해야 함(STRAIGHT_APPROACH_WALLS 참고).

def _make_pick_pos_near_y_max(offset_from_wall_mm):
    cx = (TRAY_X_MIN_MM + TRAY_X_MAX_MM) / 2
    y = TRAY_Y_MAX_MM - offset_from_wall_mm
    return (cx, y, 10.0, *_REAL_PICK_ORIENTATION)


def test_compute_wall_aware_approach_keeps_orientation_for_pure_y_max_wall():
    def _fake_posx(x, y, z, rx, ry, rz):
        return (x, y, z, rx, ry, rz)

    pick_pos = _make_pick_pos_near_y_max(WALL_PROXIMITY_MARGIN_MM / 2)
    hover_pos, tilted_pick_pos = compute_wall_aware_approach(pick_pos, _fake_posx)

    assert hover_pos is not None  # 벽 근처로는 감지돼야 함(hover 경유점은 여전히 생김)
    # 기울이지 않으므로 orientation(rx,ry,rz)은 원래 자세와 거의 같아야 함
    # (부동소수점 오차만 허용).
    for actual, expected in zip(hover_pos[3:], _REAL_PICK_ORIENTATION):
        assert abs(actual - expected) < 1e-6
    for actual, expected in zip(tilted_pick_pos[3:], _REAL_PICK_ORIENTATION):
        assert abs(actual - expected) < 1e-6


def test_compute_wall_aware_approach_still_tilts_for_y_max_corner():
    # y_max 벽 + x_max(오른쪽) 벽에 동시에 가까운 모서리는 "직선 접근" 예외에서
    # 빠지고, 기존처럼(모서리 tilt) 기울여야 함.
    def _fake_posx(x, y, z, rx, ry, rz):
        return (x, y, z, rx, ry, rz)

    x = TRAY_X_MAX_MM - (WALL_PROXIMITY_MARGIN_MM / 2)
    y = TRAY_Y_MAX_MM - (WALL_PROXIMITY_MARGIN_MM / 2)
    pick_pos = (x, y, 10.0, *_REAL_PICK_ORIENTATION)
    hover_pos, tilted_pick_pos = compute_wall_aware_approach(pick_pos, _fake_posx)

    assert hover_pos is not None
    assert hover_pos[3:] != _REAL_PICK_ORIENTATION
