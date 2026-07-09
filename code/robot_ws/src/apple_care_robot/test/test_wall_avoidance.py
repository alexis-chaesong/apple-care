import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from apple_care_robot.wall_avoidance import (
    is_within_tray_bounds, _nearest_wall_escape_dir, compute_wall_aware_approach,
    TRAY_X_MIN_MM, TRAY_X_MAX_MM, TRAY_Y_MIN_MM, TRAY_Y_MAX_MM,
    WALL_PROXIMITY_MARGIN_MM,
)


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
    min_dist, escape_dir, is_corner = _nearest_wall_escape_dir(x_near_max_wall, cy)
    assert escape_dir is not None
    assert is_corner is False


def test_nearest_wall_escape_dir_ignores_position_far_outside_x_max():
    cy = (TRAY_Y_MIN_MM + TRAY_Y_MAX_MM) / 2
    x_far_outside = TRAY_X_MAX_MM + 130.0  # 실제 로그에서 관측된 이탈 폭과 유사
    min_dist, escape_dir, is_corner = _nearest_wall_escape_dir(x_far_outside, cy)
    assert escape_dir is None
    assert is_corner is False


def test_nearest_wall_escape_dir_ignores_position_far_outside_y_min():
    cx = (TRAY_X_MIN_MM + TRAY_X_MAX_MM) / 2
    y_far_outside = TRAY_Y_MIN_MM - 130.0
    min_dist, escape_dir, is_corner = _nearest_wall_escape_dir(cx, y_far_outside)
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
