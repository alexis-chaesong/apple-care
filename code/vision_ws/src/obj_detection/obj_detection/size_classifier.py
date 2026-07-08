"""
size_classifier.py
====================
ObjectDetectionNode에 섞어 쓰는(mixin) 사과 크기 재분류 로직.

depth로 잰 실제 지름(mm)을 최근 정상 사과들의 평균과 비교해서, 확연히 작으면
"apple_small"로 재분류한다.

YOLO가 이제 apple_small을 직접 학습해서 인식하지만, 이건 "이 사진 한 장만 보고
작다"고 판단하는 것이라 한계가 있다. 여기서 하는 재분류는 "같은 트레이에서 방금
본 다른 정상 사과들에 비해 상대적으로 작은지"를 depth 실측값으로 다시 확인하는
것이라, YOLO 단독 판단을 보완하는 이중 안전망 역할을 한다.

detection.py의 `class ObjectDetectionNode(..., SizeClassifierMixin, ...)`로
섞여서 쓰인다. self._normal_apple_sizes/_normal_sizes_lock은
ObjectDetectionNode.__init__에서 준비되고, self._real_world_diameter_mm은
depth_utils.DepthAnalysisMixin이 제공한다는 전제로 작성됨.
"""

# "apple_normal"로 판정된 사과들의 실제 지름(mm, depth로 환산) 이동 평균을 이용해,
# 그중 상대적으로 뚜렷하게 작은 사과를 "apple_small"로 재분류한다.
NORMAL_SIZE_HISTORY_LEN = 75
SMALL_APPLE_SIZE_RATIO = 0.8

# 비교할 정상 사과 표본이 1개 이하일 때는 평균 대비 비율을 신뢰할 수 없음
# (표본 1개=그 사과 자체가 기준이 되어 비교 자체가 무의미, 0개=비교 대상 없음).
# 이 경우엔 상대 비교 대신 절대 지름(mm) 기준으로 판단한다.
ABSOLUTE_SMALL_DIAMETER_MM = 60

# YOLO가 작은 사과를 "손상(apple_damaged)"으로 잘못 인식하는 경우가 있어,
# 이 라벨들에 한해서는 depth로 잰 실제 크기가 충분히 작으면 손상이 아니라
# "작아서 그런 것"으로 보고 apple_small로 재분류한다.
# (apple_rotten은 색/질감 변화가 핵심 근거라 크기 오탐과 무관하므로 대상에서 제외,
# apple_small은 이미 YOLO가 직접 그렇게 판정한 것이라 다시 검사할 필요 없음)
SIZE_OVERRIDE_LABELS = {"apple_normal", "apple_damaged"}


class SizeClassifierMixin:
    """depth 실측 지름을 기준으로 사과를 apple_small로 재분류하는 mixin."""

    def _classify_size(self, label, depth_frame, box, debug_info=None):
        """정상/손상 판정된 것들 중, 지금까지 본 정상 사과들의 평균 크기보다
        뚜렷하게 작은 경우 apple_small로 재분류한다.

        픽셀 박스 크기만 비교하면 카메라와의 거리 차이가 크기 차이와 섞여버린다
        (같은 사과라도 카메라에 가까울수록 픽셀상으로는 더 크게 찍힘). 위에서
        내려다보는 카메라 구조를 이용해, depth(카메라~사과 윗면 거리)와 intrinsics로
        박스를 실제 물리적 지름(mm)으로 환산한 뒤 비교하면 이 왜곡이 없어진다.

        YOLO가 작은 사과를 apple_damaged로 오인하는 경우가 있어서, apple_damaged도
        같은 기준으로 검사해 실제로는 "작을 뿐"이면 apple_small로 되돌린다.

        debug_info를 넘기면 판단에 쓴 diameter_mm/avg_diameter_mm을 채워 넣어서
        디버그 창 오버레이에서 그대로 보여줄 수 있게 한다.
        """
        if label not in SIZE_OVERRIDE_LABELS:
            return label

        diameter_mm = self._real_world_diameter_mm(depth_frame, box)
        if debug_info is not None:
            debug_info['diameter_mm'] = diameter_mm
        if diameter_mm is None:
            # depth를 못 구하면 크기 비교 없이 원래 라벨 그대로 둔다.
            return label

        sample_count, avg_diameter = self._normal_size_stats()
        if debug_info is not None:
            debug_info['avg_diameter_mm'] = avg_diameter

        final_label = self._preview_size_label(label, diameter_mm, avg_diameter, sample_count)
        if final_label != label:
            # sample_count<=1이면 avg_diameter가 None일 수 있음(절대 지름 기준으로
            # 판단한 경우 - _preview_size_label 참고), 로그에서 None을 그대로
            # ".1f"로 포매팅하면 TypeError가 나므로 별도 처리.
            avg_txt = f"{avg_diameter:.1f}mm" if avg_diameter is not None else "N/A(표본 부족)"
            self.get_logger().info(
                f"YOLO judged '{label}' but size({diameter_mm:.1f}mm) is just "
                f"small vs avg({avg_txt}) -> overriding to {final_label}"
            )

        # apple_small로 재분류되지 않은, 즉 평균 크기 산정에 쓸만한 정상 크기만 누적.
        # apple_damaged로 판정된 것은(재분류 여부와 무관하게) 기준 크기 계산에서 제외한다.
        if final_label == "apple_normal":
            self._add_normal_sample(diameter_mm)
        return final_label

    def _add_normal_sample(self, diameter_mm):
        with self._normal_sizes_lock:
            self._normal_apple_sizes.append(diameter_mm)

    def _normal_size_stats(self):
        """지금까지 쌓인 정상 사과 지름(mm) 표본의 (개수, 평균) - 표본 없으면 (0, None)."""
        with self._normal_sizes_lock:
            samples = list(self._normal_apple_sizes)
        if not samples:
            return 0, None
        return len(samples), sum(samples) / len(samples)

    def _preview_size_label(self, label, diameter_mm, avg_diameter, sample_count):
        """diameter_mm이 충분히 작으면 "apple_small"로 바꿔서 반환.

        정상 사과 표본이 여러 개(sample_count >= 2) 쌓였으면 그 평균(avg_diameter)
        대비 비율(SMALL_APPLE_SIZE_RATIO)로 판단한다.
        표본이 1개 이하면 평균 자체를 신뢰할 수 없으므로(표본 1개짜리 "평균"은
        그 사과 자신이라 비교가 무의미, 0개면 비교 대상이 없음), 대신 절대
        지름(ABSOLUTE_SMALL_DIAMETER_MM) 기준으로 판단한다.

        `_classify_size`의 실제 판정과 depth 디버그 창의 실시간 미리보기(여러 박스를
        한꺼번에 그릴 때)가 동일한 기준을 쓰도록 순수 계산만 분리한 함수. 상태(이동
        평균 deque)를 건드리지 않으므로 미리보기 용도로 반복 호출해도 안전하다.
        """
        if label not in SIZE_OVERRIDE_LABELS or diameter_mm is None:
            return label
        if sample_count <= 1:
            if diameter_mm <= ABSOLUTE_SMALL_DIAMETER_MM:
                return "apple_small"
            return label
        if avg_diameter is not None and diameter_mm < avg_diameter * SMALL_APPLE_SIZE_RATIO:
            return "apple_small"
        return label
