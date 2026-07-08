"""
depth_utils.py
===============
ObjectDetectionNode에 섞어 쓰는(mixin) depth 분석 로직 모음.

- 배경(트레이) 캘리브레이션과 픽셀별 배경 차분 (_calibrate_scene, _foreground_mask)
- YOLO 박스 안/밖 depth 비교로 "진짜 튀어나온 물체인지" 검증 (_box_ring_depths)
- YOLO가 못 찾은 물체를 depth 덩어리로 찾기 (_find_all_unknown_blobs 등)
- 픽셀 좌표 <-> 카메라 좌표계 변환, 실측 지름 계산

detection.py의 `class ObjectDetectionNode(DepthAnalysisMixin, ...)`로 섞여서 쓰인다.
여기 정의된 메서드들은 self.img_node, self.intrinsics, self.background_distance_mm,
self.background_depth_map, self._wait_for_valid_data, self.get_logger() 등이
(ObjectDetectionNode 본체 또는 다른 mixin에 의해) 이미 준비돼 있다는 전제로 작성됨.
"""

import time

import cv2
import numpy as np

# 트레이/배경까지의 대략적인 거리(mm) - 기본값(fallback)일 뿐, 실제로는
# ObjectDetectionNode.__init__에서 _calibrate_scene()이 시작 시점의 depth 프레임
# 전체를 self.background_depth_map(픽셀별 배경 depth)으로 저장해서 쓴다.
#
# 예전엔 "배경까지 거리"를 스칼라 값 하나로 캘리브레이션해서 화면 전체에 같은
# 기준을 적용했는데, 트레이 바닥이 완전히 평평하지 않거나(기울어짐/굴곡) 시야에
# 펜스처럼 원래도 바닥보다 튀어나온 고정 구조물이 있으면, 스칼라 기준 하나로는
# 그 구조물/굴곡을 계속 "물체"로 오인했다 (펜스가 화면의 70%까지 잡히는 문제가
# 재현됨). 픽셀마다 자기 자신의 캘리브레이션 시점 depth를 기준으로 삼으면, 바닥이
# 기울어져 있든 펜스가 있든 "원래 거기 있던 배경보다 가까워졌는지"만 보게 되어
# 훨씬 안정적이다 (펜스 위에 사과가 올라가면 그 자리만 정확히 감지됨).
BACKGROUND_DISTANCE_MM = 500
PRESENCE_MARGIN_MM = 10
MIN_PRESENCE_PIXELS = 500

# 배경 캘리브레이션에 쓸 depth 프레임을 이 시간(초) 동안 모아서 픽셀별 median을
# 계산한다 (한 프레임의 일시적 노이즈/반사로 인한 무효 픽셀에 강해지도록).
BACKGROUND_CALIBRATION_DURATION_SEC = 1.5
BACKGROUND_CALIBRATION_SAMPLE_INTERVAL_SEC = 0.05

# depth 센서 노이즈로 생기는 가늘고 작은 잡음 덩어리를 미리 끊어내기 위한
# morphological opening 커널 크기(px). 사과처럼 두꺼운 덩어리는 그대로 남는다.
NOISE_OPEN_KERNEL_PX = 9

# YOLO 박스 안쪽 깊이와 바로 바깥 주변 깊이의 차이가 이 값(mm) 이상일 때만
# "물체가 배경보다 튀어나와 있다"고 보고 실제 감지로 표시한다.
# (아래 INNER_DEPTH_PERCENTILE로 박스 안쪽의 "정점 부근" 값을 쓰게 되어 이전보다
# 실제 사과 높이에 가깝게 측정되므로, median 기준으로 튜닝했던 값(100mm)은 너무
# 커서 실제 사과도 거의 다 FAIL 처리됨 - 사과 지름(약 60~90mm) 이하로 낮춰야 함)
HEIGHT_DIFF_MARGIN_MM = 20
# 박스 바깥 주변 깊이를 샘플링할 테두리 두께(px)
HEIGHT_DIFF_RING_PX = 20
# 박스 안쪽 depth 중 몇 번째 percentile을 "정점(사과 꼭대기)"으로 볼지.
# 낮을수록 카메라에 더 가까운(=값이 작은) 픽셀 쪽으로 치우쳐서 잡음.
INNER_DEPTH_PERCENTILE = 30


class DepthAnalysisMixin:
    """depth 프레임 기반 배경 모델링/물체 탐지/좌표 변환을 제공하는 mixin."""

    def _foreground_mask(self, depth_frame):
        """배경보다 가까운(=튀어나온) 픽셀의 이진 마스크(0/1, uint8)를 만든다.

        각 픽셀을 전체 화면 공통 기준(스칼라 거리)이 아니라, 캘리브레이션 시점에
        기록해둔 "그 픽셀 자신의 배경 depth"(`self.background_depth_map`)와
        비교한다. 트레이 바닥이 기울어져 있거나 펜스처럼 원래 튀어나온 고정
        구조물이 있어도, 그 자리의 원래 depth를 기준으로 삼으니 오탐이 없다
        (펜스 위에 사과가 새로 올라가면 그 자리만 정확히 "가까워짐"으로 잡힘).

        depth 센서 자체 노이즈로 생기는 가늘고 작은 잡음은 morphological
        opening(NOISE_OPEN_KERNEL_PX)으로 미리 지운다 (사과처럼 두꺼운 덩어리는
        거의 그대로 남는다).
        """
        if (
            self.background_depth_map is None
            or self.background_depth_map.shape != depth_frame.shape
        ):
            return np.zeros(depth_frame.shape, dtype=np.uint8)

        valid = (depth_frame > 0) & (self.background_depth_map > 0)
        closer_than_own_background = valid & (
            self.background_depth_map.astype(np.int32) - depth_frame.astype(np.int32)
            >= PRESENCE_MARGIN_MM
        )

        mask = closer_than_own_background.astype(np.uint8)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (NOISE_OPEN_KERNEL_PX, NOISE_OPEN_KERNEL_PX)
        )
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    def _find_all_unknown_blobs(self, depth_frame, exclude_boxes=None):
        """YOLO 박스와 무관하게, depth만으로 "배경보다 튀어나온 덩어리"를 전부 찾는다.

        YOLO가 박스를 아예 못 찾은 경우에도(예: 학습 안 된 물체) depth로는 뭔가
        튀어나와 있다는 걸 알 수 있으므로, 배경보다 확실히 가까운(=튀어나온)
        픽셀들을 모아 연결된 덩어리(connected component)로 묶는다. (단순히 전체
        픽셀의 평균 좌표를 쓰면, 서로 떨어진 잡음 덩어리 여러 개가 있을 때 그 사이
        허공이 중심으로 잡히는 문제가 생길 수 있어 피함.)

        배경은 이제 픽셀별 배경 지도(`_foreground_mask` 참고)와 비교하므로, 펜스
        같은 고정 구조물은 애초에 배경으로 흡수되어 여기까지 올라오지 않는다.
        그래서 모양(solidity)으로 다시 걸러낼 필요 없이, 충분히 크기만
        (MIN_PRESENCE_PIXELS) 하면 그대로 물체로 본다. (테이프처럼 가운데 구멍이
        뚫린 고리 모양의 실제 물체도 정상적으로 잡힌다 - 예전에 solidity로 펜스를
        걸러내던 로직이 이런 모양의 진짜 물체까지 같이 걸러내는 부작용이 있었음.)

        exclude_boxes를 넘기면(YOLO가 이미 찾은 박스들) 그 영역을 덩어리 탐색
        전에 마스크에서 미리 지운다. "가장 큰 덩어리를 찾은 뒤 YOLO 박스와
        겹치면 버리는" 순서 대신, 애초에 YOLO가 이미 아는 영역은 후보에서
        제외하고 나머지에서 찾으므로, 우연히 YOLO 박스 바로 옆에 진짜 새로운
        물체가 있어도 놓치지 않는다.

        반환값: [{'bbox': (x1,y1,x2,y2), 'centroid': (cx,cy), 'depth_mm': float, 'area': int}, ...]
        (없으면 빈 리스트). 뎁스 디버그 오버레이가 화면에 보이는 미확인 물체를
        전부 그리는 데 쓴다.
        """
        if depth_frame is None:
            return []

        mask = self._foreground_mask(depth_frame)
        if exclude_boxes:
            for x1, y1, x2, y2 in exclude_boxes:
                mask[y1:y2, x1:x2] = 0

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        if num_labels <= 1:
            # label 0은 배경 - 튀어나온 덩어리가 하나도 없음
            return []

        blobs = []
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < MIN_PRESENCE_PIXELS:
                continue

            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            cx, cy = centroids[label]
            cx, cy = int(round(cx)), int(round(cy))

            # centroid는 픽셀 좌표의 산술평균이라, 덩어리 모양이 오목하거나 안쪽에
            # depth 무효(0)인 구멍이 있으면(광택 있는 표면의 반사로 흔히 생김) 정작
            # centroid 위치 자체는 덩어리 바깥/구멍 안일 수 있다. 그 한 픽셀만
            # 읽으면 큰 덩어리를 찾아놓고도 depth=0이라 버려지는 문제가 생기므로,
            # 덩어리에 실제로 속한 모든 픽셀의 depth 중앙값을 쓴다.
            blob_depths = depth_frame[labels == label]
            blob_depths = blob_depths[blob_depths > 0]
            if blob_depths.size == 0:
                continue
            depth_mm = float(np.median(blob_depths))

            blobs.append({
                'bbox': (x, y, x + w, y + h),
                'centroid': (cx, cy),
                'depth_mm': depth_mm,
                'area': area,
            })

        return blobs

    def _find_unknown_blob(self, depth_frame, exclude_boxes=None):
        """`_find_all_unknown_blobs` 중 픽셀 수가 가장 많은 것 하나만 반환
        (또는 없으면 None). get_apple_status 서비스는 한 번에 물체 하나만
        판정하는 구조라 여기서는 항상 이걸 쓴다."""
        blobs = self._find_all_unknown_blobs(depth_frame, exclude_boxes)
        if not blobs:
            return None
        return max(blobs, key=lambda b: b['area'])

    def _find_unknown_object_position(self, depth_frame):
        """YOLO가 박스를 못 찾았을 때, depth 블롭의 위치를 카메라 좌표계 (x,y,z)로 변환.
        반환값: (x, y, z) 또는 블롭이 없으면 None."""
        blob = self._find_unknown_blob(depth_frame)
        if blob is None:
            return None
        cx, cy = blob['centroid']
        return self._pixel_to_camera_coords(cx, cy, blob['depth_mm'])

    def _box_ring_depths(self, depth_frame, box):
        """박스 안쪽 depth와, 박스 바로 바깥 테두리(ring) depth median을 반환.
        handle_get_status의 높이차 판정과 디버그 오버레이 표시에 공용으로 쓰인다.

        박스 안쪽은 median이 아니라 낮은 percentile(INNER_DEPTH_PERCENTILE)을 쓴다.
        사과는 둥글어서 박스 안에서 카메라에 가장 가까운 건 중앙 정점뿐이고, 박스
        가장자리/모서리는 배경에 가깝다 - 이 부분이 전체의 절반 가까이 차지하므로
        median을 쓰면 실제 사과 높이보다 훨씬 얕게(배경에 가깝게) 측정되어, 진짜
        사과인데도 높이차 판정에서 계속 FAIL 나는 원인이 된다. 낮은 percentile은
        "정점 부근" 픽셀들을 대표값으로 써서 이 과소평가를 줄인다.
        """
        if depth_frame is None:
            return None, None

        h, w = depth_frame.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None, None

        inner = depth_frame[y1:y2, x1:x2]
        inner_valid = inner[inner > 0]
        if inner_valid.size == 0:
            return None, None
        inner_depth = float(np.percentile(inner_valid, INNER_DEPTH_PERCENTILE))

        rx1 = max(0, x1 - HEIGHT_DIFF_RING_PX)
        ry1 = max(0, y1 - HEIGHT_DIFF_RING_PX)
        rx2 = min(w, x2 + HEIGHT_DIFF_RING_PX)
        ry2 = min(h, y2 + HEIGHT_DIFF_RING_PX)

        ring_region = depth_frame[ry1:ry2, rx1:rx2]
        ring_mask = np.ones(ring_region.shape, dtype=bool)
        ring_mask[(y1 - ry1):(y2 - ry1), (x1 - rx1):(x2 - rx1)] = False
        ring_valid = ring_region[ring_mask & (ring_region > 0)]
        if ring_valid.size == 0:
            return inner_depth, None
        ring_depth = float(np.median(ring_valid))
        return inner_depth, ring_depth

    def _real_world_diameter_mm(self, depth_frame, box):
        """박스 안쪽 depth(median)와 카메라 intrinsics를 이용해 박스의 픽셀 폭/높이를
        실제 물리적 크기(mm)로 환산한다. 둘의 평균을 사과 지름 추정치로 사용."""
        if depth_frame is None:
            return None

        h, w = depth_frame.shape[:2]
        x1, y1, x2, y2 = box
        ix1, iy1 = max(0, int(x1)), max(0, int(y1))
        ix2, iy2 = min(w, int(x2)), min(h, int(y2))
        if ix2 <= ix1 or iy2 <= iy1:
            return None

        region = depth_frame[iy1:iy2, ix1:ix2]
        valid = region[region > 0]
        if valid.size == 0:
            return None
        depth_mm = float(np.median(valid))

        fx = self.intrinsics['fx']
        fy = self.intrinsics['fy']
        width_mm = (x2 - x1) * depth_mm / fx
        height_mm = (y2 - y1) * depth_mm / fy
        return (width_mm + height_mm) / 2.0

    def _compute_position(self, cx, cy):
        """픽셀 좌표 -> 카메라 좌표계 3D 위치. depth가 없으면 (0,0,0)."""
        depth_frame = self._wait_for_valid_data(self.img_node.get_depth_frame, "depth frame")
        try:
            cz = depth_frame[cy, cx]
        except IndexError:
            self.get_logger().warn(f"Coordinates ({cx},{cy}) out of range.")
            return 0.0, 0.0, 0.0
        if cz == 0:
            return 0.0, 0.0, 0.0
        return self._pixel_to_camera_coords(cx, cy, cz)

    def _calibrate_scene(self):
        """depth 프레임 여러 장을 모아서 픽셀별 median으로 "배경 depth 지도"를 만든다.

        이후 모든 물체 감지는 각 픽셀의 현재 depth를 이 지도의 같은 픽셀 값과
        비교하는 방식(배경 차분)으로 이루어진다. 트레이 바닥이 기울어져 있든
        펜스 같은 고정 구조물이 있든, 그 픽셀의 "원래 있던 depth"를 기준으로
        삼으므로 화면 전체에 스칼라 기준 하나를 적용할 때 생기던 오탐(펜스가
        화면의 상당 부분을 "물체"로 오인하는 등)이 구조적으로 없어진다.

        프레임 한 장이 아니라 BACKGROUND_CALIBRATION_DURATION_SEC 동안 여러 장을
        모아 픽셀별 median을 쓰는 이유: depth 센서는 프레임 하나에서 반사/노이즈로
        일부 픽셀이 우연히 무효값(0)이 되는 경우가 흔한데, 그 순간의 프레임 한 장만
        배경으로 박아두면 그 픽셀은 "배경을 모르는 곳"으로 영구히 남아 이후 그
        자리에 물체가 와도 감지가 안 된다. 여러 프레임의 median을 쓰면 개별 프레임의
        일시적 노이즈에 훨씬 강해진다.

        `background_distance_mm`(90th percentile)은 이제 판정에는 안 쓰고,
        디버그 창의 depth 컬러맵 스케일링(시각화)에만 참고용으로 쓴다.

        주의: 이 캘리브레이션 동안 트레이 위에 사과가 계속 같은 자리에 있으면 그
        자리의 depth가 "배경"으로 고정돼버려 이후 그 위치의 물체를 못 볼 수 있다.
        가능하면 트레이를 비운 상태에서 노드를 시작하는 게 좋다.

        depth를 못 구하면 (BACKGROUND_DISTANCE_MM 기본값, 배경 지도 없음)으로 폴백한다.
        """
        first_frame = self._wait_for_valid_data(
            self.img_node.get_depth_frame, "depth frame(장면 캘리브레이션용)"
        )
        frames = [first_frame]
        end_time = time.time() + BACKGROUND_CALIBRATION_DURATION_SEC
        while time.time() < end_time:
            time.sleep(BACKGROUND_CALIBRATION_SAMPLE_INTERVAL_SEC)
            frame = self.img_node.get_depth_frame()
            if frame is not None:
                frames.append(frame)

        stack = np.stack(frames, axis=0).astype(np.float32)
        stack[stack == 0] = np.nan  # 무효(0) 픽셀은 median 계산에서 제외
        with np.errstate(all='ignore'):
            median_map = np.nanmedian(stack, axis=0)
        # 모든 프레임에서 계속 무효였던 픽셀은 nan -> 0으로 (배경 모름 = foreground 판정 제외)
        background_depth_map = np.nan_to_num(median_map, nan=0.0).astype(first_frame.dtype)

        valid = background_depth_map[background_depth_map > 0]
        if valid.size == 0:
            self.get_logger().warn(
                f"장면 자동 캘리브레이션 실패({len(frames)}프레임 모두 무효) - "
                f"배경 거리 기본값({BACKGROUND_DISTANCE_MM}mm), 배경 지도 없이 시작"
            )
            return float(BACKGROUND_DISTANCE_MM), None

        background_distance_mm = float(np.percentile(valid, 90))
        self.get_logger().info(
            f"배경 거리 자동 캘리브레이션(시각화용): {background_distance_mm:.0f}mm, "
            f"{len(frames)}프레임 median으로 픽셀별 배경 지도 "
            f"{background_depth_map.shape[1]}x{background_depth_map.shape[0]} 저장 완료"
        )
        return background_distance_mm, background_depth_map

    def _pixel_to_camera_coords(self, x, y, z):
        # 표준 핀홀 카메라 역투영 공식: 픽셀(x,y)+depth(z) -> 카메라 좌표계 3D 점(mm).
        # intrinsics(fx,fy=초점거리, ppx,ppy=주점)는 camera_info 토픽에서 옴.
        fx = self.intrinsics['fx']
        fy = self.intrinsics['fy']
        ppx = self.intrinsics['ppx']
        ppy = self.intrinsics['ppy']
        return (
            (x - ppx) * z / fx,
            (y - ppy) * z / fy,
            z,
        )
