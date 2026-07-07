"""
detection.py
=============
카메라(RealSense D455)로 본 사과 1개의 상태를 판정해서 알려주는 ROS2 노드
(object_detection_node). backend/vision_bridge.py가 이 노드의 get_apple_status
서비스를 주기적으로 폴링해서 결과를 가져간다.

판정 파이프라인 (handle_get_status 기준):
    1) YOLO(AppleStatusModel)가 컬러 프레임에서 사과 후보 박스를 찾는다
       (apple_normal / apple_rotten / apple_damaged 중 하나 + confidence).
    2) 그 박스가 진짜 물체인지 depth로 재검증한다 - 박스 안쪽이 바로 바깥
       배경보다 확실히 튀어나와 있어야 통과 (사진/그림자 등 오탐 방지).
       -> _has_height_difference / _box_ring_depths
    3) confidence가 너무 낮으면 "unknown", 박스 자체가 없으면 depth로 물체
       유무만 봐서 "unknown"(뭔가 있음) 또는 "empty"(트레이 빔)로 답한다.
    4) apple_normal/apple_damaged로 판정된 것들은 depth로 잰 실제 지름(mm)을
       최근 정상 사과들의 평균과 비교해서, 확연히 작으면 손상이 아니라
       "원래 작은 사과"로 보고 apple_small_normal/apple_small_damaged로
       재분류한다. -> _classify_size / _real_world_diameter_mm

디버깅용으로 별도 스레드 2개가 항상 돈다:
    - img_node용 SingleThreadedExecutor 스레드: 카메라 토픽 구독을 계속 spin.
    - _depth_debug_thread: depth 컬러맵 + 위 판정 근거를 그린 로컬 cv2 창.
둘 다 서비스 콜백(느림, ~1초)과 별개 스레드/콜백그룹에서 돌아서 서로 막지 않는다.
"""

import threading
import time
from collections import deque

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from apple_care_msgs.srv import SrvAppleStatus
from obj_detection.realsense import ImgNode
from obj_detection.yolo import AppleStatusModel, MIN_KNOWN_CONFIDENCE

# 트레이/배경까지의 대략적인 거리(mm). 실제 설치 환경에 맞춰 캘리브레이션 필요.
BACKGROUND_DISTANCE_MM = 500
PRESENCE_MARGIN_MM = 50
MIN_PRESENCE_PIXELS = 500

# YOLO 박스 안쪽 깊이와 바로 바깥 주변 깊이의 차이가 이 값(mm) 이상일 때만
# "물체가 배경보다 튀어나와 있다"고 보고 실제 감지로 표시한다.
# (아래 INNER_DEPTH_PERCENTILE로 박스 안쪽의 "정점 부근" 값을 쓰게 되어 이전보다
# 실제 사과 높이에 가깝게 측정되므로, median 기준으로 튜닝했던 값(100mm)은 너무
# 커서 실제 사과도 거의 다 FAIL 처리됨 - 사과 지름(약 60~90mm) 이하로 낮춰야 함)
HEIGHT_DIFF_MARGIN_MM = 30
# 박스 바깥 주변 깊이를 샘플링할 테두리 두께(px)
HEIGHT_DIFF_RING_PX = 20
# 박스 안쪽 depth 중 몇 번째 percentile을 "정점(사과 꼭대기)"으로 볼지.
# 낮을수록 카메라에 더 가까운(=값이 작은) 픽셀 쪽으로 치우쳐서 잡음.
INNER_DEPTH_PERCENTILE = 10

# 디버그용 인식 결과 시각화 이미지 퍼블리시 주기(초)
DEBUG_IMAGE_PERIOD_SEC = 0.05

# 뎁스 디버깅 창 이름
DEPTH_DEBUG_WINDOW_NAME = "Depth Debug"

# 디버그 오버레이 글자 크기/굵기 (이전 단일-박스 오버레이 스타일로 복원 - 다중 박스로
# 바꾸면서 한 번 줄였던 걸 다시 키움)
OVERLAY_FONT_SCALE = 0.6
OVERLAY_FONT_THICKNESS = 2

# "apple_normal"로 판정된 사과들의 실제 지름(mm, depth로 환산) 이동 평균을 이용해,
# 그중 상대적으로 뚜렷하게 작은 사과를 "apple_small"로 재분류한다.
NORMAL_SIZE_HISTORY_LEN = 75
MIN_NORMAL_SIZE_SAMPLES = 1
SMALL_APPLE_SIZE_RATIO = 0.8

# YOLO가 작은 사과를 "손상(apple_damaged)"으로 잘못 인식하는 경우가 있어,
# 이 라벨들에 한해서는 depth로 잰 실제 크기가 충분히 작으면 손상이 아니라
# "작아서 그런 것"으로 보고 재분류한다.
# (apple_rotten은 색/질감 변화가 핵심 근거라 크기 오탐과 무관하므로 대상에서 제외)
# 원래 YOLO가 뭐라고 했었는지(정상이었는지/손상으로 봤었는지) 구분이 되도록,
# 재분류 결과도 하나의 "apple_small"이 아니라 원본 라벨별로 다르게 이름 붙인다.
SIZE_OVERRIDE_LABELS = {"apple_normal", "apple_damaged"}
SMALL_LABEL_MAP = {
    "apple_normal": "apple_small_normal",
    "apple_damaged": "apple_small_damaged",
}


class ObjectDetectionNode(Node):
    """get_apple_status 서비스 + 디버그 이미지 퍼블리셔/창을 제공하는 메인 노드.

    카메라 구독(ImgNode)은 별도 노드/스레드로 분리되어 있고, 이 노드는 그로부터
    최신 프레임만 받아서 YOLO+depth 판정, 디버그 시각화, 서비스 응답을 담당한다.
    """

    def __init__(self):
        super().__init__('object_detection_node')
        self.img_node = ImgNode()
        # img_node는 별도 스레드에서 계속 spin -> 서비스 콜백이 오래 걸려도(프레임 수집 1초)
        # 디버그 이미지 타이머나 다른 콜백이 카메라 데이터 갱신에 막히지 않게 함
        self.img_executor = SingleThreadedExecutor()
        self.img_executor.add_node(self.img_node)
        self._img_spin_thread = threading.Thread(target=self.img_executor.spin, daemon=True)
        self._img_spin_thread.start()

        self.model = AppleStatusModel()
        self.intrinsics = self._wait_for_valid_data(
            self.img_node.get_camera_intrinsic, "camera intrinsics"
        )
        self._normal_apple_sizes = deque(maxlen=NORMAL_SIZE_HISTORY_LEN)
        # _normal_apple_sizes는 서비스 콜백 스레드(handle_get_status, 1초에 1개)와
        # 뎁스 디버그 스레드(_draw_all_detections, 20Hz 다중 박스)가 동시에 쓰기/읽기
        # 하므로 락으로 보호한다.
        self._normal_sizes_lock = threading.Lock()
        # handle_get_status가 최근에 어떤 근거로 판정했는지(박스/뎁스 값/최종 라벨)를
        # 뎁스 디버그 창에 그려주기 위한 공유 상태. 서비스 콜백 스레드와 디버그 창
        # 스레드가 서로 다르므로 락으로 보호한다.
        self._last_debug_info = None
        self._debug_info_lock = threading.Lock()

        # 서비스(느림, ~1초)와 디버그 이미지 타이머(0.1초 주기)가 서로를 막지 않도록
        # 콜백 그룹을 분리하고 MultiThreadedExecutor로 동시에 돌린다 (main() 참고)
        self.service_cb_group = MutuallyExclusiveCallbackGroup()
        self.timer_cb_group = MutuallyExclusiveCallbackGroup()

        self.create_service(
            SrvAppleStatus,
            'get_apple_status',
            self.handle_get_status,
            callback_group=self.service_cb_group,
        )

        self.bridge = CvBridge()
        self.debug_image_pub = self.create_publisher(Image, 'obj_detection/debug_image', 10)
        self.create_timer(
            DEBUG_IMAGE_PERIOD_SEC, self._publish_debug_image, callback_group=self.timer_cb_group
        )

        # cv2.imshow/waitKey는 GUI 백엔드(Qt/GTK) 특성상 "창을 만든 스레드"에서
        # 계속 호출해야 정상적으로 갱신된다. MultiThreadedExecutor의 콜백은 매번
        # 다른 워커 스레드에서 실행될 수 있어 타이머 콜백에 두면 창이 안 뜨거나
        # 멈춰 보일 수 있으므로, 전용 스레드 하나를 만들어 거기서만 그린다.
        self._stop_depth_debug = threading.Event()
        self._depth_debug_thread = threading.Thread(
            target=self._depth_debug_loop, daemon=True
        )
        self._depth_debug_thread.start()

        self.get_logger().info("ObjectDetectionNode initialized.")

    def _publish_debug_image(self):
        """최신 컬러 프레임에 depth 기반 판정 근거(높이차 통과 여부/실제 지름/
        크기 재분류)를 그려서 디버그 토픽으로 퍼블리시.

        예전엔 model.annotate_frame()으로 YOLO 원본 박스/라벨만 그려서 내보냈는데,
        그러면 프론트(hmi_app)의 카메라 화면에는 depth 판정이나 apple_small_* 재분류가
        전혀 안 보였다. 뎁스 디버그 창에서 쓰던 것과 동일한 `_draw_all_detections`를
        컬러 프레임 위에 그대로 적용해서, 로컬 뎁스 창과 프론트 화면이 같은 정보를
        보여주게 한다 (컬러/뎁스 프레임은 aligned_depth_to_color라 픽셀 좌표가 일치).
        """
        frame = self.img_node.get_color_frame()
        if frame is None:
            return
        depth_frame = self.img_node.get_depth_frame()
        annotated = frame.copy()
        detections = self.model.get_all_detections(frame)
        self._draw_all_detections(annotated, depth_frame, detections)
        msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        self.debug_image_pub.publish(msg)

    def _depth_debug_loop(self):
        """뎁스 프레임에 컬러맵을 입히고, 화면에 보이는 모든 사과에 대해 실시간으로
        박스/뎁스값/크기 판정을 오버레이로 그려서 로컬 창으로 띄운다.

        handle_get_status(서비스)는 1초에 한 번, "가장 confidence 높은 박스 하나"만
        판정하므로 트레이에 사과가 여러 개면 나머지는 창에 안 보였다. 여기서는
        디버그용으로 매 주기 YOLO를 다시 돌려 보이는 모든 박스를 그린다.

        executor 콜백 스레드와 분리된 전용 스레드에서 계속 돌며 이 창의
        생성/갱신을 전담한다 (GUI 백엔드가 스레드 하나에 고정되어야 하므로).
        """
        cv2.namedWindow(DEPTH_DEBUG_WINDOW_NAME, cv2.WINDOW_NORMAL)
        while not self._stop_depth_debug.is_set():
            depth_frame = self.img_node.get_depth_frame()
            color_frame = self.img_node.get_color_frame()
            if depth_frame is not None:
                # mm 단위 depth를 8bit로 스케일링 후 컬러맵 적용 (가까울수록 밝게 보이도록)
                depth_8bit = cv2.convertScaleAbs(
                    depth_frame, alpha=255.0 / BACKGROUND_DISTANCE_MM
                )
                depth_colormap = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)

                detections = []
                if color_frame is not None:
                    detections = self.model.get_all_detections(color_frame)
                self._draw_all_detections(depth_colormap, depth_frame, detections)

                with self._debug_info_lock:
                    debug_info = self._last_debug_info
                self._draw_last_service_summary(depth_colormap, debug_info)

                cv2.imshow(DEPTH_DEBUG_WINDOW_NAME, depth_colormap)
            # waitKey가 GUI 이벤트 루프를 돌려줘야 창이 실제로 그려진다.
            cv2.waitKey(1)
            time.sleep(DEBUG_IMAGE_PERIOD_SEC)
        cv2.destroyWindow(DEPTH_DEBUG_WINDOW_NAME)

    def _draw_all_detections(self, colormap, depth_frame, detections):
        """현재 프레임에서 YOLO가 찾은 모든 박스에 대해 depth 기반 판정 근거를
        각 박스 위에 그린다 (높이차 통과 여부, 실제 지름, 미리보기 최종 라벨)."""
        h, w = colormap.shape[:2]
        avg_diameter = self._avg_normal_diameter()

        for label, score, box in detections:
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            inner_depth, ring_depth = self._box_ring_depths(depth_frame, box)
            depth_ok = (
                inner_depth is not None and ring_depth is not None
                and (ring_depth - inner_depth) >= HEIGHT_DIFF_MARGIN_MM
            )
            box_color = (0, 255, 0) if depth_ok else (0, 0, 255)  # OK=초록, FAIL=빨강
            cv2.rectangle(colormap, (x1, y1), (x2, y2), box_color, 2)

            if not depth_ok:
                caption = f"{label} {score:.2f} depth-FAIL"
            else:
                diameter_mm = self._real_world_diameter_mm(depth_frame, box)
                preview_label = self._preview_size_label(label, diameter_mm, avg_diameter)
                diameter_txt = f"{diameter_mm:.0f}mm" if diameter_mm is not None else "-"
                if preview_label != label:
                    # "원본라벨->재분류라벨" 화살표 표기 대신, 재분류된 결과(apple_small_*)만 표시
                    caption = f"{preview_label} {score:.2f} {diameter_txt}"
                else:
                    caption = f"{label} {score:.2f} {diameter_txt}"

                # 실제 서비스는 1초에 "최고 confidence 박스 1개"만 봐서 평균 표본이
                # 잘 안 쌓일 수 있으므로, 여기(매 프레임 다중 박스 스캔)서도 정상
                # 사과로 확인된 것들을 같이 평균에 반영한다. 그래야 작은 사과가 한
                # 번도 "최고 박스"로 안 뽑혀도 정상 사과들 평균은 계속 채워진다.
                if (
                    label == "apple_normal" and preview_label == "apple_normal"
                    and score >= MIN_KNOWN_CONFIDENCE and diameter_mm is not None
                ):
                    self._add_normal_sample(diameter_mm)
                    avg_diameter = self._avg_normal_diameter()

            text_y = y1 - 8 if y1 - 8 > 10 else y2 + 18
            cv2.putText(
                colormap, caption, (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE, (0, 0, 0), OVERLAY_FONT_THICKNESS + 2,
            )
            cv2.putText(
                colormap, caption, (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE, (255, 255, 255), OVERLAY_FONT_THICKNESS,
            )

        if not detections:
            cv2.putText(
                colormap, "no YOLO box in current frame", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE, (255, 255, 255), OVERLAY_FONT_THICKNESS,
            )

    def _draw_last_service_summary(self, colormap, debug_info):
        """실제로 backend에 보고되는 값(1초 주기 서비스 호출의 최종 판정)을
        화면 상단에 한 줄 요약으로 보여준다. 위의 실시간 박스 미리보기는 참고용이고,
        실제 로봇 동작에 쓰이는 건 이 값이라는 걸 구분하기 위함."""
        if debug_info is None:
            text = "last service call: (none yet)"
        else:
            text = (
                f"last service call: {debug_info['raw_label']} -> "
                f"{debug_info['final_status']}"
            )
        h = colormap.shape[0]
        y = h - 12
        cv2.putText(
            colormap, text, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE, (0, 0, 0), OVERLAY_FONT_THICKNESS + 2,
        )
        cv2.putText(
            colormap, text, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE, (255, 255, 255), OVERLAY_FONT_THICKNESS,
        )

    def handle_get_status(self, request, response):
        """카메라에 보이는 사과 상태를 판정해서 반환한다."""
        label, score, box = self.model.get_best_detection(self.img_node)
        depth_frame = self.img_node.get_depth_frame()

        # 뎁스 디버그 창에 "왜 이렇게 판단했는지" 그대로 그려주기 위한 진단 정보.
        # box는 높이차 검증에서 탈락하면 아래서 None으로 지워지므로 원본을 따로 보관.
        debug_info = {
            'raw_box': box, 'raw_label': label, 'score': score,
            'inner_depth': None, 'ring_depth': None, 'height_diff': None,
            'depth_ok': None, 'diameter_mm': None, 'avg_diameter_mm': None,
            'final_status': None,
        }

        if box is not None:
            inner_depth, ring_depth = self._box_ring_depths(depth_frame, box)
            debug_info['inner_depth'] = inner_depth
            debug_info['ring_depth'] = ring_depth
            if inner_depth is not None and ring_depth is not None:
                height_diff = ring_depth - inner_depth
                debug_info['height_diff'] = height_diff
                debug_info['depth_ok'] = height_diff >= HEIGHT_DIFF_MARGIN_MM
            else:
                debug_info['depth_ok'] = False

            if not debug_info['depth_ok']:
                # YOLO는 박스를 찾았지만, 주변 배경과 높이(깊이) 차이가 없음 -> 실제 물체로 보지 않음
                self.get_logger().info(
                    f"YOLO detected '{label}' but no depth height difference -> ignoring box"
                )
                box = None

        if box is None:
            # 박스가 없거나(원래 없었거나 높이 차이 검증에서 탈락) -> depth로 "물체 자체는 있는지" 확인
            if self._object_present_by_depth(depth_frame):
                self.get_logger().info("No valid box, but depth shows an object -> unknown")
                response.status = "unknown"
                response.confidence = 0.0
                response.position = []
            else:
                response.status = "empty"
                response.confidence = 0.0
                response.position = []
            debug_info['final_status'] = response.status
            self._set_last_debug_info(debug_info)
            return response

        # 박스 중심 픽셀 좌표 -> 그 지점의 depth로 3D 위치(카메라 좌표계) 계산
        cx, cy = map(int, [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        position = self._compute_position(cx, cy)

        if score < MIN_KNOWN_CONFIDENCE:
            self.get_logger().info(f"Low confidence detection ({score:.2f}) -> unknown")
            response.status = "unknown"
        else:
            response.status = self._classify_size(label, depth_frame, box, debug_info)

        response.confidence = float(score)
        response.position = [float(x) for x in position]
        debug_info['final_status'] = response.status
        self._set_last_debug_info(debug_info)
        return response

    def _set_last_debug_info(self, debug_info):
        with self._debug_info_lock:
            self._last_debug_info = debug_info

    def _object_present_by_depth(self, depth_frame):
        """배경보다 확실히 가까운 픽셀이 충분히 있으면 물체가 있다고 판단."""
        if depth_frame is None:
            return False
        closer_than_background = (depth_frame > 0) & (
            depth_frame < (BACKGROUND_DISTANCE_MM - PRESENCE_MARGIN_MM)
        )
        return int(closer_than_background.sum()) >= MIN_PRESENCE_PIXELS

    def _has_height_difference(self, depth_frame, box):
        """박스 안쪽 깊이(높이)와 박스 바로 바깥 주변 깊이를 비교해서,
        실제로 배경보다 튀어나온 물체인지(사진/그림자 등 오탐이 아닌지) 확인한다."""
        inner_depth, ring_depth = self._box_ring_depths(depth_frame, box)
        if inner_depth is None or ring_depth is None:
            return False
        # 물체가 카메라에 더 가까울수록(더 높이 튀어나올수록) depth 값은 더 작다
        height_diff = ring_depth - inner_depth
        return height_diff >= HEIGHT_DIFF_MARGIN_MM

    def _box_ring_depths(self, depth_frame, box):
        """박스 안쪽 depth와, 박스 바로 바깥 테두리(ring) depth median을 반환.
        `_has_height_difference` 판정과 디버그 오버레이 표시에 공용으로 쓰인다.

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

        avg_diameter = self._avg_normal_diameter()
        if debug_info is not None:
            debug_info['avg_diameter_mm'] = avg_diameter

        final_label = self._preview_size_label(label, diameter_mm, avg_diameter)
        if final_label != label and label == "apple_damaged":
            self.get_logger().info(
                f"YOLO judged 'apple_damaged' but size({diameter_mm:.1f}mm) is just "
                f"small vs avg({avg_diameter:.1f}mm) -> overriding to {final_label}"
            )

        # apple_small로 재분류되지 않은, 즉 평균 크기 산정에 쓸만한 정상 크기만 누적.
        # apple_damaged로 판정된 것은(재분류 여부와 무관하게) 기준 크기 계산에서 제외한다.
        if final_label == "apple_normal":
            self._add_normal_sample(diameter_mm)
        return final_label

    def _add_normal_sample(self, diameter_mm):
        with self._normal_sizes_lock:
            self._normal_apple_sizes.append(diameter_mm)

    def _avg_normal_diameter(self):
        """지금까지 쌓인 정상 사과 지름(mm) 표본의 평균. 표본이 부족하면 None."""
        with self._normal_sizes_lock:
            samples = list(self._normal_apple_sizes)
        if len(samples) >= MIN_NORMAL_SIZE_SAMPLES:
            return sum(samples) / len(samples)
        return None

    def _preview_size_label(self, label, diameter_mm, avg_diameter):
        """avg_diameter 대비 diameter_mm이 충분히 작으면 원본 라벨에 맞는
        apple_small_normal/apple_small_damaged로 바꿔서 반환.

        `_classify_size`의 실제 판정과 depth 디버그 창의 실시간 미리보기(여러 박스를
        한꺼번에 그릴 때)가 동일한 기준을 쓰도록 순수 계산만 분리한 함수. 상태(이동
        평균 deque)를 건드리지 않으므로 미리보기 용도로 반복 호출해도 안전하다.
        """
        if label not in SIZE_OVERRIDE_LABELS or diameter_mm is None or avg_diameter is None:
            return label
        if diameter_mm < avg_diameter * SMALL_APPLE_SIZE_RATIO:
            return SMALL_LABEL_MAP[label]
        return label

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

    def _wait_for_valid_data(self, getter, description):
        # img_executor가 별도 스레드에서 계속 spin 중이므로 여기서는 그냥 대기만 한다.
        data = getter()
        while data is None or (isinstance(data, np.ndarray) and not data.any()):
            time.sleep(0.1)
            self.get_logger().info(f"Retry getting {description}.")
            data = getter()
        return data

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


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    # 서비스(get_apple_status, ~1초 소요)와 디버그 이미지 타이머(0.1초 주기)를
    # 서로 다른 콜백 그룹으로 등록해뒀으므로, MultiThreadedExecutor로 돌려야
    # 실제로 동시에(다른 스레드에서) 실행되어 서로 막지 않는다.
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.img_executor.shutdown()
        # 뎁스 디버그 창 전용 스레드도 정리해야 함 - 안 그러면 프로세스 종료 후에도
        # cv2 창이 남거나 데몬 스레드가 좀비로 남을 수 있음
        node._stop_depth_debug.set()
        node._depth_debug_thread.join(timeout=1.0)
        node.img_node.destroy_node()
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()