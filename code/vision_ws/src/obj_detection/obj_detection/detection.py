"""
detection.py
=============
카메라(RealSense D455)로 본 사과 1개의 상태를 판정해서 알려주는 ROS2 노드
(object_detection_node). backend/vision_bridge.py가 이 노드의 get_apple_status
서비스를 주기적으로 폴링해서 결과를 가져간다.

판정 파이프라인 (handle_get_status 기준):
    1) YOLO(AppleStatusModel)가 컬러 프레임에서 사과 후보 박스를 찾는다
       (apple_normal / apple_rotten / apple_damaged / apple_small 중 하나 + confidence).
    2) 그 박스가 진짜 물체인지 depth로 재검증한다 - 박스 안쪽이 바로 바깥
       배경보다 확실히 튀어나와 있어야 통과 (사진/그림자 등 오탐 방지).
       -> _box_ring_depths
    3) confidence가 너무 낮으면 "unknown", 박스 자체가 없으면 depth로 물체
       유무만 봐서 "unknown"(뭔가 있음) 또는 "empty"(트레이 빔)로 답한다.
    4) apple_normal/apple_damaged로 판정된 것들은 depth로 잰 실제 지름(mm)을
       최근 정상 사과들의 평균과 비교해서, 확연히 작으면 "apple_small"로
       재분류한다. -> _classify_size / _real_world_diameter_mm
       (YOLO가 apple_small을 직접 학습했지만, 이건 "같은 트레이의 다른 사과들에
       비해 상대적으로 작은지"를 depth 실측으로 다시 확인하는 이중 안전망)

이 파일에는 ROS2 노드 본체(초기화, get_apple_status 서비스 콜백, 종료)만 있고,
실제 로직은 mixin 3개로 나뉘어 있다 (파일이 너무 길어져서 분리함):
    - depth_utils.DepthAnalysisMixin   : 배경 캘리브레이션/배경 차분/미확인 물체 탐지/좌표 변환
    - size_classifier.SizeClassifierMixin : depth 실측 지름으로 apple_small 재분류
    - debug_overlay.DebugOverlayMixin  : 뎁스 디버그 창 + 프론트용 디버그 이미지 오버레이

디버깅용으로 별도 스레드 2개가 항상 돈다:
    - img_node용 SingleThreadedExecutor 스레드: 카메라 토픽 구독을 계속 spin.
    - _depth_debug_thread(DebugOverlayMixin): depth 컬러맵 + 판정 근거를 그린 로컬 cv2 창.
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
from obj_detection.depth_utils import DepthAnalysisMixin, HEIGHT_DIFF_MARGIN_MM
from obj_detection.size_classifier import SizeClassifierMixin, NORMAL_SIZE_HISTORY_LEN
from obj_detection.debug_overlay import DebugOverlayMixin, DEBUG_IMAGE_PERIOD_SEC


class ObjectDetectionNode(DepthAnalysisMixin, SizeClassifierMixin, DebugOverlayMixin, Node):
    """get_apple_status 서비스 + 디버그 이미지 퍼블리셔/창을 제공하는 메인 노드.

    카메라 구독(ImgNode)은 별도 노드/스레드로 분리되어 있고, 이 노드는 그로부터
    최신 프레임만 받아서 YOLO+depth 판정, 디버그 시각화, 서비스 응답을 담당한다.
    실제 로직 대부분은 위 3개 mixin에 있고, 여기서는 초기화와 서비스 콜백만 담당한다.
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
        # 배경까지의 거리를 스칼라 값 하나로 캘리브레이션하면 트레이 바닥이
        # 기울어져 있거나 펜스 같은 고정 구조물이 있을 때 계속 오탐이 났다.
        # 시작 시점의 depth 프레임 전체를 "픽셀별 배경"으로 저장해서, 이후
        # 각 픽셀을 자기 자신의 캘리브레이션 값과 비교한다 (자세한 이유는
        # depth_utils.py의 BACKGROUND_DISTANCE_MM 주석/_calibrate_scene 참고).
        self.background_distance_mm, self.background_depth_map = self._calibrate_scene()
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
            # 박스가 없거나(원래 없었거나 높이 차이 검증에서 탈락) -> YOLO는 놓쳤어도
            # depth만으로 "배경보다 튀어나온 덩어리"를 찾아 위치까지 추정해본다.
            unknown_position = self._find_unknown_object_position(depth_frame)
            if unknown_position is not None:
                self.get_logger().info("No valid box, but depth shows an object -> unknown")
                response.status = "unknown"
                response.confidence = 0.0
                response.position = [float(v) for v in unknown_position]
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

    def _wait_for_valid_data(self, getter, description):
        # img_executor가 별도 스레드에서 계속 spin 중이므로 여기서는 그냥 대기만 한다.
        data = getter()
        while data is None or (isinstance(data, np.ndarray) and not data.any()):
            time.sleep(0.1)
            self.get_logger().info(f"Retry getting {description}.")
            data = getter()
        return data


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    # 서비스(get_apple_status, ~1초 소요)와 디버그 이미지 타이머(0.05초 주기)를
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
