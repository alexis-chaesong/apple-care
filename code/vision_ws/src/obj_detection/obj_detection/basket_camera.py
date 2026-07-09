"""
basket_camera.py
=================
바스켓 4개(B1~B4)를 내려다보도록 고정 설치된 일반 USB 웹캠(예: 로지텍 C270,
/dev/videoN)으로 각 바스켓에 사과가 있는지 없는지를 판단해서 `/basket_status`
토픽으로 계속 알려주는 노드 (basket_detection_node).

픽 카메라(realsense.py/detection.py)는 depth까지 쓰는 RealSense라 ROS 토픽으로
구독하지만, 이 카메라는 "바스켓 안에 뭐라도 있는지"만 컬러 프레임으로 빠르게
판단하면 되므로 별도의 카메라 드라이버 노드 없이 cv2.VideoCapture로 이 노드
안에서 직접 연다 (RealSense가 아닌 일반 V4L2 웹캠이라 realsense2_camera로
publish할 수도 없음).

파이프라인 (_publish_status 기준):
    1) YOLO(yolo.AppleStatusModel, obj_detection과 동일 체크포인트 재사용 -
       apple_normal/apple_rotten/apple_damaged/basket 클래스가 이미 다 있음)로
       현재 프레임의 모든 후보 박스를 가져온다.
    2) label == "basket"인 박스만 추려서 좌상단 좌표(y1, x1) 오름차순으로
       정렬한다 - 실제 카메라 구도상 바스켓 4개가 위에서 아래로 배치되므로
       화면 맨 위가 b1, 맨 아래가 b4가 되도록 y(세로) 우선 정렬, 같은 높이면
       x(가로)로 tie-break. 4개보다 적게 잡히면 잡힌 만큼만 채우고
       valid=False로 표시한다.
    3) apple_normal/apple_rotten/apple_damaged 박스의 중심점이 어떤 바스켓
       bbox 안에 들어오면 그 바스켓을 occupied=True로 표시한다.
    4) {"b1":bool, "b2":bool, "b3":bool, "b4":bool, "valid":bool,
       "timestamp":float} JSON을 std_msgs/String으로 `/basket_status`에
       publish한다 (기존 /decision/result, /robot/command와 동일하게
       String+JSON 패턴을 그대로 따름 - 커스텀 topic msg를 새로 만들지 않음).

카메라는 로봇 팔에 달린 게 아니라 바스켓 전체가 잘 보이는 곳에 고정
설치되는 별도의 물리 카메라이므로, prepare_camera.py처럼 로봇을 움직여
포지셔닝할 필요가 없다. 어떤 /dev/videoN을 열지는 ROS 파라미터
`device_index`로 지정한다 (`v4l2-ctl --list-devices`로 실제 장치/인덱스
확인 - basket_camera_live_test.py의 --device_index와 동일한 값을 쓰면 됨).
"""

import json
import threading
import time

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from obj_detection.ros_image_utils import cv2_to_imgmsg
from obj_detection.yolo import AppleStatusModel

# 바스켓 점유 판정을 몇 초마다 다시 계산해서 publish할지.
STATUS_PUBLISH_PERIOD_SEC = 0.5

# 관리하는 바스켓 개수(B1~B4 고정).
BASKET_NAMES = ["b1", "b2", "b3", "b4"]

# 디버그 오버레이에 b1~b4와 함께 보여줄 표시 이름 (사용자 지정값).
# /basket_status로 나가는 JSON 키(b1..b4)는 로봇/백엔드가 그대로 쓰고 있으므로
# 여기서는 건드리지 않고, 디버그 오버레이 표시에만 쓴다.
BASKET_DISPLAY_NAMES = {
    "b1": "basket_a_s",
    "b2": "basket_a_d",
    "b3": "basket_a_n",
    "b4": "basket_a_r",
}

APPLE_LABELS = {"apple_normal", "apple_rotten", "apple_damaged"}


class WebcamCapture:
    """일반 V4L2 웹캠을 백그라운드 스레드에서 계속 읽어 최신 프레임만 들고 있는 헬퍼.

    realsense.py의 ImgNode(콜백 기반 구독)와 달리 cv2.VideoCapture는 자체적으로
    스트림을 안 밀어주고 read()를 호출해야 하므로, 이 스레드가 계속 read()를
    돌며 최신 프레임을 갱신해둔다 - 그래야 _publish_status 타이머가 항상 최신
    프레임을 즉시 가져다 쓸 수 있다(느린 read()가 타이머 콜백을 막지 않음)."""

    def __init__(self, device_index: int, logger):
        self.cap = cv2.VideoCapture(device_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"/dev/video{device_index}를 열 수 없습니다. "
                "'v4l2-ctl --list-devices'로 장치/인덱스를 확인하세요."
            )
        self._logger = logger
        self._frame = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        self._logger.info(f"웹캠 오픈 완료: /dev/video{device_index}")

    def _capture_loop(self):
        while not self._stop_event.is_set():
            ok, frame = self.cap.read()
            if not ok:
                self._logger.warn("웹캠 프레임 읽기 실패 - 재시도")
                time.sleep(0.1)
                continue
            with self._lock:
                self._frame = frame

    def get_color_frame(self):
        with self._lock:
            return self._frame

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        self.cap.release()


def _box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _point_in_box(px, py, box):
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def assign_baskets(detections):
    """YOLO 전체 후보 목록에서 basket 박스만 골라 좌상단(y1,x1) 오름차순으로
    정렬한 뒤 b1..b4에 순서대로 배정한다 (화면 맨 위=b1 ... 맨 아래=b4).

    반환값: {"b1": box_or_None, "b2": ..., "b3": ..., "b4": ...}, 그리고
    실제로 4개가 다 잡혔는지(valid) 여부.
    """
    basket_boxes = sorted(
        (box for label, _score, box in detections if label == "basket"),
        key=lambda box: (box[1], box[0]),
    )
    assignment = {name: None for name in BASKET_NAMES}
    for name, box in zip(BASKET_NAMES, basket_boxes):
        assignment[name] = box
    valid = len(basket_boxes) >= len(BASKET_NAMES)
    return assignment, valid


def compute_occupancy(detections, basket_assignment):
    """각 바스켓 bbox 안에 사과 클래스 검출의 중심점이 들어오는지로 점유 여부 판단."""
    apple_centers = [
        _box_center(box) for label, _score, box in detections if label in APPLE_LABELS
    ]
    occupancy = {}
    for name, basket_box in basket_assignment.items():
        if basket_box is None:
            occupancy[name] = False
            continue
        occupancy[name] = any(
            _point_in_box(cx, cy, basket_box) for cx, cy in apple_centers
        )
    return occupancy


def draw_basket_overlay(frame, basket_assignment, occupancy):
    """바스켓 bbox + b1~b4 라벨 + 점유 여부를 그린 프레임을 반환 (원본은 건드리지 않음).

    BasketDetectionNode(_publish_status)와 ROS 없이 실시간으로 로직만 확인하는
    basket_camera_live_test.py가 동일한 시각화를 공유하기 위해 분리했다.
    """
    annotated = frame.copy()
    for name, box in basket_assignment.items():
        if box is None:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        color = (0, 0, 255) if occupancy[name] else (0, 255, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        display_name = BASKET_DISPLAY_NAMES.get(name, name)
        label = f"{name}({display_name}) {'APPLE' if occupancy[name] else 'EMPTY'}"
        cv2.putText(
            annotated, label, (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )
    return annotated


class BasketDetectionNode(Node):
    """바스켓 점유 상태를 주기적으로 판정해서 /basket_status로 publish하는 노드."""

    def __init__(self):
        super().__init__('basket_detection_node')

        self.declare_parameter('device_index', 0)
        device_index = self.get_parameter('device_index').get_parameter_value().integer_value

        self.webcam = WebcamCapture(device_index, self.get_logger())
        self.model = AppleStatusModel()

        self.status_pub = self.create_publisher(String, 'basket_status', 10)
        self.debug_image_pub = self.create_publisher(Image, 'basket_camera/debug_image', 10)

        self.create_timer(STATUS_PUBLISH_PERIOD_SEC, self._publish_status)

        self.get_logger().info(
            f"BasketDetectionNode initialized (device_index={device_index})"
        )

    def _publish_status(self):
        frame = self.webcam.get_color_frame()
        if frame is None:
            return

        detections = self.model.get_all_detections(frame)
        basket_assignment, valid = assign_baskets(detections)
        occupancy = compute_occupancy(detections, basket_assignment)

        if not valid:
            self.get_logger().warn(
                f"바스켓 4개를 전부 찾지 못함 - 감지된 개수: "
                f"{sum(1 for b in basket_assignment.values() if b is not None)}/4"
            )

        payload = {name: occupancy[name] for name in BASKET_NAMES}
        payload["valid"] = valid
        payload["timestamp"] = time.time()

        self.status_pub.publish(String(data=json.dumps(payload)))
        annotated = draw_basket_overlay(frame, basket_assignment, occupancy)
        self.debug_image_pub.publish(cv2_to_imgmsg(annotated, encoding='bgr8'))


def main(args=None):
    rclpy.init(args=args)
    node = BasketDetectionNode()
    try:
        rclpy.spin(node)
    finally:
        node.webcam.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
